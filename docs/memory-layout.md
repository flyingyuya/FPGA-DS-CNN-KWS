# 加速器存储布局与数据流契约（RTL 规格）

本文件是 RTL 的唯一地址/格式规格，所有内容由 `sim/reference_model.py` 的
`replay()` 逐位验证过：`software/int_model.py`（张量实现）与 `replay()`（按本文件
的地址规则读打包镜像）在测试集上**逐位相同**。

生成方式：`uv run python software/export_mem.py --install`。

---

## 1. 全局常量

| 项 | 值 |
|---|---|
| LANES（并行通道/输出通道数） | 32 |
| 激活码 / 权重码位宽 | 8 / 8 bit |
| 累加器 / bias 位宽 | 24 bit 有符号 |
| 移位字段位宽 | 4 bit（0..15）|
| 乘积位宽 | 16 bit 有符号（`127×255 = 32385 < 2^15`）|
| 激活 bank 字宽 × 深度 | 256 bit × 1024 |
| 权重 ROM 字宽 × 深度 | 256 bit × 1024（实用 552）|
| 通道常量总数 | 492（bias 24 bit + shift 4 bit）|
| 输入 | 40×101 UINT8，4040 字节 |
| 输出 | 12 路 logit 码 + 类别 |

## 2. 层表（顺序即 FSM 顺序）

| 层 | 类型 | 通道 | 输入→输出 | stride | NG | 像素 | 输出字数 | wrom 基址+长度 | ch 基址+长度 |
|---|---|---|---|---|---|---|---|---|---|
| stem | std | 1→32 | 40×101→20×51 | 2 | 1 | 1020 | 1020 | 0+9 | 0+32 |
| dw1 | dw | 32→32 | 20×51→10×26 | 2 | 1 | 260 | 260 | 9+9 | 32+32 |
| pw1 | pw | 32→48 | 10×26→10×26 | 1 | 2 | 260 | 520 | 18+64 | 64+48 |
| dw2 | dw | 48→48 | 10×26→5×13 | 2 | 2 | 65 | 130 | 82+18 | 112+48 |
| pw2 | pw | 48→64 | 5×13→5×13 | 1 | 2 | 65 | 130 | 100+96 | 160+64 |
| dw3 | dw | 64→64 | 5×13→5×13 | 1 | 2 | 65 | 130 | 196+18 | 224+64 |
| pw3 | pw | 64→64 | 5×13→5×13 | 1 | 2 | 65 | 130 | 214+128 | 288+64 |
| dw4 | dw | 64→64 | 5×13→5×13 | 1 | 2 | 65 | 130 | 342+18 | 352+64 |
| pw4 | pw | 64→64 | 5×13→5×13 | 1 | 2 | 65 | 130 | 360+128 | 416+64 |
| gap | gap | 64→64 | 5×13→1×1 | 1 | 2 | 1 | 2 | 488+0 | 480+0 |
| fc | fc | 64→12 | 1×1→1×1 | 1 | 1 | 1 | 1 | 488+64 | 480+12 |

`NG = ceil(cout/32)`，`输出字数 = 像素 × NG`。

## 3. 存储格式

### 3.1 激活 bank（两块乒乓，各 1024 字）

```
一个字 = 32 个通道 × 8 bit，LSB 起第 i 个字节 = 通道 i
word_addr = pixel * NG + group
byte_lane = c mod 32
```

- 写：本层一个 job 的 32 个输出通道写成一个字；通道数不是 32 的倍数时，
  **第二个字的无效 lane 必须写 0**（`_clip_lanes` 的契约）。
- 读：见 §4 的三种数据流。
- 越界（padding）读回：**stem 层返回 `KWS_ZP_IN`，其余层返回 0**（定理 3 的前提）。

### 3.2 权重 ROM（`mem/weights/w_all.mem`）

每行 64 个 hex 字符 = 256 bit。按 Verilog `.mem` 的惯例**高位在前**：行首 2 个字符是
lane 31，行尾 2 个字符是 lane 0（RTL 切片 `word[i*8 +: 8]` = lane i）。lane 的含义按层类型不同：

| 层类 | 地址公式 | lane 的含义 |
|---|---|---|
| std | `base + ic*9 + tap` | 输出通道 |
| dw | `base + g*9 + tap` | 组 g 内的通道 `g*32+lane` |
| pw | `base + g*cin + ic` | 输出通道 `g*32+lane` |
| fc | `base + ic` | 类别 |

`tap = ky*3 + kx`（ky 沿特征图行 = 频率轴，kx 沿列 = 时间轴）。
通道数不是 32 倍数的层，无效 lane 填 0。

### 3.3 通道常量 ROM（`mem/bias/b_all.mem`）

```
32 bit = {4'b0, shift[3:0], bias[23:0]}     地址 = 绝对通道号
```

`bias` 已折入输入零点（`−zp·Σq_w`）和舍入常数（`+2^(shift−1)`），
所以 RTL 的 `>>>` 直接就是"四舍五入"。共 492 个字。

每层开始时按 `ch_base` 起 **串行装载 32 个 lane 的 24+4 bit 寄存器**
（每层 ≤32 拍，可忽略）。装载一次的粒度是 `(layer, channel-group)`，
因为下面 §4 的循环顺序把 group 放在最外层。

## 4. 三种数据流（都是 1 MAC/lane/cycle）

### 4.1 std（stem：cin=1，lane = 输出通道）
```
for oy, ox:                      # 每个输出像素一个 job
    for ic, tap:                 # cin*9 拍；首拍 acc = bias + w*a（不是只有 bias）
        iy = oy*stride + ky - pad ; ix = ox*stride + kx - pad
        a  = in_range ? FMEM[iy*101+ix] : ZP_IN     # 标量广播到 32 lane
        w  = wrom[base + ic*9 + tap]                     # 256 bit
        acc += w * a
    写 bank_dst[pix*NG + g] = clamp(acc >>> shift)
```

### 4.2 dw（lane = 通道）
```
for g, oy, ox:                   # group 在最外层，bias 只装一次
    for tap:                     # 9 拍；首拍 acc = bias + w*a
        iy, ix = ...
        a = in_range ? bank_src[(iy*wi+ix)*NG + g] : 0    # 整字读，32 通道
        w = wrom[base + g*9 + tap]
        acc += w * a             # 逐 lane 相乘，无需跨 lane 归约
    写 bank_dst[pix*NG + g] = clamp(acc >>> shift)
```

### 4.3 pw（lane = 输出通道，激活广播）
```
for g, oy, ox:
    acc = bias[c0:c0+32]
    for ic:                      # cin 拍
        a = bank_src[pix*NG_in + ic/32][ic%32]            # 单字节广播
        w = wrom[base + g*cin + ic]                       # 256 bit
        acc += w * a
    写 bank_dst[pix*NG + g] = clamp(acc >>> shift)
```
PW 占全网 70% 的 MAC，
这条顺序把它打到 100% lane 利用率。

### 4.4 gap
```
for pix in range(65): for g in range(2):
    total[g*32 : g*32+32] += bank_src[pix*2 + g]
code = clamp((total + 2^5) >>> 6, 0, 255)      # 64 个码，存 feature file
```
`65` 不是 2 的幂，但 `s_gap = s_in · 2^6/65 = s_in·64/65`
这个常数因子由 FC 的 `rho` 自动吸收，硬件只做移位。

### 4.5 fc（lane = 类别，只用到 12 lane）
```
acc = bias[ch_base : ch_base+12]  # 首拍同样与 w*a 相加
for ic in range(64):
    acc += wrom[base + ic][:12] * gap_code[ic]
logit_oc = acc >>> shift_oc        # 有符号，不饱和
```
输出 `argmax(logit)` + `max − second_max`（margin）。不做 softmax。

## 5. 输出级（每 lane 一个）

```
out = clamp( acc >>> shift , 0, 255 )      # 算术右移 + ReLU 饱和
```
- `acc` 是 24 bit 有符号，`shift` 是 4 bit（每通道）
- 移位对负数也是 floor，这是有符号算术右移的语义
- 通道数不是 32 倍数时，无效 lane 写 0（bank 图像可确定性对拍）

## 6. 周期预算（验收基准）

| 层 | 输出像素 | NG | job 数 | 每 job 拍数 | 总拍数 |
|---|---|---|---|---|---|
| stem | 1020 | 1 | 1020 | 9 | 9180 |
| dw1 | 260 | 1 | 260 | 9 | 2340 |
| pw1 | 260 | 2 | 520 | 32 | 16640 |
| dw2 | 65 | 2 | 130 | 9 | 1170 |
| pw2 | 65 | 2 | 130 | 48 | 6240 |
| dw3 | 65 | 2 | 130 | 9 | 1170 |
| pw3 | 65 | 2 | 130 | 64 | 8320 |
| dw4 | 65 | 2 | 130 | 9 | 1170 |
| pw4 | 65 | 2 | 130 | 64 | 8320 |
| gap | 1 | 2 | 2 | 65 | 130 |
| fc | 1 | 1 | 1 | 64 | 64 |

合计 **54,744** 拍（11 层）→ 48 MHz 下 **1.141 ms**/帧（帧跳距 10 ms，余量 8.86 ms）。
双累加器（SEL=0/1）把每个 job 的 drain 藏在下一个 job 的累加里，所以稳态
吞吐正好是 1 累加/拍，job 之间没有气泡。

## 7. 黄金数据与测试文件

| 文件 | 用途 |
|---|---|
| `mem/weights/w_all.mem` | 共享权重 ROM 镜像（1024 行，实用 {sum(l['wrom_words'] for l in L)} 行）|
| `mem/bias/b_all.mem` | `{{shift, bias}}` 通道常量 ROM（{rm.CH_TOTAL} 字）|
| `mem/input/mfcc_<vec>.mem` | 14 个输入向量（12 类各一 + 全 0 + 全 255），{rm.H_IN*rm.W_IN} 行 |
| `golden/act_<layer>_<vec>.mem` | 层级黄金：该层 bank 的前 `输出字数` 个字（2 个向量）|
| `golden/golden.json` | 14 个向量的 logits 码 + 类别 |
| `golden/unit/pe_*.mem` | PE 阵列单元向量：31 个 job × 9 taps，含 shift=0/15、饱和、负值边界 |
| `golden/unit/argmax_*.mem` | argmax 树向量（含并列、全负）|
| `docs/quant_summary.csv` | 逐层契约表（shift 范围、max|q|、累加器位数）|
