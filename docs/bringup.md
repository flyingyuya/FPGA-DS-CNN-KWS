# 上板指南（M7 bring-up）

全部命令行，不需要 Vivado GUI。

## 0. 一次性准备

```fish
# 构建比特流（已生成过就跳过）
vivado -mode batch -notrace -source scripts/impl.tcl      # -> report/board_top.bit

# UART 脚本依赖
cd software && uv add pyserial && cd ..
```

Linux 下普通用户访问 JTAG 需要 udev 规则，否则要 `sudo`。先用 `detect` 确认：

```fish
bash scripts/prog.sh detect
```

输出应当能看到 `IDCODE=0x0362D093 PART=xc7a100t...`。看不到就按脚本打印的三条排查
（上电 / 插的是 JTAG 口还是 UART 口 / udev 权限）。

## 1. 烧录（易失，掉电即失）

```fish
bash scripts/prog.sh                 # 烧 report/board_top.bit
bash scripts/prog.sh flash           # 写 QSPI，掉电保持（bring-up 阶段别用）
bash scripts/prog.sh /path/x.bit     # 烧指定文件
```

## 2. 看分类结果

```fish
uv run --project software python scripts/uart_rx.py          # 自动找串口
uv run --project software python scripts/uart_rx.py --raw    # 同时打原始字节
uv run --project software python scripts/uart_rx.py --selftest   # 不用硬件，验解码
```

板子每完成一次分类发 **5 个字节**：

```
0xA5  <class>  <margin_hi>  <margin_lo>  <checksum>
                                          checksum = 前四字节异或
```

`margin` 是**有符号 16 位**（第一名与第二名的 logit 差，越大越确信）；
`checksum` 不对的帧会被上位机丢弃并计数。
类别顺序：`0=silence 1=unknown 2=yes 3=no 4=up 5=down 6=left 7=right 8=on 9=off 10=stop 11=go`。

## 2.5 LED 指示灯（重要）

烧录后先看两个 LED，它们能直接告诉你卡在哪一步：

| led[0] | led[1] | 含义 | 下一步 |
|---|---|---|---|
| **不闪** | 灭/任意 | **时钟或复位没起来** | 见下面"全灭排查" |
| 闪 ~1.4 Hz | 灭 | 就绪，等按键 | 按 W21 说话 |
| 闪 ~1.4 Hz | 常亮 | 正在算（MFCC ~21 ms + 分类 1 ms） | 等 1–2 秒 |
| 闪 ~1.4 Hz | 闪一下（175 ms） | 出结果了 | 看 `uart_rx.py` 输出 |
| 闪 ~1.4 Hz | 快闪 ~5.7 Hz | **WM8960 的 I2C 没配上** | 查上拉/管脚/codec 供电 |

设计上刻意做了两件事：

* `led[0]` 是**自由运行心跳**（`clk/2^25`），与任何状态机无关 —— 它不闪就说明
  比特流没加载、MMCM 没锁、或者 `sys_rstn` 一直是低（**复位键没松开**）。
* I2C 初始化加了 **350 ms 超时**：codec 不响应时不会整机死等，而是继续走到 S_KEY，
  同时用 `led[1]` 快闪报警。这样即使音频侧坏了，按键路径仍然可用、可观测。

## 2.9 两个按键

| 按键 | 管脚 | 作用 |
|---|---|---|
| KEY1 | W21 | 单次识别：按一下采 1 秒、出一次结果 |
| **KEY2** | **Y21** | **切换 单次 / 连续识别** |

**连续模式**：加速器一算完，下一个 1 秒窗口立刻开始——相当于一直在识别，每秒一次判决。
中间只丢加速器那 0.56 ms 的音频（占窗口的 0.06%，听不出来），**而且不需要额外存储**，
因为前端只保留一个窗口的历史。

上电默认是单次模式（UART 不会刷屏）；按一下 KEY2 进入连续模式，`led[1]` 会慢闪提示。

实时识别的另一种做法是"每 100 ms 滑窗"，但那需要窗口重叠 + FMEM 环形列寻址，
所以本设计选了"每秒一整窗"：人感受不到 1 秒的节奏，资源却是最省的。

## 3. 分层调试顺序（重要）

一次全上会把问题混在一起。按这个顺序，每步都有明确的"好"的判据：

| 步骤 | 判据 | 失败先查 |
|---|---|---|
| ① 烧录 | `prog.sh` 打印 `✅ programmed` | JTAG 权限 / 线 |
| ② I2C 配置 | 板子 LED 行为稳定；如有条件测 `aud_scl` 有 33 次突发 | 上拉电阻、器件地址 0x34 |
| ③ I2S 收到数据 | `uart_rx.py` 按键后 ~1 s 内有输出 | BCLK/LRCLK 极性、MCLK 悬空是否可接受 |
| ④ MFCC 完成 | 按键后约 21 ms 出结果 | 前端输入幅度太小（麦克风增益） |
| ⑤ 分类正确 | 说 "yes" 输出 class 2 | 麦克风位置/增益、是否录到 1 s 完整语音 |

`margin` 现在发完整 16 位（第一版只发低 8 位，超过 255 会回绕、负值变成大正数，
已修正）。

## 4. 时序/资源现状（本次构建）

| 指标 | 数值 |
|---|---|
| LUT / BRAM / DSP | 9,101 (14.4%) / 24.5 (18.2%) / 17 (7.1%) |
| 功耗 | 0.248 W |
| WNS @48 MHz | +3.751 ns，0 失败端点 |
| `aud_bclk` 域 | WNS +974 ns（1.024 MHz） |
| MFCC | 768,913 cyc = 16.0 ms / 整图 |
| CNN | 27,035 cyc = **0.563 ms** |
| 精度 | 93.37% test（20 mel 带） |

## 5. 时钟与管脚（来自板卡 JM2 连接器表格）

| 信号 | 管脚 | 备注 |
|---|---|---|
| `sys_clk_p/n` | R4 / T4 | 200 MHz 差分，MMCM ×3.0/12.5 = 48.000 MHz |
| `sys_rstn` | R14 | |
| `key` | W21 | 高有效，已同步 + 边沿检测 |
| `led[1:0]` | W22 / Y22 | `led[0]`=busy，`led[1]`=新结果 |
| `uart_tx` | P15 | 115200 8N1，只发不收 |
| `aud_bclk` | J20 | WM8960 输出，1.024 MHz（16 kHz × 64） |
| `aud_lrclk` | K18 | 16 kHz |
| `aud_adcdat` | H20 | |
| `aud_scl/sda` | K21 / K22 | I2C，模块自带上拉 |
