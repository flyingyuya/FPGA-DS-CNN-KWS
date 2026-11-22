#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
uart_rx.py -- 读板子发回来的分类结果（115200 8N1）。

协议（见 rtl/board_top.v 的 result reporting）：
    每次分类完成，板子连发 5 个字节：
        0xA5   <class>   <margin_hi>   <margin_lo>   <checksum>
        checksum = 前四个字节的异或
    class  : 0=silence 1=unknown 2=yes 3=no 4=up 5=down 6=left 7=right
             8=on 9=off 10=stop 11=go
    margin : 第一名与第二名的 logit 差，**有符号 16 位**（越大越确信）
    checksum 不对的帧直接丢弃并计数——3 字节无校验时，margin 恰好等于 0xA5
    会被误当成帧头，现在不会了

用法：
    ./scripts/uart_rx.py                  # 自动找板子的串口，一直打印
    ./scripts/uart_rx.py -p /dev/ttyUSB0
    ./scripts/uart_rx.py --raw            # 同时打印原始字节（调试用）
    ./scripts/uart_rx.py --selftest       # 不用硬件，验证解码逻辑
"""
import argparse
import sys
import time

LABELS = ["silence", "unknown", "yes", "no", "up", "down",
          "left", "right", "on", "off", "stop", "go"]

SYNC = 0xA5
FRAME = 5


def decode(stream):
    """把字节流解成 (class, margin, ok) 列表；能容忍杂字节和坏帧。

    返回的第三项是校验结果：False 表示这一帧的 checksum 不对（被丢弃前先报出来，
    便于发现链路问题）。调用方负责把 False 的帧计入 dropped。
    """
    out, i = [], 0
    while i + FRAME <= len(stream):
        if stream[i] == SYNC:
            b = stream[i:i + FRAME]
            ok = (b[0] ^ b[1] ^ b[2] ^ b[3]) == b[4]
            margin = ((b[2] << 8) | b[3]) & 0xFFFF
            if margin & 0x8000:
                margin -= 0x10000          # signed INT16
            out.append((b[1], margin, ok))
            i += FRAME
        else:
            i += 1
    return out


def find_port():
    """按 VID/PID 找板子的 USB 串口（FTDI / Digilent / Xilinx）。"""
    try:
        from serial.tools import list_ports
    except ImportError:
        return None
    known = {0x0403, 0x03FD, 0x6014, 0x10C4}  # FTDI, Xilinx, Digilent, CP210x
    usb = [p for p in list_ports.comports() if p.vid is not None]
    for p in usb:
        if p.vid in known:
            return p.device
    # only USB adapters: /dev/ttyS* are motherboard ports and will open fine but
    # never see the board, which is a confusing failure mode
    return usb[0].device if usb else None


def selftest():
    # a clean frame, a frame preceded by noise, and a truncated tail
    stream = [0x00, 0xFF, SYNC, 2, 82, SYNC, 10, 7, 0x13, SYNC, 5]
    got = decode(stream)
    def mk(cls, m):
        hi, lo = (m >> 8) & 0xFF, m & 0xFF
        return [SYNC, cls, hi, lo, SYNC ^ cls ^ hi ^ lo]

    stream = [0x00, 0xFF] + mk(2, 85) + mk(10, -7) + [SYNC, 5, 0, 0, 0x00]
    got = decode(stream)
    want = [(2, 85, True), (10, -7, True)]
    ok = got[:2] == want and got[2][2] is False
    print(f"  selftest: decoded {got} (expected {want}) -> {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--port", help="serial device (default: auto-detect)")
    ap.add_argument("-b", "--baud", type=int, default=115200)
    ap.add_argument("--raw", action="store_true", help="also dump raw bytes")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    try:
        import serial
    except ImportError:
        print("需要 pyserial：uv add pyserial  或  pip install pyserial", file=sys.stderr)
        return 1

    port = args.port or find_port()
    if not port:
        print("找不到串口。插上板子的 USB 后重试，或用 -p /dev/ttyUSB0 指定。",
              file=sys.stderr)
        return 1

    print(f"  打开 {port} @ {args.baud} 8N1，等分类结果（Ctrl-C 退出）…")
    buf = bytearray()
    n = 0
    dropped = 0
    t0 = None
    try:
        ser = serial.Serial(port, args.baud, timeout=0.2)
    except Exception as e:
        print(f"打不开 {port}: {e}", file=sys.stderr)
        try:
            from serial.tools import list_ports
            print("  可用串口：", file=sys.stderr)
            for p in list_ports.comports():
                print(f"    {p.device:20s} {p.description}", file=sys.stderr)
        except Exception:
            pass
        return 1

    with ser:
        try:
            while True:
                chunk = ser.read(64)
                if not chunk:
                    continue
                if args.raw:
                    print("  raw:", " ".join(f"{b:02x}" for b in chunk))
                buf += chunk
                for cls, margin, ok in decode(buf):
                    if not ok:
                        dropped += 1
                        # say it out loud: "frames arrive but fail the checksum"
                        # is a completely different fault from "no bytes at all"
                        print(f"  [drop {dropped}] bad frame: "
                              + " ".join(f"{b:02x}" for b in [SYNC, cls, margin >> 8 & 0xFF, margin & 0xFF]))
                        continue
                    n += 1
                    now = time.time()
                    dt = "" if t0 is None else f"  (+{now - t0:6.2f}s)"
                    t0 = now
                    name = LABELS[cls] if cls < len(LABELS) else f"?{cls}"
                    print(f"  [{n:3d}] class {cls:2d} = {name:<8s} margin {margin:3d}{dt}")
                # drop whole bytes we have already consumed, keep a partial tail
                if len(buf) >= FRAME:
                    del buf[:len(buf) - (len(buf) % FRAME)]
        except KeyboardInterrupt:
            print(f"\n  收到 {n} 个结果，丢弃 {dropped} 个坏帧，退出。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
