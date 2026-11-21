"""sim/reference_model.py -- single source of truth for the accelerator.

Two jobs, both numpy-only so the file stays usable without torch:

  1. `replay()`  a hardware-exact simulation: it walks the *packed memory
     images* with the same word/lane addressing the RTL uses, then produces the
     golden activations.  If this and `software/int_model.py` agree bit for bit,
     the memory layout and the numeric contract are consistent and the RTL has
     an unambiguous spec.

  2. emitters:  `rtl/kws_params.vh`, the `.mem` images, the golden vectors and
     the unit-test vectors.

Memory layout (everything derives from these three rules):

    activation bank   word = 32 channels x 8 bit
                      word_addr = pixel * NG + group,  byte_lane = c % 32
                      NG = ceil(C / 32)
    weight ROM        word = 32 lanes x 8 bit, lane meaning depends on the layer
                      std : address = base + ic*9 + tap,  lane = output channel
                      dw  : address = base + g*9 + tap,   lane = channel in group g
                      pw  : address = base + g*cin + ic,  lane = output channel
                      fc  : address = base + ic,          lane = class
    bias ROM          word = {4'b0, shift[3:0], bias[23:0]}, address = channel

Weight and bias words are written most-significant-lane-last: byte i of the hex
string is lane i.
"""

import json
import os
import sys

import numpy as np

# The geometry must follow the training front end, not a literal: this file used
# to hardcode 40x101 and raised IndexError as soon as N_MELS changed.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "software"))
import config as _C                                                     # noqa: E402

# ---------------------------------------------------------------------------
# global widths -- must equal the RTL parameters
# ---------------------------------------------------------------------------
LANES = 32
ACT_BITS = 8
W_BITS = 8
ACC_BITS = 24
SHIFT_BITS = 4
ACT_QMAX = (1 << ACT_BITS) - 1
W_QMAX = (1 << (W_BITS - 1)) - 1
SHIFT_MAX = (1 << SHIFT_BITS) - 1
# Bank depth follows the largest layer output, rounded up to a power of two.
# It used to be hardcoded 1024 (sized for the 40x101 model); with a 20x101 input
# the largest layer is the stem output 10x51 = 510, so 512 words suffice and the
# activation buffer halves in BRAM.  This is where the memory saving comes from,
# so it must be derived, not literal.

NUM_CLASSES = _C.NUM_CLASSES
H_IN, W_IN = _C.H_IN, _C.W_IN
GAP_SHIFT = 6
LABELS = ["_silence_", "_unknown_", "yes", "no", "up", "down",
          "left", "right", "on", "off", "stop", "go"]

# ---------------------------------------------------------------------------
# layer table: (name, kind, cin, cout, hi, wi, ho, wo, stride, pad)
# kind: std = 3x3 standard conv, dw = 3x3 depthwise, pw = 1x1, gap, fc
# ---------------------------------------------------------------------------
_ARCH = [
    # (name, kind, cin, cout, stride, pad) -- spatial sizes are derived below
    ("stem", "std", 1, 32, 2, 1),
    ("dw1", "dw", 32, 32, 2, 1),
    ("pw1", "pw", 32, 32, 1, 0),
    ("dw2", "dw", 32, 32, 2, 1),
    ("pw2", "pw", 32, 64, 1, 0),
    ("dw3", "dw", 64, 64, 1, 1),
    ("pw3", "pw", 64, 64, 1, 0),
    ("dw4", "dw", 64, 64, 1, 1),
    ("pw4", "pw", 64, 64, 1, 0),
    ("gap", "gap", 64, 64, 1, 0),
    ("fc", "fc", 64, NUM_CLASSES, 1, 0),
]


def _derive_spec():
    h, w, out = H_IN, W_IN, []
    for (name, kind, cin, cout, stride, pad) in _ARCH:
        if kind in ("std", "dw"):
            ho = (h + 2 * pad - 3) // stride + 1
            wo = (w + 2 * pad - 3) // stride + 1
        else:                       # pw keeps the size, gap/fc collapse to 1x1
            ho, wo = (h, w) if kind == "pw" else (1, 1)
        out.append((name, kind, cin, cout, h, w, ho, wo, stride, pad))
        h, w = ho, wo
    return out


_SPEC = _derive_spec()

# the bank stores a layer's OUTPUT; the stems' 20x101 is the FMEM, not a bank
BANK_WORDS = 1 << max(1, int(np.ceil(np.log2(
    max(l[6] * l[7] for l in _SPEC)))))


def _ng(c):
    return (c + LANES - 1) // LANES


def table():
    """Layer table with the derived fields (groups, ROM/channel bases, ...)."""
    out, wbase, cbase = [], 0, 0
    for (name, kind, cin, cout, hi, wi, ho, wo, stride, pad) in _SPEC:
        ng = _ng(cout) if kind != "fc" else 1
        if kind == "std":
            nw = cin * 9
        elif kind == "dw":
            nw = ng * 9
        elif kind == "pw":
            nw = ng * cin
        elif kind == "fc":
            nw = cin
        else:
            nw = 0
        out.append(dict(name=name, kind=kind, cin=cin, cout=cout, hi=hi, wi=wi,
                        ho=ho, wo=wo, stride=stride, pad=pad, ng=ng,
                        pixels=ho * wo, ng_in=_ng(cin),
                        wrom_base=wbase, wrom_words=nw,
                        ch_base=cbase, ch_words=cout,
                        out_words=ho * wo * ng))
        wbase += nw
        cbase += cout
    return out


LAYERS = table()


def _wrom_depth():
    """Weight-ROM depth: the next power of two above what the network needs.

    Depth is a hard BRAM granularity.  At 552 words the ROM had to be 1024 deep
    (8 x BRAM36); narrowing block 1 took it to 479, which fits 512 (4 x BRAM36).
    Deriving the depth from the layer table is what turns that into a real saving.
    """
    need = sum(l["wrom_words"] for l in LAYERS)
    d = 512
    while d < need:
        d *= 2
    return d


WROM_WORDS = _wrom_depth()
CH_TOTAL = LAYERS[-1]["ch_base"] + LAYERS[-1]["ch_words"]
GAP = LAYERS[-2]
GAP_N = GAP["hi"] * GAP["wi"]


# ---------------------------------------------------------------------------
# packing helpers
# ---------------------------------------------------------------------------
def hexbytes(b):
    """Verilog `.mem` word of a 32-lane array, big-endian like $readmemh wants it.

    The *first* two characters of the line are the most significant byte, i.e.
    lane LANES-1; the last two are lane 0.  The RTL slices lane l as
    `word[l*8 +: 8]` (LSB first), so the byte order in the file has to be
    reversed relative to the lane order.  Getting this backwards silently swaps
    lane 0 with lane 31 -- which is exactly what the PE testbench caught.
    """
    return "".join(f"{int(v) & 0xFF:02x}" for v in b[::-1])


def unhexbytes(s, n):
    return np.frombuffer(bytes.fromhex(s.strip()), dtype=np.uint8)[:n].astype(np.int64)


def pack_weights(hw):
    """(WROM_WORDS, LANES) int8 weight codes, exactly as the RTL addresses them."""
    rom = np.zeros((WROM_WORDS, LANES), dtype=np.int8)
    for L in LAYERS:
        n = L["name"]
        if "q" not in hw.get(n, {}):      # the pool has no weights
            continue
        q = np.asarray(hw[n]["q"])
        base = L["wrom_base"]
        if L["kind"] == "std":
            for ic in range(L["cin"]):
                for tap in range(9):
                    row = rom[base + ic * 9 + tap]
                    for lane in range(LANES):
                        if lane < L["cout"]:
                            row[lane] = q[lane, ic, tap // 3, tap % 3]
        elif L["kind"] == "dw":                      # q: (cout, 1, 3, 3)
            for g in range(L["ng"]):
                for tap in range(9):
                    row = rom[base + g * 9 + tap]
                    for lane in range(LANES):
                        oc = g * LANES + lane
                        if oc < L["cout"]:
                            row[lane] = q[oc, 0, tap // 3, tap % 3]
        elif L["kind"] == "pw":                      # q: (cout, cin)
            for g in range(L["ng"]):
                for ic in range(L["cin"]):
                    row = rom[base + g * L["cin"] + ic]
                    for lane in range(LANES):
                        oc = g * LANES + lane
                        if oc < L["cout"]:
                            row[lane] = q[oc, ic]
        elif L["kind"] == "fc":                      # q: (12, 64)
            for ic in range(L["cin"]):
                row = rom[base + ic]
                for lane in range(NUM_CLASSES):
                    row[lane] = q[lane, ic]
    return rom


def pack_bias(hw):
    """(CH_TOTAL,) int64 bias and (CH_TOTAL,) int64 shift, in ROM word order."""
    bias = np.zeros(CH_TOTAL, dtype=np.int64)
    shift = np.zeros(CH_TOTAL, dtype=np.int64)
    for L in LAYERS:
        n = L["name"]
        if n not in hw:
            continue
        if L["kind"] == "gap":
            # The global average pool is a depthwise pass with unit weights, so
            # it uses the same PE and the same constant ROM: bias = 2**(G-1)
            # makes the folded shift a round-half-up of sum/2**G.
            bias[L["ch_base"]:L["ch_base"] + L["ch_words"]] = 1 << (GAP_SHIFT - 1)
            shift[L["ch_base"]:L["ch_base"] + L["ch_words"]] = GAP_SHIFT
            continue
        b = np.asarray(hw[n]["B"], dtype=np.int64)
        s = np.asarray(hw[n]["shift"], dtype=np.int64)
        bias[L["ch_base"]:L["ch_base"] + L["ch_words"]] = b
        shift[L["ch_base"]:L["ch_base"] + L["ch_words"]] = s
    return bias, shift


def bias_word(b, s):
    """32-bit bias ROM word: {4'b0, shift, bias[23:0]}."""
    return ((int(s) & SHIFT_MAX) << 24) | (int(b) & 0xFFFFFF)


# ---------------------------------------------------------------------------
# hardware-exact replay
# ---------------------------------------------------------------------------
def _tap_xy(tap):
    return tap // 3, tap % 3


def _clip_lanes(acc, shift, n_valid):
    """shift + ReLU clamp, and force lanes outside [0, n_valid) to 0.

    A 48-channel layer only fills 16 lanes of its second word; zeroing the rest
    keeps the bank images (and therefore the golden vectors) deterministic.
    """
    out = np.clip(acc >> shift, 0, ACT_QMAX).astype(np.uint8)
    if n_valid < LANES:
        out[max(0, n_valid):] = 0
    return out


def replay(hw, mfcc):
    """Run one UINT8 MFCC frame through the packed images, RTL addressing.

    Returns dict(layers={name: bank snapshot}, logits=(12,), cls=int).
    """
    rom = pack_weights(hw)
    bias, shift = pack_bias(hw)
    zp_in = int(hw["stem"].get("zp_in", 0))

    banks = [np.zeros((BANK_WORDS, LANES), dtype=np.uint8) for _ in range(2)]
    snap = {}

    def read_word(bank, pix, group, ng):
        return bank[pix * ng + group].astype(np.int64)

    for idx, L in enumerate(LAYERS):
        n, kind = L["name"], L["kind"]
        if kind in ("std", "dw"):
            src = None if kind == "std" else banks[(idx - 1) % 2]
            dst = banks[idx % 2]
            if kind == "std":
                dst = banks[0]
            for g in range(L["ng"]):
                c0 = L["ch_base"] + g * LANES
                for oy in range(L["ho"]):
                    for ox in range(L["wo"]):
                        acc = bias[c0:c0 + LANES].copy()
                        # std walks input channels, dw has none (one tap per channel)
                        for ic in range(L["cin"] if kind == "std" else 1):
                            for tap in range(9):
                                ky, kx = _tap_xy(tap)
                                iy = oy * L["stride"] + ky - L["pad"]
                                ix = ox * L["stride"] + kx - L["pad"]
                                inside = 0 <= iy < L["hi"] and 0 <= ix < L["wi"]
                                if kind == "std":
                                    a = int(mfcc[iy, ix]) if inside else zp_in
                                    w = rom[L["wrom_base"] + ic * 9 + tap]
                                    acc += w.astype(np.int64) * a
                                else:
                                    a = read_word(src, iy * L["wi"] + ix, g, L["ng_in"]) \
                                        if inside else np.zeros(LANES, dtype=np.int64)
                                    w = rom[L["wrom_base"] + g * 9 + tap]
                                    acc += w.astype(np.int64) * a
                        dst[(oy * L["wo"] + ox) * L["ng"] + g] = _clip_lanes(
                            acc, shift[c0:c0 + LANES], L["cout"] - g * LANES)
            snap[n] = dst.copy()
        elif kind == "pw":
            src = banks[(idx - 1) % 2]
            dst = banks[idx % 2]
            for g in range(L["ng"]):
                c0 = L["ch_base"] + g * LANES
                for pix in range(L["pixels"]):
                    acc = bias[c0:c0 + LANES].copy()
                    for ic in range(L["cin"]):
                        a = int(src[pix * L["ng_in"] + ic // LANES][ic % LANES])
                        w = rom[L["wrom_base"] + g * L["cin"] + ic]
                        acc += w.astype(np.int64) * a
                    dst[pix * L["ng"] + g] = _clip_lanes(
                        acc, shift[c0:c0 + LANES], L["cout"] - g * LANES)
            snap[n] = dst.copy()
        elif kind == "gap":
            src = banks[(idx - 1) % 2]
            total = np.zeros(L["cin"], dtype=np.int64)
            # the pool reads the *source* tensor, so use hi*wi (the gap row in
            # the table has ho=wo=1, which is the pooled output)
            for pix in range(L["hi"] * L["wi"]):
                for g in range(_ng(L["cin"])):
                    total[g * LANES:(g + 1) * LANES] += read_word(src, pix, g, _ng(L["cin"]))
            code = np.clip((total + 2 ** (GAP_SHIFT - 1)) >> GAP_SHIFT, 0, ACT_QMAX)
            snap[n] = code.astype(np.uint8)
        elif kind == "fc":
            c0 = L["ch_base"]
            acc = bias[c0:c0 + NUM_CLASSES].copy()
            feat = snap["gap"].astype(np.int64)
            for ic in range(L["cin"]):
                w = rom[L["wrom_base"] + ic]
                acc += w[:NUM_CLASSES].astype(np.int64) * int(feat[ic])
            logits = acc >> shift[c0:c0 + NUM_CLASSES]
            snap[n] = logits.astype(np.int64)
    logits = snap["fc"]
    return dict(layers=snap, logits=logits, cls=int(np.argmax(logits)))


# ---------------------------------------------------------------------------
# emitters
# ---------------------------------------------------------------------------
def emit_vh(hw, path):
    L = {l["name"]: l for l in LAYERS}
    lines = [
        "// ---------------------------------------------------------------",
        "// kws_params.vh -- GENERATED by sim/reference_model.py. Do not edit.",
        "// Geometry, widths and ROM bases for the shift-only DS-CNN accelerator.",
        "// ---------------------------------------------------------------",
        "`ifndef KWS_PARAMS_VH", "`define KWS_PARAMS_VH", "",
        "// ---- global widths ----",
        f"`define KWS_ACT_BITS     {ACT_BITS}",
        f"`define KWS_W_BITS       {W_BITS}",
        f"`define KWS_LANES        {LANES}",
        f"`define KWS_ACT_WORD     {LANES * ACT_BITS}",
        f"`define KWS_WGT_WORD     {LANES * W_BITS}",
        f"`define KWS_ACC_BITS     {ACC_BITS}",
        f"`define KWS_SHIFT_BITS   {SHIFT_BITS}",
        f"`define KWS_NUM_CLASSES  {NUM_CLASSES}",
        f"`define KWS_NUM_LAYERS   {len(LAYERS)}", "",
        "// ---- input tensor (MFCC) ----",
        f"`define KWS_H_IN         {H_IN}",
        f"`define KWS_W_IN         {W_IN}",
        f"`define KWS_IN_PIXELS    {H_IN * W_IN}",
        f"`define KWS_IN_ADDR_W    {int(np.ceil(np.log2(H_IN * W_IN)))}",   # 4040 -> 12
        f"`define KWS_ZP_IN        {int(hw['stem'].get('zp_in', 0))}", "",
        "// ---- memories ----",
        f"`define KWS_BANK_WORDS   {BANK_WORDS}",
        f"`define KWS_BANK_ADDR_W  {int(np.ceil(np.log2(BANK_WORDS)))}",
        f"`define KWS_WROM_WORDS   {WROM_WORDS}",
        f"`define KWS_WROM_ADDR_W  {int(np.ceil(np.log2(WROM_WORDS)))}",
        f"`define KWS_CH_TOTAL     {CH_TOTAL}",
        f"`define KWS_CH_ADDR_W    {int(np.ceil(np.log2(CH_TOTAL)))}", "",
        "// ---- layer ids (order of the top-level FSM) ----",
    ]
    for i, l in enumerate(LAYERS):
        lines.append(f"`define KWS_L_{l['name'].upper():6s} {i}")
    lines += ["", "// ---- global average pool ----",
              f"`define KWS_GAP_N        {GAP_N}",
              f"`define KWS_GAP_SHIFT    {GAP_SHIFT}", ""]
    for l in LAYERS:
        n = l["name"].upper()
        lines += [
            f"// ---- {l['name']} ({l['kind']}) ----",
            f"`define KWS_{n}_KIND_{l['kind'].upper():4s} 1",
            f"`define KWS_{n}_CIN      {l['cin']}",
            f"`define KWS_{n}_COUT     {l['cout']}",
            f"`define KWS_{n}_HI       {l['hi']}",
            f"`define KWS_{n}_WI       {l['wi']}",
            f"`define KWS_{n}_HO       {l['ho']}",
            f"`define KWS_{n}_WO       {l['wo']}",
            f"`define KWS_{n}_STRIDE   {l['stride']}",
            f"`define KWS_{n}_PAD      {l['pad']}",
            f"`define KWS_{n}_NG       {l['ng']}",
            f"`define KWS_{n}_PIXELS   {l['pixels']}",
            f"`define KWS_{n}_WROM_BASE {l['wrom_base']}",
            f"`define KWS_{n}_CH_BASE  {l['ch_base']}",
            f"`define KWS_{n}_OUT_WORDS {l['out_words']}", ""]
    lines.append("`endif // KWS_PARAMS_VH")
    _write(path, "\n".join(lines) + "\n")


def emit_roms(hw, root):
    os.makedirs(os.path.join(root, "mem", "weights"), exist_ok=True)
    os.makedirs(os.path.join(root, "mem", "bias"), exist_ok=True)
    rom = pack_weights(hw)
    bias, shift = pack_bias(hw)
    _write(os.path.join(root, "mem", "weights", "w_all.mem"),
           "".join(hexbytes(row) + "\n" for row in rom))
    _write(os.path.join(root, "mem", "bias", "b_all.mem"),
           "".join(f"{bias_word(bias[i], shift[i]):08x}\n" for i in range(CH_TOTAL)))
    return rom, bias, shift


def emit_inputs(frames, root):
    """frames: {name: (40,101) uint8}"""
    d = os.path.join(root, "mem", "input")
    os.makedirs(d, exist_ok=True)
    for name, f in frames.items():
        _write(os.path.join(d, f"mfcc_{name}.mem"),
               "".join(f"{int(v):02x}\n" for v in np.asarray(f).reshape(-1)))


def emit_golden(hw, frames, root, act_vectors=2):
    """End-to-end golden for every vector; per-layer activations for a subset
    (the stem bank alone is 1020 words, so dumping all 11 layers x 14 vectors
    would be ~2 MB of test data for no extra coverage)."""
    os.makedirs(os.path.join(root, "golden"), exist_ok=True)
    results, words = {}, {l["name"]: l["out_words"] for l in LAYERS}
    for k, (name, frame) in enumerate(frames.items()):
        r = replay(hw, frame)
        if k < act_vectors:
            for lname, data in r["layers"].items():
                arr = np.asarray(data)
                if arr.ndim == 2:                 # activation bank words
                    txt = "".join(hexbytes(row) + "\n" for row in arr[:words[lname]])
                else:                             # gap codes / logits, 24-bit words
                    txt = "".join(f"{int(v) & 0xFFFFFF:06x}\n" for v in arr.reshape(-1))
                _write(os.path.join(root, "golden", f"act_{lname}_{name}.mem"), txt)
        results[name] = dict(logits=[int(v) for v in r["logits"]], cls=r["cls"],
                             label=LABELS[r["cls"]])
    _write(os.path.join(root, "golden", "golden.json"),
           json.dumps(results, indent=1) + "\n")
    # flat forms the RTL testbench can $readmemh: one line per vector / per logit
    _write(os.path.join(root, "golden", "top_class.mem"),
           "".join(f"{results[n]['cls']:x}\n" for n in results))
    _write(os.path.join(root, "golden", "top_logits.mem"),
           "".join(f"{int(v) & 0xFFFFFF:06x}\n" for n in results
                   for v in results[n]["logits"]))
    _write(os.path.join(root, "golden", "top_vectors.mem"),
           "".join(f"{n}\n" for n in results))
    return results


def emit_unit_vectors(hw, rng, root):
    """Vectors for the PE array and the argmax tree.

    File formats (everything is one word per line so `$readmemh` can load it):

        pe_w.mem, pe_a.mem   64 hex chars = one 256-bit word (lane i = byte i)
                             NJOBS*T lines: job-major, tap-minor
        pe_b.mem             6 hex chars = one 24-bit bias, NJOBS*32 lines
                             (job-major, lane-minor)
        pe_s.mem             1 hex char  = one 4-bit shift, NJOBS*32 lines
        pe_y.mem             2 hex chars = one 8-bit expected output, NJOBS*32 lines
        argmax_l.mem         6 hex chars = one 24-bit logit, NROWS*12 lines
                             (row-major, lane-minor)
        argmax_y.mem         1 hex char  = expected class, NROWS lines
    """
    d = os.path.join(root, "golden", "unit")
    os.makedirs(d, exist_ok=True)

    T = 9
    jobs = []
    for _ in range(24):                                  # random accumulation
        jobs.append(_pe_job(rng, T, shift=None, bias=None))
    for sh in (0, 1, 8, 15):                             # edge shifts
        jobs.append(_pe_job(rng, T, shift=sh, bias=None))
    jobs.append(_pe_job(rng, T, shift=7, bias="max"))     # saturation
    jobs.append(_pe_job(rng, T, shift=7, bias="min"))     # ReLU kills everything
    jobs.append(_pe_job(rng, T, shift=7, bias=None, wmax=127, amax=255))  # extreme

    w_txt, a_txt, b_txt, s_txt, y_txt = [], [], [], [], []
    for (w, a, b, s, y) in jobs:
        for t in range(T):
            w_txt.append(hexbytes(w[t]))
            a_txt.append(hexbytes(a[t]))
        b_txt += [f"{int(v) & 0xFFFFFF:06x}" for v in b]
        s_txt += [f"{int(v) & 0xF:x}" for v in s]
        y_txt += [f"{int(v) & 0xFF:02x}" for v in y]
    _write(os.path.join(d, "pe_w.mem"), "\n".join(w_txt) + "\n")
    _write(os.path.join(d, "pe_a.mem"), "\n".join(a_txt) + "\n")
    _write(os.path.join(d, "pe_b.mem"), "\n".join(b_txt) + "\n")
    _write(os.path.join(d, "pe_s.mem"), "\n".join(s_txt) + "\n")
    _write(os.path.join(d, "pe_y.mem"), "\n".join(y_txt) + "\n")

    # --- argmax tree: 12 logits incl. ties and negatives ---
    rows, exp = [], []
    for _ in range(8):
        rows.append(rng.integers(-4000, 4000, NUM_CLASSES))
        exp.append(int(np.argmax(rows[-1])))
    rows.append(np.array([5, 5] + [0] * 10))             # tie -> lowest index wins
    exp.append(0)
    rows.append(np.array([-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12]))
    exp.append(0)
    _write(os.path.join(d, "argmax_l.mem"),
           "".join(f"{int(v) & 0xFFFFFF:06x}\n" for r in rows for v in r))
    _write(os.path.join(d, "argmax_y.mem"),
           "".join(f"{e:x}\n" for e in exp))
    return len(jobs), len(rows)


def _pe_job(rng, T, shift, bias, wmax=127, amax=255):
    w = rng.integers(-wmax, wmax + 1, size=(T, LANES)).astype(np.int64)
    a = rng.integers(0, amax + 1, size=(T, LANES)).astype(np.int64)
    if shift is None:
        s = rng.integers(0, SHIFT_MAX + 1, size=LANES).astype(np.int64)
    else:
        s = np.full(LANES, shift, dtype=np.int64)
    if bias is None:
        b = rng.integers(-(1 << 20), 1 << 20, size=LANES).astype(np.int64)
    elif bias == "max":
        b = np.full(LANES, (1 << 22), dtype=np.int64)
    else:
        b = np.full(LANES, -(1 << 20), dtype=np.int64)
    acc = np.array(b, dtype=np.int64)
    for t in range(T):
        acc = acc + w[t] * a[t]
    assert np.abs(acc).max() < (1 << 23), "unit vector exceeds the 24-bit accumulator"
    y = np.clip(acc >> s, 0, ACT_QMAX).astype(np.uint8)
    return w, a, b, s, y


def _write(path, text):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="\n") as f:
        f.write(text)


def quant_summary(rows):
    hdr = ("layer,kind,cin,cout,hi,wi,ho,wo,stride,pad,ng,pixels,"
           "s_in,s_out,shift_min,shift_max,max_abs_q,acc_bits,wrom_base,ch_base\n")
    out = [hdr]
    for r in rows:
        out.append(",".join(str(x) for x in r) + "\n")
    return "".join(out)
