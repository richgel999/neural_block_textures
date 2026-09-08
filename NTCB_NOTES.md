# NTCB_NOTES.md — the bit-packed container (`--write-ntcb`) implementation notes

September 7, 2026. Implements NTCB_PLAN.md (with its "Decisions taken for the open questions": fp16 rounding
under the flag, mode-0 levels reinstalled as frozen `--qes` levels, an explicit flag, the simulator's unsigned DC
width, no per-channel alignment) on top of commit 3629dbe. Two of the three hand-written mirrors changed
(main.cpp, ntc_decode.cpp); nothing in `cuda/` changed. Every number below is copied from program output of the
command shown; nothing is estimated. Nothing is committed. Builds: `cmake --build build --config Release` and
`cmake --build build_cuda --config Release`, both with 0 compiler warnings after the last edit (the CUDA link
prints only the pre-existing `LNK4098` defaultlib warning).

## 1. What was built

- **`--write-ntcb`** (main.cpp, off by default): at the end of a run (or of a `--load ... --iters 0` evaluation)
  the trainer writes `<out>/model.ntcb`, the container of NTCB_PLAN.md section 1: a little-endian header (magic
  `6E 74 63 62`, version 1, header length H, decoded and source size, file_bytes, an FNV-1a 32 hash of the
  payload, latent cell, dct_block, textures, 3 channels per texture, levels, activation, clamp, hidden widths,
  leak, nin, nout, the positional spec, then one record per level: W, H, C, filter, mode, per-channel bits,
  the DCT dc_step and q, and the side info (mode 0/1: C x fp32 (lo, hi); mode 2: 2^B fp32 palette values per
  channel; mode 3: none)); then, each behind an 8-byte section header (kind, coding 0, level, reserved, byte_len),
  one raw fixed-width body per level and one fp16 MLP body, every body zero-padded to a byte. Level modes:
  `0` = the post-hoc `--qbits` grid (B-bit indices, lo/hi = the shadow's per-channel min/max, the index computed
  as `bitrate_stats` computes it), `1` = a live `--qes` grid (index `qes_index`, asserted against the snapped
  value), `2` = the `--qat` palette (`qat_index`, asserted; palette = `QAT_GRID[bits][0..levels]`), `3` = the DCT
  level: per channel the 4-bit scale codes in block raster order, then per block the unsigned `dc_raw_bits` DC
  symbol and the 63 zigzag ACs as (run 7 bits, |q|-1 8 bits, sign 1 bit) tokens with a 7-bit EOB (run 64) when
  the last nonzero sits before zigzag 63 — the pass-2 loop of `dct_analyze` without its entropy terms. Bit order:
  LSB-first in a little-endian byte stream (64-bit accumulator). Every multi-byte field is written and read byte
  by byte; no host-order memory access anywhere in the container code.
- **fp16 rounding under the flag** (decision 2): the final MLP weights are rounded through IEEE binary16
  (round-to-nearest-even, denormals handled, overflow refused) once — at the last iteration right after the
  device download and before the last decode, or right after the loop for `--iters 0` — and printed as
  `ntcb     : mlp weights rounded to fp16 (N of M changed, max |dw| ...)`. On the GPU the rounded weights are
  pushed with `upload_model` (existing API), then `set_qes` and `set_dct` re-install the host's grids and symbols
  (`set_dct` last, since `upload_model` re-snaps the copy from the shadow). The last progress line, `model.bin`
  (`--resave` included), `recon_q_final.png`, the `done:` line and the container then describe the same weights.
  A 65536-pattern fp16 round-trip self-check (plus fixed rounding / overflow cases) runs once when the flag is on.
- **Reconciliation** (`file:` line before `done:`): `bitrate_stats` gained `bits_raw_latent` and `bits_header`;
  the writer's unpadded content bits must equal `bits_raw_latent + bits_mlp` exactly (`[OK]`), otherwise
  `MISMATCH` is printed and the process exits 1 after `done:`. The line also prints the file size in bytes and
  bpp next to the simulator's raw bpp, and the container's overhead (8H header bits, 64 per section header,
  padding bits).
- **Writer refusals** (message, `return false`, exit 1): a `--dct-q` run whose plane is not live, `|q| > 256`, a
  DC symbol outside its unsigned field, a value off its `--qat` / `--qes` grid, a mode-0 depth outside 2..12, a
  non-finite / overflowing weight, a weight that is not an fp16 value (the rounding hook did not run), a header
  above 65535 bytes, an unwritable file.
- **Trainer reader** (`--load model.ntcb`): on the magic the loader reads the whole file, validates the header
  (every range, H <= size, file_bytes == size, payload hash, mode/level constraints, nin), fills the same
  `saved_*` locals the v9..v13 cascade fills (so the existing option-mismatch message runs unchanged), then
  `ntcb_restore` fills `D.lat.z` (modes 0/1/2, dequantized through the strict-FP `qes_rebuild` grid or the stored
  palette), `D.dct.sym / code` (mode 3, checked run symbols and positions) and `D.mlp.p` (fp16). The existing
  post-load steps run as they are (qat snap: 0 moved; qes ranges restored; `dct_recon_level0`). A mode-0 level is
  installed as a frozen `QesLevel` with the file's lo/hi/B (`ntcb     : level l: 8-bit grid restored ...`), so
  `bitrate_stats` charges it through the qes branch and returns the values untouched; the CUDA `ModelDesc`
  receives that level's bits so `set_qes` accepts it (one tagged hook on the `md.qes_bits` line).
- **ntc_decode reader** (`load_ntcb`, inside the strict-FP section): the same header parser, then the sections
  into the existing `Model` (`version = 13`, `qat_ch` for mode 2, `qes_bits / lo / hi` for modes 0/1 so
  `level_quantized` holds and `q8_quantize` skips them, `dct_*` for mode 3 so the existing `dct_recon_level0`
  path runs). Summary: `model    : ... (ntcb v1)`, a mode-0 level prints `ntcb q8 (post-hoc grid) 8-bit, range
  ...`, and a `file     :` line recomputes the simulator figure from the header (and, for a dct level, from the
  token counts of the reconstructed symbols) and prints the exact-match verdict.
- Every hook and both region boundaries carry the `[NTCB]` tag; the region bodies are delimited by the tagged
  start / end lines (`// ---- NTCB: bit-packed container [NTCB]` ... `// ---- end of the NTCB region [NTCB]` in
  main.cpp, the same pair in ntc_decode.cpp), so `grep -n "\[NTCB\]"` finds every hook and both regions. The two
  untagged blank lines the review found (after the region end in main.cpp, after the forward declaration of
  `load_ntcb` in ntc_decode.cpp) were removed on September 7, 2026 (section 9). A DEPENDENCY note at the top of
  main.cpp and of ntc_decode.cpp names the layout as a mirrored item.

## 2. Layout as implemented: deviations from NTCB_PLAN.md and why

1. **The trainer's `file:` prefix.** The plan writes `file:` for the trainer line and `file     :` for the
   ntc_decode line; both are implemented literally (the trainer's `done:` line has the same short form).
2. **Mode 0/1 levels must have one bit depth for all channels.** The record carries a byte per channel (as
   planned); both readers refuse differing depths on a grid level ("per-channel bit depths on a grid level are
   not supported") because the trainer's `QesLevel` has one `bits` per level. Mode 2 keeps per-channel depths.
3. **The "level 0 is not DCT-coded yet" refusal is unreachable from the command line.** With `--dct-start F`
   the switch fires at `ceil(F * iters)`, which for F <= 1 is at or before the last iteration (F > 1 is refused),
   and `--iters 0` switches at 0; the plan's `--dct-start 1.0 --iters 5` therefore switches at iteration 5 and
   writes a valid file (section 3(f)). The guard stays in `ntcb_write` as a belt-and-braces check.
4. **`ntc_decode` exits 1 on a MISMATCH** (since the review fixes of September 7, 2026; before them it only
   printed it). A mismatch there can only come from the dct token bookkeeping (modes 0/1/2 are computed from the
   same header fields), i.e. a decoder bug; a failed sync marker or a corrupt file is refused at load with the
   named message and exit 1, like every other refusal; the trainer's `file:` MISMATCH exits 1 as planned.
5. **The reload of a container reports its fp32 PSNR equal to its quantized PSNR** (visible in 3(e1)): the
   container holds no fp32 shadow, so the "final psnr" of a reloaded file is decoded from the file's grid values.
   The quantized part of the `done:` line (PSNR, raw / entropy / ctx bpp, `[level 0 alone]`, the `[dct: ...]`
   bracket) is what (c) compares and it is identical in every case below.
6. The plan's 32-bit FNV-1a is implemented as its own function in both files (`ntcb_fnv1a32`); ntc_decode's
   existing `fnv1a` is 64-bit and is used for the RGB8 hash only.
7. The trainer's magic test on an ntcb file happens after the existing `fread(hdr, sizeof(int), 6, f)`; a file
   shorter than 24 bytes therefore prints the existing `cannot read <file>` (rc 1) instead of the container's own
   message (3(f), "truncated at 20 bytes").

## 3. Verification (commands and output; every figure from the final binaries)

(The output quoted in this section is from the binaries as first built. The review fixes of section 9 changed the
`file:` / `file     :` lines: "payload" became "content", the hash covers the whole file, a `sync` term and the
bpp denominator were added, and every file grew by a few bytes; section 9 quotes the current lines.)

### (a) Size vs simulator

```
build_cuda\Release\ntc.exe image3.png --cuda --block 8 --latent 0 0 1 --latent2 0 0 4 --filter nearest,bilinear --pos lv1local --dct-q 50 --qes 0,8 --mlp 27,27 --leak 0.0009765625 --load out_image3_b8_dct50_c4bilin_qes8_mlp27_cuda8k_fd50/model.bin --iters 0 --write-ntcb --resave --out out_ntcb_dct50
iter      0  mse 1.774  psnr  45.64 dB  (initial)
ntcb     : mlp weights rounded to fp16 (1056 of 1056 changed, max |dw| 4.784e-04)
resave   : wrote out_ntcb_dct50/model.bin as v13 (level 0 DCT-coded, q 50, DC step 4; symbols and codes from the loaded v13 file)
file: out_ntcb_dct50/model.ntcb 565294 bytes = 2.146 bpp (simulator raw 2.145 bpp); payload 4521126 bits = simulator 4521478 - 352 side-info bits [OK]; container header 1032 + section headers 192 + padding 2 bits
done: final psnr 45.64 dB (best -1.00) at fp32 34.016 bpp | --dct-q 50 level 0 (bit simulator) + --qes 0,8 latent + fp16 mlp: psnr 45.64 dB at 2.145 bpp raw, 1.071 bpp entropy-coded, 0.925 bpp with an (up, left) context on level 0 [level 0 alone: 0.482] [dct: raw 1.637 h0 0.628 ctx 0.482 code raw/ctx 0.062/0.013 bpp; nz 5.36/blk eob0 0.3%] | 1.3s
rc=0; out_ntcb_dct50/model.ntcb 565294 bytes, sha256 38aedf194b771ddde885c8ab31325bb2bc5fbdcedff8e1b0b7a9f2282fd8c8ce
```
The payload is exactly the plan's worked expectation (4521126 = 3450470 dct + 1053760 level 1 + 16896 mlp);
the run's log had quoted the same `done:` figures (45.64 dB, 2.145 / 1.071 / 0.925 bpp) before the fp16
rounding. The same command on the CPU build (`build\Release\ntc.exe`, no `--cuda`, `--out out_ntcb_dct50_cpu`):
```
ntcb     : mlp weights rounded to fp16 (1056 of 1056 changed, max |dw| 4.784e-04)
file: out_ntcb_dct50_cpu/model.ntcb 565294 bytes = 2.146 bpp (simulator raw 2.145 bpp); payload 4521126 bits = simulator 4521478 - 352 side-info bits [OK]; container header 1032 + section headers 192 + padding 2 bits
done: (the same line as above) ... 45.64 dB at 2.145 bpp raw, 1.071 bpp entropy-coded, 0.925 bpp ... | 1.4s
cmp out_ntcb_dct50/model.ntcb out_ntcb_dct50_cpu/model.ntcb -> identical; the two resaved model.bin -> identical
```
`out_chk` (the 5-iteration chief1 run of DCT_NOTES section 3, 67.4% EOB-only) and `out_dct_a_cpu` (the
500-iteration CPU run), `build\Release\ntc.exe chief1.png --block 8 --latent 0 0 1 --latent2 0 0 4 --filter
nearest,bilinear --pos lv1local --dct-q 50 --qes 0,8 --mlp 17,17 --leak 0.0009765625 --load <dir>/model.bin
--iters 0 --write-ntcb --resave --out out_ntcb_<dir>`:
```
ntcb     : mlp weights rounded to fp16 (496 of 496 changed, max |dw| 4.838e-04)
file: out_ntcb_out_chk/model.ntcb 30055 bytes = 0.917 bpp (simulator raw 0.914 bpp); payload 239216 bits = simulator 239568 - 352 side-info bits [OK]; container header 1032 + section headers 192 + padding 0 bits
done: final psnr 11.55 dB (best -1.00) at fp32 34.061 bpp | --dct-q 50 level 0 (bit simulator) + --qes 0,8 latent + fp16 mlp: psnr 11.55 dB at 0.914 bpp raw, 0.564 bpp entropy-coded, 0.561 bpp with an (up, left) context on level 0 [level 0 alone: 0.074] [dct: raw 0.382 h0 0.077 ctx 0.074 code raw/ctx 0.062/0.006 bpp; nz 0.34/blk eob0 67.4%] | 0.2s
ntcb     : mlp weights rounded to fp16 (496 of 496 changed, max |dw| 4.071e-04)
file: out_ntcb_out_dct_a_cpu/model.ntcb 177277 bytes = 5.410 bpp (simulator raw 5.407 bpp); payload 1416989 bits = simulator 1417341 - 352 side-info bits [OK]; container header 1032 + section headers 192 + padding 3 bits
done: final psnr 29.95 dB (best -1.00) at fp32 34.061 bpp | --dct-q 50 level 0 (bit simulator) + --qes 0,8 latent + fp16 mlp: psnr 29.95 dB at 5.407 bpp raw, 2.359 bpp entropy-coded, 2.099 bpp with an (up, left) context on level 0 [level 0 alone: 1.611] [dct: raw 4.875 h0 1.871 ctx 1.611 code raw/ctx 0.062/0.011 bpp; nz 18.32/blk eob0 0.0%] | 0.2s
```
Hand check of `out_chk`'s level-0 body (ntc_decode below prints `nnz 1399` for it; DCT_NOTES quotes the same):
231280 level bits − 64·64·4·8 = 100208 level-0 bits = 4096 × 8 (DC) + 1399 × 16 + 4096 × 7 (an EOB in every
block: no block reaches zigzag 63) + 4096 × 4 (codes) = 32768 + 22384 + 28672 + 16384 = 100208.

### (b) Decoder byte identity

```
build\Release\ntc_decode.exe out_ntcb_dct50/model.ntcb -o n --compare out_ntcb_dct50/recon_q_final.png --verify
note: every level was quantized in training; --q8 and --fp32-latent decode the same values
model    : out_ntcb_dct50/model.ntcb (ntcb v1)
level 0  : 1480x1424x1 nearest, dct 8x8, q 50, dc step 4, 4-bit scale codes, 5.4 nonzero ACs/block, 0.3% EOB-only, fp32 planar
level 1  : 185x178x4 bilinear, qes 8-bit, range [-3.563, 7.957] [-5.49, 3.345] [-6.003, 5.431] [-9.91, 5.356]
dct      : nz mean 5.36/blk, eob0 0.3%, lnz mean 15.6, clamped 0.31%, codes k0..k15 55 621 551 1922 5533 9863 11877 2277 231 0 0 0 0 0 0 0; raw 104.8 bits/blk = 1.6372 bpp (fixed-length simulator: 8 DC + 16 per nonzero + 7 per EOB + 4 code bits)
idct     : 11.14 ms single-threaded (32930 blocks; 8.8 MAC/texel with zero-skipping, 16 dense; nnz 176550)
file     : out_ntcb_dct50/model.ntcb 565294 bytes = 2.146 bpp (simulator raw 2.145 bpp: 4504230 level bits + 352 side-info + 16896 mlp bits; payload 4521126 bits matches the simulator exactly; container header 1032 + section headers 192 + padding 2 bits)
compare t0: out_ntcb_dct50/recon_q_final.png (1478x1424): PSNR 90.94 dB, max |diff| 1, |diff| histogram: 0: 6313685 (99.9948%), 1: 331 (0.0052%), 2: 0 (0.0000%), >2: 0 (0.0000%)
verify   : pre-nonlinearity: max 0 ulp (0 of 6322560 values differ); output fp32: max 42 ulp, max |diff| 7.75e-07; RGB8 mismatches: 332 of 6322560 (0.00525%)
rc=0
build\Release\ntc_decode.exe out_ntcb_dct50/model.bin -o b --compare out_ntcb_dct50/recon_q_final.png --verify
model    : out_ntcb_dct50/model.bin (v13)
compare t0: ... PSNR 90.94 dB, max |diff| 1, |diff| histogram: 0: 6313685 (99.9948%), 1: 331 (0.0052%), 2: 0, >2: 0
verify   : pre-nonlinearity: max 0 ulp (0 of 6322560 values differ); output fp32: max 42 ulp, max |diff| 7.75e-07; RGB8 mismatches: 332 of 6322560 (0.00525%)
rc=0
cmp n.png b.png -> identical; --q8 vs --fp32-latent on the .ntcb -> identical
```
`out_ntcb_out_chk` and `out_ntcb_out_dct_a_cpu` the same way:
```
file     : out_ntcb_out_chk/model.ntcb 30055 bytes = 0.917 bpp (simulator raw 0.914 bpp: 231280 level bits + 352 side-info + 7936 mlp bits; payload 239216 bits matches the simulator exactly; container header 1032 + section headers 192 + padding 0 bits)
compare t0: out_ntcb_out_chk/recon_q_final.png (512x512): PSNR 88.17 dB, max |diff| 1, |diff| histogram: 0: 786354 (99.9901%), 1: 78 (0.0099%), 2: 0 (0.0000%), >2: 0 (0.0000%)
verify   : pre-nonlinearity: max 0 ulp (0 of 786432 values differ); output fp32: max 24 ulp, max |diff| 7.75e-07; RGB8 mismatches: 78 of 786432 (0.00992%)
ntcb PNG == model.bin PNG (cmp identical)
file     : out_ntcb_out_dct_a_cpu/model.ntcb 177277 bytes = 5.410 bpp (simulator raw 5.407 bpp: 1409053 level bits + 352 side-info + 7936 mlp bits; payload 1416989 bits matches the simulator exactly; container header 1032 + section headers 192 + padding 3 bits)
compare t0: out_ntcb_out_dct_a_cpu/recon_q_final.png (512x512): PSNR 90.19 dB, max |diff| 1, |diff| histogram: 0: 786383 (99.9938%), 1: 49 (0.0062%), 2: 0 (0.0000%), >2: 0 (0.0000%)
verify   : pre-nonlinearity: max 0 ulp (0 of 786432 values differ); output fp32: max 42 ulp, max |diff| 7.75e-07; RGB8 mismatches: 50 of 786432 (0.00636%)
ntcb PNG == model.bin PNG (cmp identical)
```

### (c) Trainer reload

`<same options as (a)> --load out_ntcb_dct50/model.ntcb --iters 0 --out out_ntcb_dct50_reload`, CPU build:
```
qes      : ranges restored from the file (frozen)
dct      : symbols and scale codes restored from the file (level 0 = IDCT of the symbols, no re-snap at load; codes frozen)
loaded   : out_ntcb_dct50/model.ntcb
iter      0  mse 1.774  psnr  45.64 dB  (initial)
done: final psnr 45.64 dB (best -1.00) at fp32 34.016 bpp | --dct-q 50 level 0 (bit simulator) + --qes 0,8 latent + fp16 mlp: psnr 45.64 dB at 2.145 bpp raw, 1.071 bpp entropy-coded, 0.925 bpp with an (up, left) context on level 0 [level 0 alone: 0.482] [dct: raw 1.637 h0 0.628 ctx 0.482 code raw/ctx 0.062/0.013 bpp; nz 5.36/blk eob0 0.3%] | 1.4s
rc=0
```
With `--cuda` (`build_cuda\Release\ntc.exe`): the same `iter 0` and `done:` lines (`45.64 dB at 2.145 bpp raw,
1.071 bpp entropy-coded, 0.925 bpp ... [dct: raw 1.637 h0 0.628 ctx 0.482 ...]`, 1.3s), rc=0 — identical to the
writing invocation's `done:` line in (a). Refusals (`--load out_ntcb_dct50/model.ntcb --iters 0` without
`--dct-q`, with `--dct-q 30`, with `--mlp 17,17`), each rc=1:
```
out_ntcb_dct50/model.ntcb was saved with --latent 1480 1424 1 --latent2 185 178 4 --filter nearest,bilinear --leak 0.000977 --qes 0,8 --dct-q 50 --dct-dc-step 4 --mlp 27,27 --pos lv1local and 1 texture(s); pass the same options
```

### (d) qat-mode file

```
build_cuda\Release\ntc.exe image3.png --cuda --block 8 --latent 0 0 1 --latent2 0 0 4 --filter nearest,bilinear --pos lv1local --qat 4 --qes 0,8 --mlp 27,27 --leak 0.0009765625 --load out_image3_b8_c1q4_c4bilin_qes8_mlp27_cuda8k_fd50_psnr8/model.bin --iters 0 --write-ntcb --resave --out out_ntcb_q4
iter      0  mse 0.831  psnr  48.94 dB  (initial)
ntcb     : mlp weights rounded to fp16 (1056 of 1056 changed, max |dw| 4.272e-04)
file: out_ntcb_q4/model.ntcb 1187807 bytes = 4.509 bpp (simulator raw 4.508 bpp); payload 9500736 bits = simulator 9500992 - 256 side-info bits [OK]; container header 1528 + section headers 192 + padding 0 bits
done: final psnr 48.94 dB (best -1.00) at fp32 34.016 bpp | --qes 0,8 latent + fp16 mlp: psnr 48.94 dB at 4.508 bpp raw, 4.146 bpp entropy-coded, 1.295 bpp with an (up, left) context on level 0 [level 0 alone: 0.866] | 0.9s
```
Level-0 section = 1480 × 1424 × 4 = 8430080 bits, level 1 = 185 × 178 × 4 × 8 = 1053760, sum 9483840 = the
`level bits` below; the palette is 16 floats (header 1528 bits = 191 bytes). PSNR after the fp16 rounding
48.94 dB against the log's 48.94 dB: delta 0.00 dB at two decimals.
```
build\Release\ntc_decode.exe out_ntcb_q4/model.ntcb -o q4n --compare out_ntcb_q4/recon_q_final.png --verify
model    : out_ntcb_q4/model.ntcb (ntcb v1)
level 0  : 1480x1424x1 nearest, qat 4, fp32 planar
level 1  : 185x178x4 bilinear, qes 8-bit, range [-1.956, 9.106] [-7.533, 3.016] [-3.967, 8.845] [-8.646, 6.154]
file     : out_ntcb_q4/model.ntcb 1187807 bytes = 4.509 bpp (simulator raw 4.508 bpp: 9483840 level bits + 256 side-info + 16896 mlp bits; payload 9500736 bits matches the simulator exactly; container header 1528 + section headers 192 + padding 0 bits)
compare t0: out_ntcb_q4/recon_q_final.png (1478x1424): PSNR 90.67 dB, max |diff| 1, |diff| histogram: 0: 6313664 (99.9944%), 1: 352 (0.0056%), 2: 0 (0.0000%), >2: 0 (0.0000%)
verify   : pre-nonlinearity: max 0 ulp (0 of 6322560 values differ); output fp32: max 42 ulp, max |diff| 7.75e-07; RGB8 mismatches: 344 of 6322560 (0.00544%)
model.bin (v12): the same compare and verify lines; cmp q4n.png q4b.png -> identical; --q8 vs --fp32-latent -> identical
--pack-selectors: level 0  : 1480x1424x1 nearest, qat 4, packed uint8 indices; PNG identical
reload CPU and --cuda (--load out_ntcb_q4/model.ntcb --iters 0): iter 0 psnr 48.94 dB; done: ... 48.94 dB at 4.508 bpp raw, 4.146 bpp entropy-coded, 1.295 bpp ... [level 0 alone: 0.866]  (both rc=0, = the writing line)
```

### (e) 4-texture materials

(e1) `out_m1234_512c2q2_128c4_cuda8k_fd50` (v10, `--qat 2` on 512x512x2, a continuous nearest 128x128x4 level 1
-> mode 0, the frozen-grid install):
```
build_cuda\Release\ntc.exe m1.png m2.png m3.png m4.png --cuda --latent 512 512 2 --latent2 128 128 4 --filter nearest,nearest --pos lv1local --qat 2 --mlp 36,36 --leak 0.0009765625 --load out_m1234_512c2q2_128c4_cuda8k_fd50/model.bin --iters 0 --write-ntcb --resave --out out_ntcb_m1234
iter      0  mse 74.672  psnr  29.40 dB  (initial) | tex 25.97 33.46 29.33 33.47
ntcb     : mlp weights rounded to fp16 (2100 of 2100 changed, max |dw| 9.527e-04)
file: out_ntcb_m1234/model.ntcb 200992 bytes = 6.134 bpp (1.533/tex) (simulator raw 6.129 bpp (1.532/tex)); payload 1606464 bits = simulator 1606720 - 256 side-info bits [OK]; container header 1280 + section headers 192 + padding 0 bits
done: final psnr 29.40 dB (best -1.00) tex 25.97 33.46 29.33 33.47 at fp32 72.256 bpp (18.064/tex) | 8-bit latent + fp16 mlp: psnr 29.38 dB qtex 25.96 33.45 29.32 33.40 at 6.129 bpp raw, 5.649 bpp entropy-coded, 5.111 bpp with an (up, left) context on level 0 (1.532, 1.412, 1.278 /tex) [level 0 alone: 3.324] | 0.5s
build\Release\ntc_decode.exe out_ntcb_m1234/model.ntcb -o mn --compare <the four recon_q_final_tK.png> --verify
level 0  : 512x512x2 nearest, qat 2,2, fp32 planar
level 1  : 128x128x4 nearest, ntcb q8 (post-hoc grid) 8-bit, range [-17.65, 12.71] [-12.72, 9.838] [-11.94, 12.99] [-14.02, 6.319]
file     : out_ntcb_m1234/model.ntcb 200992 bytes = 6.134 bpp (simulator raw 6.129 bpp: 1572864 level bits + 256 side-info + 33600 mlp bits; payload 1606464 bits matches the simulator exactly; container header 1280 + section headers 192 + padding 0 bits)
compare t0: PSNR 89.09 dB, max |diff| 1, 1: 63 (0.0080%)   t1: PSNR 87.40 dB, max |diff| 1, 1: 93   t2: PSNR 87.95 dB, max |diff| 1, 1: 82   t3: PSNR 95.95 dB, max |diff| 1, 1: 13
verify   : pre-nonlinearity: max 0 ulp (0 of 3145728 values differ); output fp32: max 42 ulp, max |diff| 7.75e-07; RGB8 mismatches: 252 of 3145728 (0.00801%)
model.bin (v12, level 1 "q8 per channel"): the same four compare lines and verify line; mn_t0..t3.png == mb_t0..t3.png (cmp identical)
reload CPU (--load out_ntcb_m1234/model.ntcb --iters 0):
ntcb     : level 1: 8-bit grid restored from the file (frozen; its stored values are the run's post-hoc quantization)
iter      0  mse 74.966  psnr  29.38 dB  (initial) | tex 25.96 33.45 29.32 33.40
done: final psnr 29.38 dB (best -1.00) tex 25.96 33.45 29.32 33.40 at fp32 72.256 bpp (18.064/tex) | 8-bit latent + fp16 mlp: psnr 29.38 dB qtex 25.96 33.45 29.32 33.40 at 6.129 bpp raw, 5.649 bpp entropy-coded, 5.111 bpp with an (up, left) context on level 0 (1.532, 1.412, 1.278 /tex) [level 0 alone: 3.324] | 0.5s
reload --cuda: the same ntcb, iter 0 and done: lines (0.4s), rc=0
```
The quantized part of the reloaded `done:` line equals the writing run's; the fp32 part reads 29.38 instead of
29.40 dB because the container has no fp32 shadow (section 2, item 5).

(e2) `out_m1234` (v12, bilinear 128x128x4 and 64x64x4, `--pos uv`: two mode-0 levels, a bilinear mode-0 level 0):
```
build\Release\ntc.exe m1.png m2.png m3.png m4.png --latent 128 128 4 --latent2 64 64 4 --mlp 36,36 --load out_m1234/model.bin --iters 0 --write-ntcb --resave --out out_ntcb_m1234v12
iter      0  mse 185.374  psnr  25.45 dB  (initial) | tex 23.17 31.47 23.22 29.55
ntcb     : mlp weights rounded to fp16 (2172 of 2172 changed, max |dw| 2.873e-04)
file: out_ntcb_m1234v12/model.ntcb 86444 bytes = 2.638 bpp (0.660/tex) (simulator raw 2.635 bpp (0.659/tex)); payload 690112 bits = simulator 690624 - 512 side-info bits [OK]; container header 1248 + section headers 192 + padding 0 bits
done: final psnr 25.45 dB (best -1.00) tex 23.17 31.47 23.22 29.55 at fp32 10.265 bpp (2.566/tex) | 8-bit latent + fp16 mlp: psnr 25.45 dB qtex 23.17 31.47 23.22 29.54 at 2.635 bpp raw, 2.213 bpp entropy-coded, 0.949 bpp with an (up, left) context on level 0 (0.659, 0.553, 0.237 /tex) [level 0 alone: 0.388] | 0.5s
ntc_decode: level 0  : 128x128x4 bilinear, ntcb q8 (post-hoc grid) 8-bit, range [-5.328, 4.734] [-4.179, 6.264] [-3.926, 5.37] [-4.016, 5.06]
            level 1  : 64x64x4 bilinear, ntcb q8 (post-hoc grid) 8-bit, range [-4.707, 3.937] [-5.561, 3.216] [-4.079, 3.718] [-3.977, 4.76]
file     : out_ntcb_m1234v12/model.ntcb 86444 bytes = 2.638 bpp (simulator raw 2.635 bpp: 655360 level bits + 512 side-info + 34752 mlp bits; payload 690112 bits matches the simulator exactly; container header 1248 + section headers 192 + padding 0 bits)
compare t0: PSNR 90.01 dB, max |diff| 1, 1: 51   t1: PSNR 87.74 dB, 1: 86   t2: PSNR 88.51 dB, 1: 72   t3: PSNR 92.04 dB, 1: 32   (max |diff| 1 on all four)
verify   : pre-nonlinearity: max 0 ulp (0 of 3145728 values differ); output fp32: max 41 ulp, max |diff| 7.75e-07; RGB8 mismatches: 243 of 3145728 (0.00772%)
model.bin (v12): the same lines; vn_t0..t3.png == vb_t0..t3.png (cmp identical)
reload CPU and --cuda: ntcb : level 0: 8-bit grid restored ...; ntcb : level 1: 8-bit grid restored ...; iter 0 psnr 25.45 dB; done: ... 25.45 dB qtex 23.17 31.47 23.22 29.54 at 2.635 bpp raw, 2.213 bpp entropy-coded, 0.949 bpp ... [level 0 alone: 0.388]  (both rc=0)
```

### The training-loop rounding hook (a run that writes the file itself, not a conversion)

CPU, the regression configuration with the flag (`build\Release\ntc.exe --crop 512 --mlp-pairs 32 --iters 200
--print-every 200 --save-every 200 --write-ntcb --out out_reg_ntcb`):
```
ntcb     : mlp weights rounded to fp16 (843 of 843 changed, max |dw| 4.741e-04)
iter    200  mse 269.324  psnr  23.83 dB  best  23.83 | q8 psnr  23.83 @ 0.552 bpp (ent 0.500, ctx 0.084) | mlp batch 0.00442 dstd 9.65e-05 | lat mean -0.051 sd 0.551 max 2.08 | 7.6s (26.17 it/s)
file: out_reg_ntcb/model.ntcb 18194 bytes = 0.555 bpp (simulator raw 0.552 bpp); payload 144560 bits = simulator 144816 - 256 side-info bits [OK]; container header 864 + section headers 128 + padding 0 bits
done: final psnr 23.83 dB (best 23.83) at fp32 2.103 bpp | 8-bit latent + fp16 mlp: psnr 23.83 dB at 0.552 bpp raw, 0.500 bpp entropy-coded, 0.084 bpp with an (up, left) context on level 0 [level 0 alone: 0.032] | 7.8s
ntc_decode out_reg_ntcb/model.ntcb --compare out_reg_ntcb/recon_q_final.png --verify: level 0  : 64x64x4 bilinear, ntcb q8 (post-hoc grid) 8-bit, range [-2.022, 1.731] [-1.736, 1.772] [-1.761, 2.083] [-1.661, 1.888]
file     : out_reg_ntcb/model.ntcb 18194 bytes = 0.555 bpp (simulator raw 0.552 bpp: 131072 level bits + 256 side-info + 13488 mlp bits; payload 144560 bits matches the simulator exactly; container header 864 + section headers 128 + padding 0 bits)
compare t0: out_reg_ntcb/recon_q_final.png (512x512): PSNR 88.11 dB, max |diff| 1, |diff| histogram: 0: 786353 (99.9900%), 1: 79 (0.0100%), 2: 0 (0.0000%), >2: 0 (0.0000%)
verify   : pre-nonlinearity: max 0 ulp (0 of 786432 values differ); output fp32: max 42 ulp, max |diff| 7.75e-07; RGB8 mismatches: 77 of 786432 (0.00979%)
ntcb PNG == model.bin PNG (cmp identical)
ntc --crop 512 --mlp-pairs 32 --load out_reg_ntcb/model.ntcb --iters 0: ntcb : level 0: 8-bit grid restored ...; iter 0 psnr 23.83 dB; done: ... 23.83 dB at 0.552 bpp raw, 0.500 bpp entropy-coded, 0.084 bpp ... [level 0 alone: 0.032]  rc=0
```
(the rounding moves the iteration-200 mse from 269.314 to 269.324; the PSNR is unchanged at two decimals.)
GPU, the README's `--dct-q 50` check line trained for real with the flag (`build_cuda\Release\ntc.exe chief1.png
--cuda --block 8 --latent 0 0 1 --latent2 0 0 4 --filter nearest,bilinear --pos lv1local --dct-q 50 --dct-start 0
--qes 0,8 --qes-start 0 --mlp 17,17 --leak 0.0009765625 --iters 5 --print-every 5 --write-ntcb --out out_ntcb_chk5`):
```
ntcb     : mlp weights rounded to fp16 (496 of 496 changed, max |dw| 4.838e-04)
iter      5  mse 4553.149  psnr  11.55 dB  best  11.55 | dct50,q8 psnr  11.55 @ 0.914 bpp (ent 0.564, ctx 0.561) | dct nz 0.34 lnz 1.8 eob0 67.4% clamp 0.00% bits/blk 24.5/5.0/4.8 chg sym 4443 zq 223232 | ...
file: out_ntcb_chk5/model.ntcb 30055 bytes = 0.917 bpp (simulator raw 0.914 bpp); payload 239216 bits = simulator 239568 - 352 side-info bits [OK]; container header 1032 + section headers 192 + padding 0 bits
done: final psnr 11.55 dB (best 11.55) at fp32 34.061 bpp | --dct-q 50 level 0 (bit simulator) + --qes 0,8 latent + fp16 mlp: psnr 11.55 dB at 0.914 bpp raw, 0.564 bpp entropy-coded, 0.561 bpp ... [dct: raw 0.382 h0 0.077 ctx 0.074 code raw/ctx 0.062/0.006 bpp; nz 0.34/blk eob0 67.4%] | 0.5s
ntc_decode: file     : ... payload 239216 bits matches the simulator exactly ...; compare t0: PSNR 88.11 dB, max |diff| 1, 1: 79; verify max 0 ulp (0 of 786432); ntcb PNG == model.bin PNG (cmp identical)
reload CPU: iter 0 psnr 11.55 dB; done: ... 11.55 dB at 0.914 bpp raw, 0.564 bpp entropy-coded, 0.561 bpp ... (rc=0); reload --cuda: the same (rc=0)
cmp out_ntcb_chk5/model.ntcb out_ntcb_out_chk/model.ntcb -> identical (sha256 983bf019b97f496d5bcc67be3f16c95dd5db535c511c057150d612348bf8fec1 both)
```
The file written by the GPU run at its last iteration is byte-identical to the conversion of `out_chk/model.bin`
(the same configuration trained by the same command in DCT_NOTES): the loop hook and the `--iters 0` hook
produce the same bytes.

### (f) Refusals

Script `ntcb_corrupt.py` (scratchpad; Python) on `out_ntcb_out_chk/model.ntcb` (dct level 0, qes level 1) and
`out_ntcb_m1234/model.ntcb` (qat 2,2 level 0, mode-0 level 1). Header edits are outside the payload hash and are
caught by the range checks; the section-header and token-stream edits are re-hashed so the specific check fires
(without the re-hash the hash check fires first, as the "byte flipped" case shows). Each line: the case, then
`ntc_decode`'s last output line and rc, and `ntc --load`'s (chief1 / m1..m4 options as in (a) / (e1)):
```
truncated at 20 bytes            ntc_decode rc=1: header length 129 exceeds the file size 20 (truncated or corrupt)   |  ntc rc=1: cannot read <file>
truncated at H - 1               ntc_decode rc=1: header length 129 exceeds the file size 128 (truncated or corrupt)  |  ntc rc=1: corrupt or truncated ntcb file: header length 129 exceeds the file size 128 (truncated or corrupt)
truncated at H + 100             ntc_decode rc=1: file_bytes 30055 != actual size 229 (truncated or extended)         |  ntc rc=1: corrupt or truncated ntcb file: file_bytes 30055 != actual size 229 (truncated or extended)
truncated at file_bytes - 1      ntc_decode rc=1: file_bytes 30055 != actual size 30054 (truncated or extended)       |  ntc rc=1: ... file_bytes 30055 != actual size 30054 (truncated or extended)
one payload byte flipped         ntc_decode rc=1: payload hash mismatch (corrupt file)                                |  ntc rc=1: ... payload hash mismatch (corrupt file)
level 0 mode 9                   ntc_decode rc=1: level 0: bad dims (512x512x1), filter flag (1) or mode (9)          |  ntc rc=1: ... level 0: bad dims (512x512x1), filter flag (1) or mode (9)
level 0 filter 2                 ntc_decode rc=1: level 0: bad dims (512x512x1), filter flag (2) or mode (3)          |  ntc rc=1: the same
level 1 bits 13                  ntc_decode rc=1: level 1: bit depth 13 out of range for mode 1                       |  ntc rc=1: the same
level 0 q 0                      ntc_decode rc=1: level 0: dct quality 0 out of range (1..100)                        |  ntc rc=1: the same
level 0 dc_step 0                ntc_decode rc=1: level 0: dc_step 0 out of range (1..64)                             |  ntc rc=1: the same
section 0 byte_len + 1 (re-hashed) ntc_decode rc=1: level 0: section length 12527 != the 12526 bytes its content occupies  |  ntc rc=1: the same
magic changed                    ntc_decode rc=1: not an ntc model file (bad magic)                                   |  ntc rc=1: not a v9..v13 ntc model file (the legacy header and v2..v8 were retired in v0.8)
version 2                        ntc_decode rc=1: ntcb version 2 (this build reads version 1)                        |  ntc rc=1: corrupt or truncated ntcb file: ntcb version 2 (this build reads version 1)
token run 63 at block 0 (re-hashed)        ntc_decode rc=1: level 0: bad run symbol 63 at block 0 (0..62 or the EOB 64)   |  ntc rc=1: the same
token runs 30 then 40 at block 0 (re-hashed) ntc_decode rc=1: level 0: token stream runs past the 63 AC positions at block 0  |  ntc rc=1: the same
ALL CASES EXIT 1: True   (both files; the m1234 container's level-1 case reads "bit depth 13 out of range for mode 0", its byte_len case "131073 != the 131072 bytes")
```
(The "filter 2" and "mode 9" messages were run with the final binaries; an earlier build printed only the mode in
the parenthesis, which is why the message now names all three fields.) The plan's last case, `--dct-start 1.0
--iters 5 --write-ntcb` (`build\Release\ntc.exe chief1.png --block 8 --latent 0 0 1 --latent2 0 0 4 --filter
nearest,bilinear --pos lv1local --dct-q 50 --dct-start 1.0 --qes 0,8 --mlp 17,17 --leak 0.0009765625 --iters 5
--print-every 5 --write-ntcb --out out_ntcb_nolive`) does switch at iteration 5 (section 2, item 3) and writes a
valid file:
```
iter      5  dct: scale codes fitted from the decoder sensitivity probe (frozen), level 0 decodes from the DCT-snapped plane from here; psnr 11.53 -> 11.51 dB; nz 0.24/blk, eob-only 76.7%, ...
ntcb     : mlp weights rounded to fp16 (496 of 496 changed, max |dw| 4.779e-04)
file: out_ntcb_nolive/model.ntcb 30269 bytes = 0.924 bpp (simulator raw 0.920 bpp); payload 240928 bits = simulator 241280 - 352 side-info bits [OK]; ...
rc=0
```

### (g) Firewall (flag absent, non-ntcb files)

Regression pin, CPU build:
```
build\Release\ntc.exe --crop 512 --mlp-pairs 32 --iters 200 --print-every 200 --save-every 200 --out out_reg
iter    200  mse 269.314  psnr  23.83 dB  best  23.83 | q8 psnr  23.83 @ 0.552 bpp (ent 0.500, ctx 0.084) | mlp batch 0.00442 dstd 9.65e-05 | lat mean -0.051 sd 0.551 max 2.08 | 8.3s (23.99 it/s)
done: final psnr 23.83 dB (best 23.83) at fp32 2.103 bpp | 8-bit latent + fp16 mlp: psnr 23.83 dB at 0.552 bpp raw, 0.500 bpp entropy-coded, 0.084 bpp with an (up, left) context on level 0 [level 0 alone: 0.032] | 8.5s
xxd -l 4 out_reg/model.bin -> 3c43 544e (v12); sha256 cb88470777e1d041c8803bcf3ca531434b9a2b2b5bce2ef88e031fbc86d69600 (= DCT_NOTES section 3); no out_reg/model.ntcb
ntc_decode out_reg/model.bin --compare out_reg/recon_q_final.png --verify: PSNR 88.70 dB, max |diff| 1, 1: 69 (0.0088%); verify max 0 ulp (0 of 786432); RGB8 mismatches 68  (= DCT_NOTES)
```
The six README `--cuda-check` lines and the `--dct-q 90` / `20` variants, `build_cuda\Release\ntc.exe`, each
`cuda-check: all passed`, rc=0, with the DCT_NOTES figures: decode 1.788e-07 / 1.788e-07 / 1.192e-07 /
1.192e-07 / 1.192e-07 / 1.192e-07; mlp ES 5.599e-07 / 3.334e-07 / 2.439e-07 / 3.991e-07 / 2.424e-07 / 3.442e-07;
latent ES grad 2.174e-05 / 7.003e-06 / 4.645e-05 / 3.884e-05 / 3.669e-05 / 3.305e-05; qat search `3 of 524288`
(worse 0, better 1, ties 1) / `3 of 786432` (worse 2) / `1 of 262144` (ties 1); qes snap 0 / 0 on every line;
the `--dct-q 50` line's check 7 `symbols 0 of 262144 differ, plane 0 of 262144 values differ (max 0 ulp), codes 0
of 4096 differ ...; after a latent step with lr 0.02: 0 / 0 (max 0 ulp) / 0` and `dct refit: host re-probe moved
195 of 4096 codes ... 0 / 0 / 0` PASS; `--dct-q 90`: mlp ES 3.746e-07, FD 7.959e-05, latent 3.974e-05, dct snap
0 / 0 / 0, refit moved 193; `--dct-q 20`: mlp ES 3.670e-07, FD 8.316e-05, latent 3.304e-05, dct snap 0 / 0 / 0,
refit moved 196; all passed.
`ntc_decode --compare --verify` on the DCT_NOTES section 3 files (same figures as there):
```
out_model10_b6_c1q2_c4bilin_mlp17_cuda8k_fd50 (v10): PSNR 90.42 dB, max |diff| 1, 1: 149; verify max 0 ulp (0 of 2524500); RGB8 mismatches 150; rc=0
out_model12_b8_c1q2_c4bilin_mlp27_cuda8k_fd50 (v11): PSNR 89.53 dB, max |diff| 1, 1: 513; verify max 0 ulp (0 of 7077888); mismatches 513; rc=0
out_image3_b8_c1q4_c4bilin_qes8_mlp27_cuda8k_fd50 (v12): PSNR 90.66 dB, max |diff| 1, 1: 353; verify max 0 ulp (0 of 6322560); mismatches 339; rc=0
out_chk (v13): PSNR 88.40 dB, max |diff| 1, 1: 74; verify max 0 ulp (0 of 786432); mismatches 75; rc=0
out_dct_a_gpu (v13): PSNR 89.23 dB, max |diff| 1, 1: 61; verify max 0 ulp (0 of 786432); mismatches 60; rc=0
out_m1234 (v12, 4 textures): t0..t3 PSNR 89.45 / 88.28 / 87.84 / 92.04 dB, max |diff| 1; verify max 0 ulp (0 of 3145728); mismatches 243; rc=0
```
Builds: 0 compiler warnings in both trees (CPU: `ntc.vcxproj -> ...ntc.exe`, `ntc_decode.vcxproj -> ...`; CUDA:
the same plus the pre-existing `LNK4098`).

### GCC cross-check (CPU_DECODER_NOTES section 7 setup, WSL, `build_linux`)

`./build_linux/ntc_decode out_ntcb_dct50/model.ntcb -o /tmp/gcc_n --compare out_ntcb_dct50/recon_q_final.png --verify`
prints the same `file     :` line (`payload 4521126 bits matches the simulator exactly`), `PSNR 90.94 dB, max
|diff| 1, 1: 331`, `verify max 0 ulp (0 of 6322560)`, rgb8 hash `9dc7d15e4a6bba1f`, and its PNG is byte-identical
to the MSVC binary's `n.png`; the four `out_ntcb_m1234` PNGs are byte-identical between the two compilers as
well (byte-wise packing, no host-order reads). The GCC build prints two pre-existing warnings outside the NTCB
region (`-Wswitch` in the kernel's positional switch at L710 and the `__m128` template-attribute note at L747).

## 4. Files, regions and hooks (the removal list; line numbers of the tree after the review fixes of section 9)

`grep -n "\[NTCB\]"` finds every hook and both region boundaries. main.cpp (3232 -> 3821 lines):
- L42-47: the DEPENDENCY note item.
- L172 `Options::write_ntcb`; L248-250 the two usage lines; L2893 the parser line.
- L398 `QesLevel::from_ntcb`; L430-437 `q8_index` / `q8_value` in the strict block; L1856 the `save_model` hook (a
  `from_ntcb` level is written as continuous); L1907 `BitrateStats::bits_raw_latent / bits_header`; L1984-1985 the strict
  post-hoc index / value in `bitrate_stats`; L2002 the assignment of the two fields.
- L2006-2490 the region `// ---- NTCB: bit-packed container [NTCB]` ... `// ---- end of the NTCB region [NTCB]`: the
  constants (magic, version, fixed header size, the four sync markers), `NtcbBitWriter / NtcbBitReader / NtcbByteReader`,
  the LE byte helpers, `ntcb_fnv1a32 / ntcb_file_hash / ntcb_finite32 / ntcb_finite16`, `ntcb_f32_to_f16 / ntcb_f16_to_f32 /
  ntcb_selfcheck / ntcb_mlp_round_fp16`, `NtcbLevel / NtcbHeader`, `ntcb_write_header / ntcb_min_section_bytes /
  ntcb_read_header / ntcb_put_section`, `NtcbReport`, `ntcb_write` (with the self-check re-read), `ntcb_restore`.
- L2913 the fp16 self-check hook; L2914 the `--block > 255` refusal; L3124-3144 the loader's `is_ntcb` branch (the v9..v13
  cascade is untouched behind an `else`); L3255-3257 the payload hook; L3270 the specific error message; L3288-3305 the
  mode-0 grid install with its two skip notes; L3313 the no-fp32-shadow note; L3421 the `md.qes_bits` hook; L3548-3566 the
  `ntcb_round` lambda (before / after PSNR); L3668-3669 the loop hook; L3736 the `--iters 0` hook; L3742 `ntcb_rc`;
  L3784-3796 the final-block write and `file:` line; L3819 the exit code.
ntc_decode.cpp (1217 -> 1500 lines):
- L25-28 the DEPENDENCY note; L94-97 the `Model` fields; L112-113 `NTCB_MAGIC` and the forward declaration; L121 the magic
  dispatch in `load_model`; L473-727 the region (`NtcbByteReader / NtcbBitReader / ntcb_file_hash / ntcb_finite32 /
  ntcb_finite16 / ntcb_f16_to_f32 / NtcbLevel / NtcbHeader / ntcb_min_section_bytes / ntcb_read_header / load_ntcb`), inside
  the strict-FP section; L1347 `rc`; L1350 the `model    :` hook; L1364 the level-line hook; L1380-1396 the `file     :` line
  and the MISMATCH exit code.
README.md: the "Bit-packed container" paragraph after "How bitrates are reported", the `--write-ntcb` row and the `--load`
row's note in the flag table, the "Container round trip" paragraph in "Regression testing".
NTCB_PLAN.md: the dated amendment of section 1.1 and the matching edits of 1.2-1.4. NTCB_NOTES.md: this file. cuda/: untouched.
Scratchpad (not in the tree): `ntcb_corrupt.py`, `matrix.sh`, `matrix_all.sh`, the captured outputs.
Output directories written by the runs above (untracked, like every `out_*`): `out_ntcb_dct50`, `out_ntcb_dct50_cpu`,
`out_ntcb_dct50_reload`, `out_ntcb_dct50_reload_cuda`, `out_ntcb_out_chk`, `out_ntcb_out_dct_a_cpu`, `out_ntcb_q4`,
`out_ntcb_q4_reload`, `out_ntcb_q4_reload_cuda`, `out_ntcb_m1234`, `out_ntcb_m1234_reload`, `out_ntcb_m1234_reload_cuda`,
`out_ntcb_m1234v12`, `out_ntcb_m1234v12_reload`, `out_ntcb_m1234v12_reload_cuda`, `out_ntcb_chk5`, `out_ntcb_chk5_reload`,
`out_ntcb_chk5_reload_cuda`, `out_ntcb_nolive`, `out_reg_ntcb`, `out_reg_ntcb_reload`, `out_reg`, `out_chk_g`, `out_ntcb_tmp`.

## 5. Open items

- The container's overhead over the simulator is 0.001-0.005 bpp in the runs above (the fixed header, the
  section headers and, since September 7, 2026, the sync markers). The simulator's side-info charge is stored in
  the header: a mode-0/1 level's per-channel (lo, hi) floats are exactly its 64-bit charge; the DCT side info is
  charged at 96 bits per channel by the simulator but stored as 8(1 + C) bits (dc_step and q); the `--qat`
  palette is not charged by the simulator at all (2^B floats per channel in the header). The fixed header, the
  section headers and the sync markers are extra on top of all of that.
- Section `coding` values 1 and 2 (NTCB_PLAN.md section 5) are refused with "coded sections (coding N) need a
  newer decoder" by both readers; nothing else is reserved for them yet.
- Training on from a container (`--iters > 0`) treats its mode-0 levels as frozen `--qes` levels and prints a
  note; a v12/v13 `model.bin` is the file to continue from when the fp32 shadow matters.

## 8. Settings-independence matrix (September 7, 2026; the quoted lines are from the binaries of section 3, before the review fixes of section 9)

NTCB_PLAN.md section 7, every row, on the binaries of section 3 (`build\Release\ntc.exe`, `build_cuda\Release\ntc.exe`
with `--cuda`, `build\Release\ntc_decode.exe`). Each row is three commands, in the forms of section 3 (a)-(c):

1. conversion: `ntc <images> [--cuda] <row options> --load <dir>/model.bin --iters 0 --write-ntcb --resave --out out_ntcb_<row>`
   (the CUDA build where the row's log has a `cuda :` line, the CPU build otherwise; the options are the ones the
   loader's "was saved with ..." message names for the file, in the `--block N --latent 0 0 C` form of the log);
2. decoder byte identity: `ntc_decode out_ntcb_<row>/model.ntcb -o n_<row> --compare <out_ntcb_<row>/recon_q_final[_tK].png> --verify`,
   the same on `out_ntcb_<row>/model.bin` (the resaved v12/v13 file) -> `b_<row>`, then `cmp` of the two PNGs (one per texture);
3. trainer reload: `ntc <images> [--cuda] <row options> --load out_ntcb_<row>/model.ntcb --iters 0 --out out_ntcb_<row>_reload`, same build.

The three rows without an existing file (3lv-dct, 3lv-qat, t2-dct) are fresh CUDA runs with `--write-ntcb` on the training command
(`--iters 400 --print-every 200 --save-every 200`, `--crop 512` on model.png for the first two, m1.png m2.png for the third, the
options of NTCB_PLAN.md section 7), then steps 2 and 3 on their outputs. The driver script and every captured output are in the
session scratchpad (`matrix.sh`, `conv_<row>.txt`, `dec_n_<row>.txt`, `dec_b_<row>.txt`, `reload_<row>.txt`, `fresh_<row>.txt`).

Result: 27 of 27 rows pass every check. Every `file:` verdict is `OK`; on every row the two decoder PNGs are byte-identical
(`cmp`), `--compare` against the trainer's `recon_q_final` PNG prints max |diff| 1 and PSNR between 87.13 and 95.95 dB,
`--verify` prints `pre-nonlinearity: max 0 ulp`, both `ntc_decode` invocations print the same compare and verify lines, and the
reload's `done:` line equals the converting invocation's in its quantized part (PSNR, raw / entropy / ctx bpp, the `[dct: ...]`
bracket; the trailing time and the fp32 `final psnr` are excluded from the comparison, see the notes below). No refusal, no crash,
no MISMATCH, every rc 0.

Table columns: file bytes / file bpp / simulator raw bpp and the verdict from the conversion's `file:` line; "identical" = `cmp` of the
ntcb-decoded and model.bin-decoded PNGs (all textures); compare / verify from the ntcb `ntc_decode` line (the model.bin line is the
same on every row); the last column is the quantized `psnr` of the converting invocation's `done:` line and of the reload's.
The layout column is the `level` lines `ntc_decode` prints for the ntcb (ranges and the fp32-planar note dropped; "mode-0 8-bit grid"
= its "ntcb q8 (post-hoc grid) 8-bit").

| row | dir | build | layout (ntc_decode level lines, abbreviated) | file bytes | file bpp | sim raw bpp | verdict | decoder PNG identical (ntcb vs model.bin) | compare max diff / PSNR vs recon_q_final | verify (pre-nonlinearity) | reload done-line equal | quantized PSNR (conv / reload) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dct50 | out_image3_b8_dct50_c4bilin_qes8_mlp27_cuda8k_fd50 | cuda | 1480x1424x1 nearest, dct 8x8, q 50, dc step 4, 4-bit scale codes; 185x178x4 bilinear, qes 8-bit | 565294 | 2.146 | 2.145 | OK | yes (1 PNG) | t0: 1 / 90.94 dB | max 0 ulp (0 of 6322560 values differ) | yes | 45.64 / 45.64 |
| dct-b6 | out_model11_b6_dct15_c4bilin_qes8_mlp27_refit500_cuda8k_fd50 | cuda | 888x1032x1 nearest, dct 8x8, q 15, dc step 4, 4-bit scale codes; 148x172x4 bilinear, qes 8-bit | 155241 | 1.355 | 1.354 | OK | yes (1 PNG) | t0: 1 / 88.96 dB | max 0 ulp (0 of 2749248 values differ) | yes | 35.78 / 35.78 |
| dct-b4 | out_m1_b4_dct30_c4bilin_qes8_mlp27_refit500_cuda8k_fd50 | cuda | 512x512x1 nearest, dct 8x8, q 30, dc step 4, 4-bit scale codes; 128x128x4 bilinear, qes 8-bit | 126933 | 3.874 | 3.870 | OK | yes (1 PNG) | t0: 1 / 89.68 dB | max 0 ulp (0 of 786432 values differ) | yes | 26.08 / 26.08 |
| dct-eob-chk | out_chk | cuda | 512x512x1 nearest, dct 8x8, q 50, dc step 4, 4-bit scale codes; 64x64x4 bilinear, qes 8-bit | 30055 | 0.917 | 0.914 | OK | yes (1 PNG) | t0: 1 / 88.11 dB | max 0 ulp (0 of 786432 values differ) | yes | 11.55 / 11.55 |
| dct-eob-cpu | out_dct_a_cpu | cpu | 512x512x1 nearest, dct 8x8, q 50, dc step 4, 4-bit scale codes; 64x64x4 bilinear, qes 8-bit | 177277 | 5.410 | 5.407 | OK | yes (1 PNG) | t0: 1 / 90.19 dB | max 0 ulp (0 of 786432 values differ) | yes | 29.95 / 29.95 |
| qat4 | out_image3_b8_c1q4_c4bilin_qes8_mlp27_cuda8k_fd50_psnr8 | cuda | 1480x1424x1 nearest, qat 4; 185x178x4 bilinear, qes 8-bit | 1187807 | 4.509 | 4.508 | OK | yes (1 PNG) | t0: 1 / 90.67 dB | max 0 ulp (0 of 6322560 values differ) | yes | 48.94 / 48.94 |
| qat1 | out_model15_b8_c1q1_c3bilin_qes8_mlp27_cuda8k_fd50_psnr8 | cuda | 1080x1080x1 nearest, qat 1; 135x135x3 bilinear, qes 8-bit | 202683 | 1.390 | 1.389 | OK | yes (1 PNG) | t0: 1 / 89.21 dB | max 0 ulp (0 of 3499200 values differ) | yes | 28.55 / 28.55 |
| qat2 | out_model15_b8_c1q2_c3bilin_qes8_mlp27_cuda8k_fd50_psnr8 | cuda | 1080x1080x1 nearest, qat 2; 135x135x3 bilinear, qes 8-bit | 348491 | 2.390 | 2.389 | OK | yes (1 PNG) | t0: 1 / 89.37 dB | max 0 ulp (0 of 3499200 values differ) | yes | 32.30 / 32.30 |
| qat3 | out_model15_b8_c1q3_c3bilin_qes8_mlp27_cuda8k_fd50_psnr8 | cuda | 1080x1080x1 nearest, qat 3; 135x135x3 bilinear, qes 8-bit | 494307 | 3.390 | 3.389 | OK | yes (1 PNG) | t0: 1 / 89.22 dB | max 0 ulp (0 of 3499200 values differ) | yes | 35.25 / 35.25 |
| qat-2ch | out_model15_b4_c2q4q2_c4bilin_qes8_mlp27_cuda8k_fd50_psnr8 | cuda | 1080x1080x2 nearest, qat 4,2; 270x270x4 bilinear, qes 8-bit | 1168798 | 8.016 | 8.015 | OK | yes (1 PNG) | t0: 1 / 89.45 dB | max 0 ulp (0 of 3499200 values differ) | yes | 43.15 / 43.15 |
| qes6 | out_image3_b8_c1q4_c4bilin_qes6_mlp27_cuda8k_fd50 | cuda | 1480x1424x1 nearest, qat 4; 185x178x4 bilinear, qes 6-bit | 1154877 | 4.384 | 4.383 | OK | yes (1 PNG) | t0: 1 / 91.04 dB | max 0 ulp (0 of 6322560 values differ) | yes | 48.20 / 48.20 |
| qes10 | out_image3_b8_c1q4_c4bilin_qes10_mlp27_cuda8k_fd50 | cuda | 1480x1424x1 nearest, qat 4; 185x178x4 bilinear, qes 10-bit | 1220737 | 4.634 | 4.633 | OK | yes (1 PNG) | t0: 1 / 90.88 dB | max 0 ulp (0 of 6322560 values differ) | yes | 48.99 / 48.99 |
| c1float-image3 | out_image3_b8_c1float_c4bilin_qes8_mlp27_cuda8k_fd50 | cuda | 1480x1424x1 nearest, mode-0 8-bit grid; 185x178x4 bilinear, qes 8-bit | 2241511 | 8.509 | 8.508 | OK | yes (1 PNG) | t0: 1 / 91.11 dB | max 0 ulp (0 of 6322560 values differ) | yes | 46.85 / 46.85 |
| c1float-model15 | out_model15_b8_c1float_c3bilin_qes8_mlp27_cuda8k_fd50 | cuda | 1080x1080x1 nearest, mode-0 8-bit grid; 135x135x3 bilinear, qes 8-bit | 1223283 | 8.390 | 8.389 | OK | yes (1 PNG) | t0: 1 / 89.68 dB | max 0 ulp (0 of 3499200 values differ) | yes | 39.61 / 39.61 |
| c1qes8 | out_image3_b8_c1qes8_c4bilin_qes8_mlp27_cuda8k_fd50 | cuda | 1480x1424x1 nearest, qes 8-bit; 185x178x4 bilinear, qes 8-bit | 2241511 | 8.509 | 8.508 | OK | yes (1 PNG) | t0: 1 / 91.16 dB | max 0 ulp (0 of 6322560 values differ) | yes | 47.36 / 47.36 |
| c1qes8-c6 | out_image3_b8_c1qes8_c6bilin_qes8_mlp27_cuda8k_fd50 | cuda | 1480x1424x1 nearest, qes 8-bit; 185x178x6 bilinear, qes 8-bit | 2307497 | 8.759 | 8.759 | OK | yes (1 PNG) | t0: 1 / 91.16 dB | max 0 ulp (0 of 6322560 values differ) | yes | 50.83 / 50.83 |
| mat | out_m1234_512c2q2_128c4_cuda8k_fd50 | cuda | 512x512x2 nearest, qat 2,2; 128x128x4 nearest, mode-0 8-bit grid | 200992 | 6.134 | 6.129 | OK | yes (4 PNGs) | t0: 1 / 89.09 dB ; t1: 1 / 87.40 dB ; t2: 1 / 87.95 dB ; t3: 1 / 95.95 dB | max 0 ulp (0 of 3145728 values differ) | yes | 29.38 / 29.38 |
| mat-bilin | out_m1234 | cpu | 128x128x4 bilinear, mode-0 8-bit grid; 64x64x4 bilinear, mode-0 8-bit grid | 86444 | 2.638 | 2.635 | OK | yes (4 PNGs) | t0: 1 / 90.01 dB ; t1: 1 / 87.74 dB ; t2: 1 / 88.51 dB ; t3: 1 / 92.04 dB | max 0 ulp (0 of 3145728 values differ) | yes | 25.45 / 25.45 |
| mat-b6 | out_m1234_b6_c2q44_c4bilin_mlp27_cuda8k_fd50 | cuda | 516x516x2 nearest, qat 4,4; 86x86x4 bilinear, mode-0 8-bit grid | 298790 | 8.978 | 8.970 | OK | yes (4 PNGs) | t0: 1 / 88.70 dB ; t1: 1 / 87.74 dB ; t2: 1 / 89.03 dB ; t3: 1 / 93.29 dB | max 0 ulp (0 of 3195072 values differ) | yes | 32.24 / 32.24 |
| 3lv | out_model6_b6_c2q32_c4_l3c2_mlp26_cuda8k_fd50 | cuda | 960x960x2 nearest, qat 3,2; 160x160x4 nearest, mode-0 8-bit grid; 80x80x2 nearest, mode-0 8-bit grid | 693576 | 6.021 | 6.019 | OK | yes (1 PNG) | t0: 1 / 88.86 dB | max 0 ulp (0 of 2764800 values differ) | yes | 38.99 / 38.99 |
| old-v9 | out_2lv_512c1_128c4_qat2 | cpu | 512x512x1 nearest, qat 2; 128x128x4 nearest, mode-0 8-bit grid | 134701 | 4.111 | 4.107 | OK | yes (1 PNG) | t0: 1 / 89.61 dB | max 0 ulp (0 of 786432 values differ) | yes | 30.76 / 30.76 |
| old-v11 | out_model12_b8_c1q2_c4bilin_mlp27_cuda8k_fd50 | cuda | 2048x1152x1 nearest, qat 2; 256x144x4 bilinear, mode-0 8-bit grid | 739559 | 2.508 | 2.507 | OK | yes (1 PNG) | t0: 1 / 89.95 dB | max 0 ulp (0 of 7077888 values differ) | yes | 40.29 / 40.29 |
| big-b32 | out_model13_b32_c1q3_c4bilin_qes8_mlp27_cuda8k_fd50 | cuda | 864x1280x1 nearest, qat 3; 27x40x4 bilinear, qes 8-bit | 421335 | 3.048 | 3.047 | OK | yes (1 PNG) | t0: 1 / 89.05 dB | max 0 ulp (0 of 3317760 values differ) | yes | 36.79 / 36.79 |
| big-b12 | out_model14_b12_c1q2_c3bilin_qes8_mlp27_cuda8k_fd50 | cuda | 756x1128x1 nearest, qat 2; 63x94x3 bilinear, qes 8-bit | 233174 | 2.187 | 2.186 | OK | yes (1 PNG) | t0: 1 / 89.98 dB | max 0 ulp (0 of 2558304 values differ) | yes | 33.49 / 33.49 |
| 3lv-dct | (fresh run) | cuda | 512x512x1 nearest, dct 8x8, q 50, dc step 4, 4-bit scale codes; 64x64x4 bilinear, qes 6-bit; 32x32x2 bilinear, qes 10-bit | 135997 | 4.150 | 4.146 | OK | yes (1 PNG) | t0: 1 / 90.10 dB | max 0 ulp (0 of 786432 values differ) | yes | 28.78 / 28.78 |
| 3lv-qat | (fresh run) | cuda | 512x512x1 nearest, qat 2; 64x64x4 bilinear, qes 8-bit; 32x32x2 bilinear, qes 8-bit | 86393 | 2.637 | 2.632 | OK | yes (1 PNG) | t0: 1 / 88.89 dB | max 0 ulp (0 of 786432 values differ) | yes | 28.95 / 28.95 |
| t2-dct | (fresh run) | cuda | 512x512x1 nearest, dct 8x8, q 30, dc step 4, 4-bit scale codes; 128x128x4 bilinear, qes 8-bit | 148905 | 4.544 | 4.541 | OK | yes (2 PNGs) | t0: 1 / 90.28 dB ; t1: 1 / 87.13 dB | max 0 ulp (0 of 1572864 values differ) | yes | 26.62 / 26.62 |

Row options (the images and the `<row options>` of steps 1 and 3):

- dct50 (cuda): `image3.png` `--block 8 --latent 0 0 1 --latent2 0 0 4 --filter nearest,bilinear --pos lv1local --dct-q 50 --qes 0,8 --mlp 27,27 --leak 0.0009765625`
- dct-b6 (cuda): `model11.png` `--block 6 --latent 0 0 1 --latent2 0 0 4 --filter nearest,bilinear --pos lv1local --dct-q 15 --qes 0,8 --mlp 27,27 --leak 0.0009765625`
- dct-b4 (cuda): `m1.png` `--block 4 --latent 0 0 1 --latent2 0 0 4 --filter nearest,bilinear --pos lv1local --dct-q 30 --qes 0,8 --mlp 27,27 --leak 0.0009765625`
- dct-eob-chk (cuda): `chief1.png` `--block 8 --latent 0 0 1 --latent2 0 0 4 --filter nearest,bilinear --pos lv1local --dct-q 50 --qes 0,8 --mlp 17,17 --leak 0.0009765625`
- dct-eob-cpu (cpu): `chief1.png` `--block 8 --latent 0 0 1 --latent2 0 0 4 --filter nearest,bilinear --pos lv1local --dct-q 50 --qes 0,8 --mlp 17,17 --leak 0.0009765625`
- qat4 (cuda): `image3.png` `--block 8 --latent 0 0 1 --latent2 0 0 4 --filter nearest,bilinear --pos lv1local --qat 4 --qes 0,8 --mlp 27,27 --leak 0.0009765625`
- qat1 (cuda): `model15.png` `--block 8 --latent 0 0 1 --latent2 0 0 3 --filter nearest,bilinear --pos lv1local --qat 1 --qes 0,8 --mlp 27,27 --leak 0.0009765625`
- qat2 (cuda): `model15.png` `the qat1 options with --qat 2`
- qat3 (cuda): `model15.png` `the qat1 options with --qat 3`
- qat-2ch (cuda): `model15.png` `--block 4 --latent 0 0 2 --latent2 0 0 4 --filter nearest,bilinear --pos lv1local --qat 4,2 --qes 0,8 --mlp 27,27 --leak 0.0009765625`
- qes6 (cuda): `image3.png` `--block 8 --latent 0 0 1 --latent2 0 0 4 --filter nearest,bilinear --pos lv1local --qat 4 --qes 0,6 --mlp 27,27 --leak 0.0009765625`
- qes10 (cuda): `image3.png` `the qes6 options with --qes 0,10`
- c1float-image3 (cuda): `image3.png` `--block 8 --latent 0 0 1 --latent2 0 0 4 --filter nearest,bilinear --pos lv1local --qes 0,8 --mlp 27,27 --leak 0.0009765625`
- c1float-model15 (cuda): `model15.png` `--block 8 --latent 0 0 1 --latent2 0 0 3 --filter nearest,bilinear --pos lv1local --qes 0,8 --mlp 27,27 --leak 0.0009765625`
- c1qes8 (cuda): `image3.png` `--block 8 --latent 0 0 1 --latent2 0 0 4 --filter nearest,bilinear --pos lv1local --qes 8,8 --mlp 27,27 --leak 0.0009765625`
- c1qes8-c6 (cuda): `image3.png` `--block 8 --latent 0 0 1 --latent2 0 0 6 --filter nearest,bilinear --pos lv1local --qes 8,8 --mlp 27,27 --leak 0.0009765625`
- mat (cuda): `m1.png m2.png m3.png m4.png` `--latent 512 512 2 --latent2 128 128 4 --filter nearest,nearest --pos lv1local --qat 2 --mlp 36,36 --leak 0.0009765625`
- mat-bilin (cpu): `m1.png m2.png m3.png m4.png` `--latent 128 128 4 --latent2 64 64 4 --mlp 36,36`
- mat-b6 (cuda): `m1.png m2.png m3.png m4.png` `--block 6 --latent 0 0 2 --latent2 0 0 4 --filter nearest,bilinear --pos lv1local --qat 4,4 --mlp 27,27 --leak 0.0009765625`
- 3lv (cuda): `model6.png` `--block 6 --latent 0 0 2 --latent2 0 0 4 --latent3 0 0 2 --filter nearest,nearest,nearest --pos lv1local --qat 3,2 --mlp 26,26 --leak 0.0009765625`
- old-v9 (cpu): `mario_512.png` `--latent 512 512 1 --latent2 128 128 4 --filter nearest,nearest --pos lv1local --qat 2 --mlp 36,36`
- old-v11 (cuda): `model12.png` `--block 8 --latent 0 0 1 --latent2 0 0 4 --filter nearest,bilinear --pos lv1local --qat 2 --mlp 27,27 --leak 0.0009765625`
- big-b32 (cuda): `model13.png` `--block 32 --latent 0 0 1 --latent2 0 0 4 --filter nearest,bilinear --pos lv1local --qat 3 --qes 0,8 --mlp 27,27 --leak 0.0009765625`
- big-b12 (cuda): `model14.png` `--block 12 --latent 0 0 1 --latent2 0 0 3 --filter nearest,bilinear --pos lv1local --qat 2 --qes 0,8 --mlp 27,27 --leak 0.0009765625`
- 3lv-dct (cuda, fresh): `model.png --cuda --crop 512 --iters 400 --print-every 200 --save-every 200 --block 8 --latent 0 0 1 --latent2 0 0 4 --latent3 0 0 2 --filter nearest,bilinear,bilinear --pos lv1local --dct-q 50 --dct-refit 100 --qes 0,6,10 --mlp 27,27 --leak 0.0009765625 --write-ntcb --out out_ntcb_3lv-dct`
- 3lv-qat (cuda, fresh): the same with `--qat 2 --qes 0,8,8` instead of `--dct-q 50 --dct-refit 100 --qes 0,6,10`, `--out out_ntcb_3lv-qat`
- t2-dct (cuda, fresh): `m1.png m2.png --cuda --iters 400 --print-every 200 --save-every 200 --block 4 --latent 0 0 1 --latent2 0 0 4 --filter nearest,bilinear --pos lv1local --dct-q 30 --qes 0,8 --mlp 27,27 --leak 0.0009765625 --write-ntcb --out out_ntcb_t2-dct`

The verbatim `file:` line of every row (the conversion's; for the three fresh rows the training run's):

```
dct50: file: out_ntcb_dct50/model.ntcb 565294 bytes = 2.146 bpp (simulator raw 2.145 bpp); payload 4521126 bits = simulator 4521478 - 352 side-info bits [OK]; container header 1032 + section headers 192 + padding 2 bits
dct-b6: file: out_ntcb_dct-b6/model.ntcb 155241 bytes = 1.355 bpp (simulator raw 1.354 bpp); payload 1240701 bits = simulator 1241053 - 352 side-info bits [OK]; container header 1032 + section headers 192 + padding 3 bits
dct-b4: file: out_ntcb_dct-b4/model.ntcb 126933 bytes = 3.874 bpp (simulator raw 3.870 bpp); payload 1014240 bits = simulator 1014592 - 352 side-info bits [OK]; container header 1032 + section headers 192 + padding 0 bits
dct-eob-chk: file: out_ntcb_dct-eob-chk/model.ntcb 30055 bytes = 0.917 bpp (simulator raw 0.914 bpp); payload 239216 bits = simulator 239568 - 352 side-info bits [OK]; container header 1032 + section headers 192 + padding 0 bits
dct-eob-cpu: file: out_ntcb_dct-eob-cpu/model.ntcb 177277 bytes = 5.410 bpp (simulator raw 5.407 bpp); payload 1416989 bits = simulator 1417341 - 352 side-info bits [OK]; container header 1032 + section headers 192 + padding 3 bits
qat4: file: out_ntcb_qat4/model.ntcb 1187807 bytes = 4.509 bpp (simulator raw 4.508 bpp); payload 9500736 bits = simulator 9500992 - 256 side-info bits [OK]; container header 1528 + section headers 192 + padding 0 bits
qat1: file: out_ntcb_qat1/model.ntcb 202683 bytes = 1.390 bpp (simulator raw 1.389 bpp); payload 1620264 bits = simulator 1620456 - 192 side-info bits [OK]; container header 1008 + section headers 192 + padding 0 bits
qat2: file: out_ntcb_qat2/model.ntcb 348491 bytes = 2.390 bpp (simulator raw 2.389 bpp); payload 2786664 bits = simulator 2786856 - 192 side-info bits [OK]; container header 1072 + section headers 192 + padding 0 bits
qat3: file: out_ntcb_qat3/model.ntcb 494307 bytes = 3.390 bpp (simulator raw 3.389 bpp); payload 3953064 bits = simulator 3953256 - 192 side-info bits [OK]; container header 1200 + section headers 192 + padding 0 bits
qat-2ch: file: out_ntcb_qat-2ch/model.ntcb 1168798 bytes = 8.016 bpp (simulator raw 8.015 bpp); payload 9348528 bits = simulator 9348784 - 256 side-info bits [OK]; container header 1664 + section headers 192 + padding 0 bits
qes6: file: out_ntcb_qes6/model.ntcb 1154877 bytes = 4.384 bpp (simulator raw 4.383 bpp); payload 9237296 bits = simulator 9237552 - 256 side-info bits [OK]; container header 1528 + section headers 192 + padding 0 bits
qes10: file: out_ntcb_qes10/model.ntcb 1220737 bytes = 4.634 bpp (simulator raw 4.633 bpp); payload 9764176 bits = simulator 9764432 - 256 side-info bits [OK]; container header 1528 + section headers 192 + padding 0 bits
c1float-image3: file: out_ntcb_c1float-image3/model.ntcb 2241511 bytes = 8.509 bpp (simulator raw 8.508 bpp); payload 17930816 bits = simulator 17931136 - 320 side-info bits [OK]; container header 1080 + section headers 192 + padding 0 bits
c1float-model15: file: out_ntcb_c1float-model15/model.ntcb 1223283 bytes = 8.390 bpp (simulator raw 8.389 bpp); payload 9785064 bits = simulator 9785320 - 256 side-info bits [OK]; container header 1008 + section headers 192 + padding 0 bits
c1qes8: file: out_ntcb_c1qes8/model.ntcb 2241511 bytes = 8.509 bpp (simulator raw 8.508 bpp); payload 17930816 bits = simulator 17931136 - 320 side-info bits [OK]; container header 1080 + section headers 192 + padding 0 bits
c1qes8-c6: file: out_ntcb_c1qes8-c6/model.ntcb 2307497 bytes = 8.759 bpp (simulator raw 8.759 bpp); payload 18458560 bits = simulator 18459008 - 448 side-info bits [OK]; container header 1224 + section headers 192 + padding 0 bits
mat: file: out_ntcb_mat/model.ntcb 200992 bytes = 6.134 bpp (1.533/tex) (simulator raw 6.129 bpp (1.532/tex)); payload 1606464 bits = simulator 1606720 - 256 side-info bits [OK]; container header 1280 + section headers 192 + padding 0 bits
mat-bilin: file: out_ntcb_mat-bilin/model.ntcb 86444 bytes = 2.638 bpp (0.660/tex) (simulator raw 2.635 bpp (0.659/tex)); payload 690112 bits = simulator 690624 - 512 side-info bits [OK]; container header 1248 + section headers 192 + padding 0 bits
mat-b6: file: out_ntcb_mat-b6/model.ntcb 298790 bytes = 8.978 bpp (2.244/tex) (simulator raw 8.970 bpp (2.243/tex)); payload 2388080 bits = simulator 2388336 - 256 side-info bits [OK]; container header 2048 + section headers 192 + padding 0 bits
3lv: file: out_ntcb_3lv/model.ntcb 693576 bytes = 6.021 bpp (simulator raw 6.019 bpp); payload 5546704 bits = simulator 5547088 - 384 side-info bits [OK]; container header 1648 + section headers 256 + padding 0 bits
old-v9: file: out_ntcb_old-v9/model.ntcb 134701 bytes = 4.111 bpp (simulator raw 4.107 bpp); payload 1076272 bits = simulator 1076528 - 256 side-info bits [OK]; container header 1144 + section headers 192 + padding 0 bits
old-v11: file: out_ntcb_old-v11/model.ntcb 739559 bytes = 2.508 bpp (simulator raw 2.507 bpp); payload 5915136 bits = simulator 5915392 - 256 side-info bits [OK]; container header 1144 + section headers 192 + padding 0 bits
big-b32: file: out_ntcb_big-b32/model.ntcb 421335 bytes = 3.048 bpp (simulator raw 3.047 bpp); payload 3369216 bits = simulator 3369472 - 256 side-info bits [OK]; container header 1272 + section headers 192 + padding 0 bits
big-b12: file: out_ntcb_big-b12/model.ntcb 233174 bytes = 2.187 bpp (simulator raw 2.186 bpp); payload 1864128 bits = simulator 1864320 - 192 side-info bits [OK]; container header 1072 + section headers 192 + padding 0 bits
3lv-dct: file: out_ntcb_3lv-dct/model.ntcb 135997 bytes = 4.150 bpp (simulator raw 4.146 bpp); payload 1086444 bits = simulator 1086924 - 480 side-info bits [OK]; container header 1272 + section headers 256 + padding 4 bits
3lv-qat: file: out_ntcb_3lv-qat/model.ntcb 86393 bytes = 2.637 bpp (simulator raw 2.632 bpp); payload 689504 bits = simulator 689888 - 384 side-info bits [OK]; container header 1384 + section headers 256 + padding 0 bits
t2-dct: file: out_ntcb_t2-dct/model.ntcb 148905 bytes = 4.544 bpp (2.272/tex) (simulator raw 4.541 bpp (2.270/tex)); payload 1190016 bits = simulator 1190368 - 352 side-info bits [OK]; container header 1032 + section headers 192 + padding 0 bits
```

The three fresh runs' `done:` lines (the training run; the reload printed the same quantized part):

```
done: final psnr 28.78 dB (best 29.13) at fp32 34.385 bpp | --dct-q 50 level 0 (bit simulator) + --qes 0,6,10 latent + fp16 mlp: psnr 28.78 dB at 4.146 bpp raw, 1.812 bpp entropy-coded, 1.648 bpp with an (up, left) context on level 0 [level 0 alone: 1.176] [dct: raw 3.624 h0 1.339 ctx 1.176 code raw/ctx 0.062/0.011 bpp; nz 13.31/blk eob0 0.6%] | 1.6s
done: final psnr 28.95 dB (best 28.95) at fp32 34.385 bpp | --qes 0,8,8 latent + fp16 mlp: psnr 28.95 dB at 2.632 bpp raw, 2.483 bpp entropy-coded, 1.161 bpp with an (up, left) context on level 0 [level 0 alone: 0.572] | 1.4s
done: final psnr 26.62 dB (best 26.62) tex 24.26 32.19 at fp32 40.139 bpp (20.070/tex) | --dct-q 30 level 0 (bit simulator) + --qes 0,8 latent + fp16 mlp: psnr 26.62 dB qtex 24.26 32.19 at 4.541 bpp raw, 2.559 bpp entropy-coded, 2.468 bpp with an (up, left) context on level 0 (2.270, 1.280, 1.234 /tex) [level 0 alone: 0.674] [dct: raw 2.470 h0 0.765 ctx 0.674 code raw/ctx 0.062/0.001 bpp; nz 8.69/blk eob0 0.0%] | 1.4s
```

### Failures

None. No row failed a step, no verdict was MISMATCH, no loader or decoder message was printed on any row, every exit code was 0.
The three-level file `out_model6_b6_c2q32_c4_l3c2_mlp26_cuda8k_fd50/model.bin` (the plan's "older format" row) loaded as it is
(`loaded   : out_model6_b6_c2q32_c4_l3c2_mlp26_cuda8k_fd50/model.bin`, no version message), so no substitute run was needed for it;
its container has three level sections (`section headers 256` bits) with levels 1 and 2 reinstalled as frozen 8-bit grids
(`ntcb     : level 1: 8-bit grid restored from the file ...`, `ntcb     : level 2: ...` on reload).

### Notes and anomalies (none is a container failure; each is quoted from program output)

1. **fp32 `final psnr` of the reload on mode-0 rows.** The rows with a mode-0 level (c1float-image3, c1float-model15, mat, mat-b6,
   3lv, old-v9, old-v11; mat-bilin's two mode-0 levels agree at two decimals) print a lower fp32 `final psnr` on reload than the
   converting invocation, because the container carries no fp32 shadow (section 2 item 5): conv / reload `final psnr` =
   c1float-image3 47.51 / 46.85, c1float-model15 39.74 / 39.61, mat 29.40 / 29.38, mat-b6 32.25 / 32.24, 3lv 39.09 / 38.99,
   old-v9 30.80 / 30.76, old-v11 40.50 / 40.29, mat-bilin 25.45 / 25.45. On these rows the reload's fp32 figure equals its own
   quantized figure, and the quantized figures of the two invocations are equal on every row (last table column).
2. **Runs logged before the PSNR redefinition.** On eight rows the converting invocation's quantized PSNR is below the row's log's
   final `done:` figure: qes6 48.20 vs 48.54, qes10 48.99 vs 49.31, big-b32 36.79 vs 36.85, big-b12 33.49 vs 33.54,
   old-v11 40.29 vs 40.35, 3lv 38.99 vs 39.03, mat 29.38 vs 29.39, mat-b6 32.24 vs 32.27 (old-v9 30.76 vs 30.76 and mat-bilin 25.45 vs 25.45, also pre-redefinition logs, are equal at two decimals).
   This is not the container: loading the same `model.bin` with `--iters 0` and no `--write-ntcb` (CUDA build) already prints the
   lower figure (`iter      0 ... psnr  48.20 dB` and `done: ... psnr 48.20 dB at 4.383 bpp raw` for qes6; 48.98 for qes10, 36.79
   for big-b32, 33.49 for big-b12, 40.30 for old-v11, 38.99 for 3lv), and `ntc_decode` on the original, unconverted `model.bin`
   matches the run's own `recon_q_final.png` (qes6: PSNR 90.63 dB, max |diff| 1, 355 pixels at 1; big-b32: 89.22 dB, 255 at 1),
   so the saved file is the run's final state. Every one of these logs is dated before the evening of September 6, 2026
   (qes6 / qes10 05:13 / 05:15, old-v11 04:18, big-b32 / big-b12 06:22 / 06:31, 3lv Sep 5 17:33, mat Sep 5 13:53, mat-b6
   Sep 5 21:20, old-v9 Sep 5 01:15), i.e. before the basisu PSNR definition of README "PSNR definition"; every row whose log
   postdates it (dct50, dct-b6, dct-b4, dct-eob-cpu, the `_psnr8` qat rows, c1float, c1qes8, c1qes8-c6) reproduces its log's
   figure exactly. The fp16 rounding itself moved the quantized PSNR by at most 0.01 dB at two decimals on the seven rows probed this way (old-v11:
   40.30 plain load -> 40.29 after rounding; qes10: 48.98 -> 48.99; the other probed rows unchanged).
3. **The fp16 rounding vs the run's PNG.** The converted `recon_q_final.png` differs from the run's original one on the rows
   checked (`cmp` reports a difference on qes6, big-b32, qat4), as expected from the rounded weights; `ntc_decode` on
   `out_ntcb_qes6/model.ntcb` against the run's original `recon_q_final.png` prints PSNR 68.86 dB, max |diff| 1, 53402 pixels
   (0.8458%) at 1. The byte-identity check of step 2 is against the converting invocation's PNG, which is written from the
   rounded weights, and there every row prints max |diff| 1 with 87-96 dB.
4. **`--compare` on padded rows.** Rows whose image was padded (dct50 1478 -> 1480 wide, big-b32 853 -> 864, mat-b6 512 -> 516,
   dct-b6 882x1024 -> 888x1032) print `... is WxH, the output is W'xH'; comparing the common W'xH'` before the compare line; the
   figures quoted are over that common extent.
5. The conversion of the `out_chk` row (dct-eob-chk) and `out_dct_a_cpu` (dct-eob-cpu) reproduces section 3 (a) byte for byte
   (30055 and 177277 bytes, the same `file:` lines); dct50, qat4, mat and mat-bilin reproduce section 3 (a), (d) and (e)
   (565294, 1187807, 200992 and 86444 bytes).

Output directories of this section: `out_ntcb_<row>` and `out_ntcb_<row>_reload` for the 27 rows named in the table
(`out_ntcb_probe` was the loader-message probe, nothing of value in it).

## 9. Review fixes and re-verification (September 7, 2026)

Two reviews of the uncommitted container plus one new requirement (sync markers) were applied on the same day, on top of the
state of sections 1-8. Nothing is committed. The layout changed in place (version stays 1): NTCB_PLAN.md section 1.1 carries the
dated amendment. Builds: `cmake --build build --config Release` and `cmake --build build_cuda --config Release`, 0 compiler
warnings in both (the CUDA link prints only the pre-existing `LNK4098`). Every figure below is copied from program output.

### The fixes, one line of evidence each (file:line of the current tree)

- **A, sync markers.** main.cpp `NTCB_SYNC_LEVEL 0x2B / _CHANNEL 0x16 / _MLP 0x31 / _END 0x0C` (main.cpp L2022-2025, ntc_decode.cpp
  L480-481; pairwise Hamming distance 5, 3, 4, 4, 3, 5), written by `NtcbBitWriter::marker` at the start and end of every level
  body, before every channel of a DCT body and at the start and end of the MLP body; checked by the `sync` lambda of
  `ntcb_restore` and the `marker` lambda of `load_ntcb`. Accounted as container overhead: `sync 42 bits` on the dct50 file line,
  `sync 36` on a two-level grid file, `48` on three levels, `54` on the three-level dct file; `8 * bytes = content + header +
  section headers + sync + padding` on every row (dct50: 8 x 565301 = 4522408 = 4521126 + 1040 + 192 + 42 + 8). Refusals:
  `sync marker mismatch before level 0 body`, `... before level 0 channel 0`, `... after level 1 body`, `... before the mlp
  body`, `... after the mlp body` (the corruption runs below).
- **B1a, size check before allocation.** `ntcb_min_section_bytes` (main.cpp L2171, ntc_decode.cpp L527) and the check at the end of
  both `ntcb_read_header`s; level dims capped at 2^20, the decode size at 16384 x 16384. Evidence: `level 0 W = 2^20, img_w =
  16384 (re-hashed): the minimum-size check | ntc_decode rc=1: the header's sections need at least 268505222 bytes but the file
  has 200838 after the header (truncated or corrupt)` (mat container); `img_w = 16385 (re-hashed) ... bad image size fields
  (decode size at most 16384 x 16384, source size within it)`.
- **B1b, whole-file hash.** `ntcb_file_hash` (main.cpp L2065, ntc_decode.cpp L501): FNV-1a 32 over every byte with offsets 28..31
  read as zero; `one header byte flipped: img_w bit 8 (hash stale) | ntc_decode rc=1: file hash mismatch (corrupt file)`.
- **B2, strict mode-0 dequantization.** `q8_index` / `q8_value` inside the strict block (main.cpp L433-437); `bitrate_stats`
  uses them (L1984-1985) and `ntcb_write` refuses on `q8_value(k, lo, range, levels) != zpost[i]` (L2349). The check fired on the
  first mode-0 row before the move: `mode-0 level 1 texel 1: file value != decoded value (fast-math drift): writer refuses`
  (the `mat` conversion, rc 1). After the move every mode-0 row passes (table below). Effect on the regression pin: none on
  the pinned figures (`iter 200 mse 269.314 psnr 23.83 dB`, `done: ... 23.83 dB at 0.552 bpp raw, 0.500 bpp entropy-coded,
  0.084 bpp`, sha256 of `out_reg/model.bin` `cb88470777e1d041c8803bcf3ca531434b9a2b2b5bce2ef88e031fbc86d69600` = DCT_NOTES
  section 3; model.bin stores the fp32 shadow, which the change does not touch). What did change: `recon_q_final.png` of a
  run with a post-hoc level is now written from the strict values, so `ntc_decode out_reg/model.bin --compare
  out_reg/recon_q_final.png --verify` prints `PSNR 88.76 dB, max |diff| 1, 1: 68 (0.0086%)`, `RGB8 mismatches: 68` where
  DCT_NOTES quotes 88.70 dB / 69 / 68 (ntc_decode's own `q8_quantize` was strict all along; the trainer now agrees with it
  on every texel). The same shows on the matrix rows with a mode-0 level: `mat` t1 87.54 dB (was 87.40), t2 87.79 (87.95);
  `mat-bilin` t0 90.10 (90.01), t1 87.69 (87.74), t2 88.34 (88.51), t3 91.90 (92.04); `mat-b6` t2 89.09 (89.03), t3 93.66
  (93.29); `c1float-image3` 91.25 (91.11); `c1float-model15` 89.64 (89.68); `old-v9` 89.68 (89.61); `old-v11` 90.04 (89.95);
  `big-b12` bpp 2.188 (2.187; the file grew 7 bytes). The `verify` figure stays `max 0 ulp` everywhere.
- **B3, frozen-grid install guard.** main.cpp L3293-3294: a mode-0 grid is skipped with a note when `o.qes_ch[l] != 0` or when
  `l == 0` and `--qat` / `--dct-q` make level 0 discrete. Evidence (out_ntcb_c1float-image3/model.ntcb, a mode-0 level 0):
  ```
  --qat 4 --qes 0,8:  ntcb     : level 0: the file's 8-bit post-hoc grid is not installed: this run makes level 0 discrete with --qat (warm start from the file's values)   (rc 0; the qat snap note printed 2107520 values snapped)
  --qes 8,8:          ntcb     : level 0: the file's 8-bit post-hoc grid is not installed: this run quantizes the level with --qes 8 (warm start from the file's values)   (rc 0; ranges fitted at iteration 0)
  --dct-q 50 --qes 0,8: ntcb   : level 0: the file's 8-bit post-hoc grid is not installed: this run makes level 0 discrete with --dct-q (warm start from the file's values)   (rc 0; codes fitted at iteration 0, psnr 29.75 -> 29.40 dB)
  ```
- **B4, self-check.** `ntcb_write` (main.cpp L2375-2390) re-reads the finished bytes with `ntcb_read_header` + `ntcb_restore`
  into a copy of the `Decoder` and memcmps symbols, codes, grid values (against `zpost`), ranges, palettes and weights;
  `self-check OK` on every `file:` line of the matrix (27 of 27); a difference prints `self-check: re-read file differs
  from written state (level l | mlp | <reader message>)` and refuses.
- **B5, `--resave` relabel.** `QesLevel::from_ntcb` (main.cpp L398); `save_model` writes such a level with bits 0 and its
  on-grid values (L1856); `ntcb_write` writes it as mode 0 again (L2300). Round trip: `--load out_ntcb_mat/model.ntcb --iters 0
  --resave --out out_ntcb_mat_resave` then `--load out_ntcb_mat_resave/model.bin --iters 0` with the original options
  (`--qat 2 --mlp 36,36`, no `--qes`): no mismatch message, no `qes` line, and the same done line both times:
  ```
  done: final psnr 29.38 dB (best -1.00) tex 25.96 33.45 29.32 33.40 at fp32 72.256 bpp (18.064/tex) | 8-bit latent + fp16 mlp: psnr 29.38 dB qtex 25.96 33.45 29.32 33.40 at 6.129 bpp raw, 5.649 bpp entropy-coded, 5.111 bpp with an (up, left) context on level 0 (1.532, 1.412, 1.278 /tex) [level 0 alone: 3.324] | 0.5s
  ```
- **B6, finiteness by bit pattern.** `ntcb_finite32` / `ntcb_finite16` (main.cpp L2071-2072, ntc_decode.cpp L506-507) replace
  every `std::isfinite` in the NTCB code of both files (`grep -n isfinite` leaves only the pre-existing `--weights` and v12
  range checks in main.cpp). The GCC build (`-ffast-math`) rebuilt and cross-checked below.
- **B7, `md.qes_bits`.** main.cpp L3421: `(o.qes && o.qes_ch[l] > 0) ? o.qes_ch[l] : (D.qes[l].fitted() ? D.qes[l].bits : 0)`.
  `build_cuda\Release\ntc.exe image3.png --cuda ... --qes 0,8 --load out_ntcb_c1float-image3/model.ntcb --iters 0`: `ntcb :
  level 0: 8-bit grid restored ...`, `done: ... 46.85 dB at 8.508 bpp raw` = the converting invocation's (the row also passed
  before the fix, so the skipped envelope check had no visible effect on this file; the fix makes the device receive the depth).
- **B8, `--block > 255`.** main.cpp L2914 (`--write-ntcb: --block N does not fit the container's 8-bit latent_cell field (at
  most 255)`, rc 1, before anything runs) and the same guard in `ntcb_write` (L2283); `latent_cell = o.block` is no longer saturated.
- **C1, bpp denominator.** Both file lines end with `bpp over the WxH decode size, as every bpp in this log` (quoted below).
- **C2, before -> after.** The `ntcb_round` lambda (main.cpp L3554-3566) decodes once before the rounding and leaves the
  post-rounding decode in `recon` / `m` / `mse` (the loop's own decode is skipped when it ran), printing
  `ntcb     : mlp weights rounded to fp16 (843 of 843 changed, max |dw| 4.741e-04); psnr 23.83 -> 23.83 dB; figures from here on are post-rounding`
  (the training-loop path, `out_reg_ntcb`), `... (1056 of 1056 changed, max |dw| 4.784e-04); psnr 45.64 -> 45.64 dB; ...`
  (the `--iters 0` path, dct50) and `... (2100 of 2100 changed, max |dw| 9.527e-04); psnr 29.40 -> 29.40 dB; ...` (mat).
- **C3, "content".** Both file lines say `content N bits` (below); "file hash" names the hashed region, which is now the whole file.
- **C4, exit code.** ntc_decode's `rc` is declared before the summary (L1347) and set to 1 on a MISMATCH (L1395); section 2 item 4 updated.
- **C5, side-info wording.** `- 352 side-info bits (stored in the header)` on the trainer line, `+ 352 side-info (stored in the
  header)` on ntc_decode's; section 5 corrected (qat palette uncharged by the simulator, DCT side info 96 charged vs 8(1 + C)
  stored, header / section headers / markers extra).
- **C6, format.** T channel bytes at offset 52 (`h.cpt`, both readers refuse a count other than 3: `texture 0 has 4 channels;
  this build decodes 3 per texture`), a reserved byte at 39 (`reserved header byte 1 ... header field out of range (...
  reserved byte 0)`), MLP coding 0 = raw fp16 and `latent_cell` / `dct_block` informational, all in NTCB_PLAN.md 1.1's
  amendment and the README.
- **C7, README figures.** The "hundredth of a dB" sentence is replaced by the quoted `psnr 23.83 -> 23.83 dB`, mse
  269.314 -> 269.324, `45.64 -> 45.64` and `29.40 -> 29.40` lines.
- **C8, tags.** Section 1 reworded; the two untagged blank lines removed (`git diff` of the two NTCB hunks now consists of
  tagged lines only, plus the untouched context).
- **C9, note on load.** `ntcb     : the container holds no fp32 shadow: the fp32 \`final psnr\` of the done: line is decoded
  from the file's quantized values` on every `--load model.ntcb` (quoted in B3 / B5 above).
- **C10, `/tex` in ntc_decode.** `file     : out_ntcb_mat/model.ntcb 201002 bytes = 6.134 bpp (1.534/tex) (simulator raw 6.129
  bpp (1.532/tex): ...` (T = 4); absent for T = 1.

### The regression pin, the cuda-check lines and the GCC build

```
build\Release\ntc.exe --crop 512 --mlp-pairs 32 --iters 200 --print-every 200 --save-every 200 --out out_reg
iter    200  mse 269.314  psnr  23.83 dB  best  23.83 | q8 psnr  23.83 @ 0.552 bpp (ent 0.500, ctx 0.084) | mlp batch 0.00442 dstd 9.65e-05 | lat mean -0.051 sd 0.551 max 2.08 | 7.5s (26.67 it/s)
done: final psnr 23.83 dB (best 23.83) at fp32 2.103 bpp | 8-bit latent + fp16 mlp: psnr 23.83 dB at 0.552 bpp raw, 0.500 bpp entropy-coded, 0.084 bpp with an (up, left) context on level 0 [level 0 alone: 0.032] | 7.7s
sha256 out_reg/model.bin cb88470777e1d041c8803bcf3ca531434b9a2b2b5bce2ef88e031fbc86d69600 (= DCT_NOTES section 3); no out_reg/model.ntcb
```
The six README `--cuda-check` lines and the `--dct-q 90` / `--dct-q 20` variants (`build_cuda\Release\ntc.exe`): every one
`cuda-check: all passed`, rc 0 (6 / 5 / 6 / 6 / 5 / 7 / 7 / 7 PASS lines, 0 FAIL).
`ntc_decode --compare --verify` on `out_chk/model.bin` (v13): `PSNR 88.40 dB, max |diff| 1, 1: 74`, `max 0 ulp (0 of 786432)`,
`mismatches 75` (= section 3 (g)); on `out_model12_..._fd50/model.bin` (v11): `89.53 dB, 1: 513`, `max 0 ulp`, `513` (= section
3 (g)); on `out_reg/model.bin`: `88.76 dB, 1: 68`, `max 0 ulp`, `68` (the B2 effect above; DCT_NOTES: 88.70 / 69 / 68).
GCC (WSL, `build_linux`, `-O3 -ffast-math`, rebuilt from the current ntc_decode.cpp; the same two pre-existing warnings outside
the region): `./build_linux/ntc_decode out_ntcb_dct50/model.ntcb ... --verify` prints `PSNR 90.94 dB, max |diff| 1, 1: 331`,
`max 0 ulp (0 of 6322560)`; `out_ntcb_mat/model.ntcb` prints the same `file     :` line as MSVC (`content 1606464 bits matches
the simulator exactly; ... sync 36 + padding 12 bits`) and t0..t3 `89.09 / 87.54 / 87.79 / 95.95 dB`, `max 0 ulp (0 of
3145728)`; all five PNGs byte-identical to the MSVC binary's (`cmp`); the flipped level-0 marker is refused with `sync marker
mismatch before level 0 body`, rc 1.

### The regression command with the flag (the training-loop hook)

```
build\Release\ntc.exe --crop 512 --mlp-pairs 32 --iters 200 --print-every 200 --save-every 200 --write-ntcb --out out_reg_ntcb
ntcb     : mlp weights rounded to fp16 (843 of 843 changed, max |dw| 4.741e-04); psnr 23.83 -> 23.83 dB; figures from here on are post-rounding
iter    200  mse 269.324  psnr  23.83 dB  best  23.83 | q8 psnr  23.83 @ 0.552 bpp (ent 0.500, ctx 0.084) | mlp batch 0.00442 dstd 9.65e-05 | lat mean -0.051 sd 0.551 max 2.08 | 7.4s (26.90 it/s)
file: out_reg_ntcb/model.ntcb 18199 bytes = 0.555 bpp (simulator raw 0.552 bpp); content 144560 bits = simulator 144816 - 256 side-info bits (stored in the header) [OK]; container header 872 + section headers 128 + sync 24 + padding 8 bits; bpp over the 512x512 decode size, as every bpp in this log; self-check OK
done: final psnr 23.83 dB (best 23.83) at fp32 2.103 bpp | 8-bit latent + fp16 mlp: psnr 23.83 dB at 0.552 bpp raw, 0.500 bpp entropy-coded, 0.084 bpp with an (up, left) context on level 0 [level 0 alone: 0.032] | 7.6s
ntc_decode out_reg_ntcb/model.ntcb -o n_reg --compare out_reg_ntcb/recon_q_final.png --verify:
level 0  : 64x64x4 bilinear, ntcb q8 (post-hoc grid) 8-bit, range [-2.022, 1.731] [-1.736, 1.772] [-1.761, 2.083] [-1.661, 1.888]
file     : out_reg_ntcb/model.ntcb 18199 bytes = 0.555 bpp (simulator raw 0.552 bpp: 131072 level bits + 256 side-info (stored in the header) + 13488 mlp bits; content 144560 bits matches the simulator exactly; container header 872 + section headers 128 + sync 24 + padding 8 bits; bpp over the 512x512 decode size, as every bpp in this log)
compare t0: out_reg_ntcb/recon_q_final.png (512x512): PSNR 88.28 dB, max |diff| 1, |diff| histogram: 0: 786356 (99.9903%), 1: 76 (0.0097%), 2: 0 (0.0000%), >2: 0 (0.0000%)
verify   : pre-nonlinearity: max 0 ulp (0 of 786432 values differ); output fp32: max 42 ulp, max |diff| 7.75e-07; RGB8 mismatches: 77 of 786432 (0.00979%)
model.bin: the same compare and verify lines; cmp n_reg.png b_reg.png -> identical
reload (--crop 512 --mlp-pairs 32 --load out_reg_ntcb/model.ntcb --iters 0): ntcb : level 0: 8-bit grid restored ...; iter 0 mse 269.392 psnr 23.83 dB; done: ... 23.83 dB at 0.552 bpp raw, 0.500 bpp entropy-coded, 0.084 bpp ... [level 0 alone: 0.032]  rc=0
```
(18194 -> 18199 bytes: one channel-count byte and 24 marker bits with 8 bits of padding.)

### The 27-row matrix, re-run with the new binaries

The driver is `matrix_all.sh` in the session scratchpad (it calls the `matrix.sh` of section 8 per row with the section 8 options;
the three fresh rows retrained on the GPU first, 400 iterations each, rc 0). Result: 27 of 27 rows pass every check: every
`file:` line `[OK]` with `self-check OK`, no MISMATCH, every conversion / decode / reload rc 0, every ntcb PNG byte-identical to
the model.bin PNG (37 PNGs), `--compare` max |diff| 1 on every texture, `--verify` `max 0 ulp` everywhere, every reload's
quantized done line equal to the converting invocation's. "was" = the byte count of section 8; the growth is the channel-count
byte(s) (T) plus the markers and their padding: +7 for a two-level single-texture file (1 + 6), +9 for three levels or two
textures, +10 for four textures (208 bytes over the 27 files).

| row | bytes (was) | growth | file bpp | sim raw bpp | verdict | sync bits | decoder PNG identical | compare max diff / PSNR | verify | reload done-line equal | quantized PSNR conv / reload |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dct50 | 565301 (565294) | +7 | 2.146 | 2.145 | OK + self-check OK | 42 | yes (1 PNG) | t0: 1 / 90.94 dB | max 0 ulp (0 of 6322560) | yes | 45.64 / 45.64 |
| dct-b6 | 155248 (155241) | +7 | 1.355 | 1.354 | OK + self-check OK | 42 | yes (1 PNG) | t0: 1 / 88.96 dB | max 0 ulp (0 of 2749248) | yes | 35.78 / 35.78 |
| dct-b4 | 126941 (126933) | +8 | 3.874 | 3.870 | OK + self-check OK | 42 | yes (1 PNG) | t0: 1 / 89.68 dB | max 0 ulp (0 of 786432) | yes | 26.08 / 26.08 |
| dct-eob-chk | 30063 (30055) | +8 | 0.917 | 0.914 | OK + self-check OK | 42 | yes (1 PNG) | t0: 1 / 88.11 dB | max 0 ulp (0 of 786432) | yes | 11.55 / 11.55 |
| dct-eob-cpu | 177284 (177277) | +7 | 5.410 | 5.407 | OK + self-check OK | 42 | yes (1 PNG) | t0: 1 / 90.19 dB | max 0 ulp (0 of 786432) | yes | 29.95 / 29.95 |
| qat4 | 1187814 (1187807) | +7 | 4.509 | 4.508 | OK + self-check OK | 36 | yes (1 PNG) | t0: 1 / 90.67 dB | max 0 ulp (0 of 6322560) | yes | 48.94 / 48.94 |
| qat1 | 202690 (202683) | +7 | 1.390 | 1.389 | OK + self-check OK | 36 | yes (1 PNG) | t0: 1 / 89.21 dB | max 0 ulp (0 of 3499200) | yes | 28.55 / 28.55 |
| qat2 | 348498 (348491) | +7 | 2.390 | 2.389 | OK + self-check OK | 36 | yes (1 PNG) | t0: 1 / 89.37 dB | max 0 ulp (0 of 3499200) | yes | 32.30 / 32.30 |
| qat3 | 494314 (494307) | +7 | 3.390 | 3.389 | OK + self-check OK | 36 | yes (1 PNG) | t0: 1 / 89.22 dB | max 0 ulp (0 of 3499200) | yes | 35.25 / 35.25 |
| qat-2ch | 1168805 (1168798) | +7 | 8.016 | 8.015 | OK + self-check OK | 36 | yes (1 PNG) | t0: 1 / 89.45 dB | max 0 ulp (0 of 3499200) | yes | 43.15 / 43.15 |
| qes6 | 1154884 (1154877) | +7 | 4.384 | 4.383 | OK + self-check OK | 36 | yes (1 PNG) | t0: 1 / 91.04 dB | max 0 ulp (0 of 6322560) | yes | 48.20 / 48.20 |
| qes10 | 1220744 (1220737) | +7 | 4.634 | 4.633 | OK + self-check OK | 36 | yes (1 PNG) | t0: 1 / 90.88 dB | max 0 ulp (0 of 6322560) | yes | 48.99 / 48.99 |
| c1float-image3 | 2241518 (2241511) | +7 | 8.509 | 8.508 | OK + self-check OK | 36 | yes (1 PNG) | t0: 1 / 91.25 dB | max 0 ulp (0 of 6322560) | yes | 46.85 / 46.85 |
| c1float-model15 | 1223290 (1223283) | +7 | 8.390 | 8.389 | OK + self-check OK | 36 | yes (1 PNG) | t0: 1 / 89.64 dB | max 0 ulp (0 of 3499200) | yes | 39.61 / 39.61 |
| c1qes8 | 2241518 (2241511) | +7 | 8.509 | 8.508 | OK + self-check OK | 36 | yes (1 PNG) | t0: 1 / 91.16 dB | max 0 ulp (0 of 6322560) | yes | 47.36 / 47.36 |
| c1qes8-c6 | 2307504 (2307497) | +7 | 8.759 | 8.759 | OK + self-check OK | 36 | yes (1 PNG) | t0: 1 / 91.16 dB | max 0 ulp (0 of 6322560) | yes | 50.83 / 50.83 |
| mat | 201002 (200992) | +10 | 6.134 | 6.129 | OK + self-check OK | 36 | yes (4 PNG) | t0: 1 / 89.09 dB; t1: 1 / 87.54 dB; t2: 1 / 87.79 dB; t3: 1 / 95.95 dB | max 0 ulp (0 of 3145728) | yes | 29.38 / 29.38 |
| mat-bilin | 86454 (86444) | +10 | 2.638 | 2.635 | OK + self-check OK | 36 | yes (4 PNG) | t0: 1 / 90.10 dB; t1: 1 / 87.69 dB; t2: 1 / 88.34 dB; t3: 1 / 91.90 dB | max 0 ulp (0 of 3145728) | yes | 25.45 / 25.45 |
| mat-b6 | 298800 (298790) | +10 | 8.978 | 8.970 | OK + self-check OK | 36 | yes (4 PNG) | t0: 1 / 88.70 dB; t1: 1 / 87.74 dB; t2: 1 / 89.09 dB; t3: 1 / 93.66 dB | max 0 ulp (0 of 3195072) | yes | 32.24 / 32.24 |
| 3lv | 693585 (693576) | +9 | 6.021 | 6.019 | OK + self-check OK | 48 | yes (1 PNG) | t0: 1 / 88.86 dB | max 0 ulp (0 of 2764800) | yes | 38.99 / 38.99 |
| old-v9 | 134708 (134701) | +7 | 4.111 | 4.107 | OK + self-check OK | 36 | yes (1 PNG) | t0: 1 / 89.68 dB | max 0 ulp (0 of 786432) | yes | 30.76 / 30.76 |
| old-v11 | 739566 (739559) | +7 | 2.508 | 2.507 | OK + self-check OK | 36 | yes (1 PNG) | t0: 1 / 90.04 dB | max 0 ulp (0 of 7077888) | yes | 40.29 / 40.29 |
| big-b32 | 421342 (421335) | +7 | 3.048 | 3.047 | OK + self-check OK | 36 | yes (1 PNG) | t0: 1 / 89.05 dB | max 0 ulp (0 of 3317760) | yes | 36.79 / 36.79 |
| big-b12 | 233181 (233174) | +7 | 2.188 | 2.186 | OK + self-check OK | 36 | yes (1 PNG) | t0: 1 / 89.98 dB | max 0 ulp (0 of 2558304) | yes | 33.49 / 33.49 |
| 3lv-dct | 136006 (135997) | +9 | 4.151 | 4.146 | OK + self-check OK | 54 | yes (1 PNG) | t0: 1 / 90.10 dB | max 0 ulp (0 of 786432) | yes | 28.78 / 28.78 |
| 3lv-qat | 86402 (86393) | +9 | 2.637 | 2.632 | OK + self-check OK | 48 | yes (1 PNG) | t0: 1 / 88.89 dB | max 0 ulp (0 of 786432) | yes | 28.95 / 28.95 |
| t2-dct | 148914 (148905) | +9 | 4.544 | 4.541 | OK + self-check OK | 42 | yes (2 PNG) | t0: 1 / 90.28 dB; t1: 1 / 87.13 dB | max 0 ulp (0 of 1572864) | yes | 26.62 / 26.62 |

The current form of the two file lines (dct50; the conversion and the decoder):
```
file: out_ntcb_dct50/model.ntcb 565301 bytes = 2.146 bpp (simulator raw 2.145 bpp); content 4521126 bits = simulator 4521478 - 352 side-info bits (stored in the header) [OK]; container header 1040 + section headers 192 + sync 42 + padding 8 bits; bpp over the 1480x1424 decode size, as every bpp in this log; self-check OK
file     : out_ntcb_dct50/model.ntcb 565301 bytes = 2.146 bpp (simulator raw 2.145 bpp: 4504230 level bits + 352 side-info (stored in the header) + 16896 mlp bits; content 4521126 bits matches the simulator exactly; container header 1040 + section headers 192 + sync 42 + padding 8 bits; bpp over the 1480x1424 decode size, as every bpp in this log)
```
The compare figures that moved against section 8 are the B2 effect listed above (rows with a mode-0 level; the PNG they are
compared with is now written from the strict values); every other row prints the section 8 figures.

### The corruption script (`ntcb_corrupt.py`, updated for the new layout)

Run on the dct50 container (dct level 0, qes level 1; 33 cases) and on the mat container (qat 2,2 level 0, mode-0 level 1; 30
cases), `build\Release\ntc_decode.exe` and `build\Release\ntc.exe --load` with the row options: **`ALL CASES EXIT 1: True` on
both**, every case with a message from both readers (`ntc` prefixes it with `corrupt or truncated ntcb file:`; the 20-byte
truncation still prints the existing `cannot read`, section 2 item 7). The cases beyond section 3 (f), with ntc_decode's message:
```
one header byte flipped: img_w bit 8 (hash stale)            file hash mismatch (corrupt file)
img_w = 16385 (re-hashed)                                    bad image size fields (decode size at most 16384 x 16384, source size within it)
level 0 W = 2^20, img_w = 16384 (re-hashed)                  mat: the header's sections need at least 268505222 bytes but the file has 200838 after the header (truncated or corrupt); dct50: a dct level must be a full-resolution nearest level ...
texture 0 channel count 4 (re-hashed)                        texture 0 has 4 channels; this build decodes 3 per texture
reserved header byte 1 (re-hashed)                           header field out of range (textures 1..4, 1..3 levels, activation 0, clamp 0/1, 1..8 hidden layers, reserved byte 0)
sync: level 0 start marker flipped (re-hashed)               sync marker mismatch before level 0 body
sync: level 0 channel 0 marker flipped (dct50)               sync marker mismatch before level 0 channel 0
sync: level 1 end marker flipped (re-hashed)                 sync marker mismatch after level 1 body
sync: mlp start marker flipped (re-hashed)                   sync marker mismatch before the mlp body
shift: one bit inserted at bit 6 of the level 0 body (dct50) sync marker mismatch before level 0 channel 0
shift: one bit inserted after the level 0 scale codes (dct50) level 0: truncated token stream   (the token parser runs off the end before the end marker is reached)
shift: one bit inserted at bit 6 of the level 1 body         sync marker mismatch after level 1 body
shift: one bit deleted at bit 6 of the level 1 body          sync marker mismatch after level 1 body
shift: one bit deleted at bit 6 of the mlp body              sync marker mismatch after the mlp body
```
(The section 3 (f) cases print their section 3 (f) messages, with "file hash mismatch" in place of "payload hash mismatch"; the
header-field edits are now re-hashed, since the hash covers the header.) The full outputs are `corrupt_dct50.txt` and
`corrupt_mat.txt` in the scratchpad.

### What the reviews got wrong, or partly

- B7 described the CUDA envelope check for a mode-0 level 0 next to `--qes 0,8` as skipped. The expression did evaluate to 0
  for that level, so the fix stands as specified, but the c1float rows had already passed the CUDA reload byte-for-byte in
  section 8 (the device decodes the copy `upload_model` builds from the shadow, which held the grid values): no observable
  effect before or after.
- Review 1's B3 named `--qat` / `--dct-q` warm starts; a third case, `--qes` on the level itself, was already guarded (the
  skip was silent); all three now print the note.
- Nothing else in either review was found to be wrong; B2 was confirmed by the writer's own refusal on the first mode-0 row.

Output directories of this section: the `out_ntcb_<row>` / `out_ntcb_<row>_reload` directories of section 8 (rewritten),
`out_ntcb_mat_resave`, `out_ntcb_mat_resave_reload`, `out_ntcb_b3_qat`, `out_ntcb_b3_qes`, `out_ntcb_b3_dct`, `out_ntcb_b7`,
`out_reg`, `out_reg_ntcb`, `out_reg_ntcb_reload`, `out_chk_g`.

## 10. Per-section sync marker values (September 7, 2026)

Richard asked that the marker values differ per section as well as per kind. Every written marker is now `ntcb_sync_value(base, level, channel) = (base + 13 * level + 7 * channel) & 63` (main.cpp and ntc_decode.cpp; the MLP markers use level = nlevels), so the same kind of marker carries a different value in every level and channel and a reader that reaches the right kind of marker in the wrong section refuses too. No change to the bit count (still 6 bits per marker) or to the reconciliation.

Re-verification on the rebuilt binaries (both builds 0 warnings): regression pin `psnr 23.83 dB`, `out_reg/model.bin` sha256 `cb884707...` unchanged; the five README `--cuda-check` lines and the `--dct-q 50 / 90 / 20` line: `cuda-check: all passed` (8 of 8). The 27-row matrix of section 8 re-run (`matrix_rerun2.txt`): 27 `[OK]` + `self-check OK`, 37 of 37 PNGs `cmp` identical, every `--verify` `max 0 ulp`, every `--compare` max |diff| 1, every reload done line equal. Corruption script on the dct50 and mat containers (`corrupt_dct50_v2.txt`, `corrupt_mat_v2.txt`): `ALL CASES EXIT 1: True` on both, every marker flip and every one-bit shift refused by name.

| row | bytes | file bpp | simulator raw | verdict |
|---|---|---|---|---|
| dct50 | 565301 | 2.146 | 2.145 | OK |
| dct-b6 | 155248 | 1.355 | 1.354 | OK |
| dct-b4 | 126941 | 3.874 | 3.870 | OK |
| dct-eob-chk | 30063 | 0.917 | 0.914 | OK |
| dct-eob-cpu | 177284 | 5.410 | 5.407 | OK |
| qat4 | 1187814 | 4.509 | 4.508 | OK |
| qat1 | 202690 | 1.390 | 1.389 | OK |
| qat2 | 348498 | 2.390 | 2.389 | OK |
| qat3 | 494314 | 3.390 | 3.389 | OK |
| qat-2ch | 1168805 | 8.016 | 8.015 | OK |
| qes6 | 1154884 | 4.384 | 4.383 | OK |
| qes10 | 1220744 | 4.634 | 4.633 | OK |
| c1float-image3 | 2241518 | 8.509 | 8.508 | OK |
| c1float-model15 | 1223290 | 8.390 | 8.389 | OK |
| c1qes8 | 2241518 | 8.509 | 8.508 | OK |
| c1qes8-c6 | 2307504 | 8.759 | 8.759 | OK |
| mat | 201002 | 6.134 | 6.129 | OK |
| mat-bilin | 86454 | 2.638 | 2.635 | OK |
| mat-b6 | 298800 | 8.978 | 8.970 | OK |
| 3lv | 693585 | 6.021 | 6.019 | OK |
| old-v9 | 134708 | 4.111 | 4.107 | OK |
| old-v11 | 739566 | 2.508 | 2.507 | OK |
| big-b32 | 421342 | 3.048 | 3.047 | OK |
| big-b12 | 233181 | 2.188 | 2.186 | OK |
| 3lv-dct | 136006 | 4.151 | 4.146 | OK |
| 3lv-qat | 86402 | 2.637 | 2.632 | OK |
| t2-dct | 148914 | 4.544 | 4.541 | OK |
