# QES_NOTES.md — quantization-aware ES (`--qes`) implementation notes

> Historical note (v0.8, September 6, 2026): written for the original implementation. Since v0.8 the `--qes-refit` / `--qes-freeze` options and the legacy..v8 loader branches are gone: the ranges are fitted once at `--qes-start` and always frozen, the loader floor is v9, and the mismatch message no longer prints `--act`. The verification numbers below still hold.

September 6, 2026. Implements QAT_LEVEL1_PLAN.md with its section 8 amendment (flag `--qes`,
every continuous level, start-at-half schedule). All three hand-written copies of the math
(main.cpp, cuda/ntc_cuda.cu, ntc_decode.cpp) were changed together. Every number below is
copied from the program output of the commands shown.

## 1. What was built

A continuous (ES-trained) latent level can be held on a per-channel min/max grid of 2^B values
during training:

- **State.** `Decoder` keeps the fp32 shadow `lat.z` (Adam state unchanged) and a snapped copy
  `zq` of the whole latent (non-`--qes` levels copied as they are). `Decoder::zdec()` returns
  `zq` once the ranges are fitted (`qes_live`) and `lat.z` before that, so with the flag off the
  decode pointer is the shadow itself and nothing changes (regression pin unaffected).
- **Every decode reads the snapped copy:** the MLP ES and FD minibatches (`MlpTrainer::step`,
  `step_fd`), both halves of every latent ES pair (`LatentTrainer::step` builds `zp/zm` from
  `zdec()`, applies Adam to `lat.z`, then `qes_refresh()`), the level-0 exhaustive search
  (`qat_search` runs on `zdec_mut()` and copies its level-0 slice back to the shadow when the two
  buffers differ), the statistics decodes and `recon_q_final.png`, `eval_filter`, the saved file,
  and the CUDA equivalents (`d_zdec`).
- **Quantizer** (strict floating point, `STRICT_FP_BEGIN/END` = the pragmas ntc_decode.cpp uses
  plus `fp_contract(off)` on MSVC): `qes_index(v, lo, range, levels) =
  clamp(nearbyintf(((v - lo) / range) * (float)levels))`, `range = max(hi - lo, 1e-6)`, grid
  `lo + k / (float)levels * range` computed on the host once per fit (`qes_fit`, `qes_rebuild`) and
  looked up by `qes_snap_level` on the host and `k_snap` on the device (never recomputed there).
  Round-half-even on both sides (`nearbyintf` / `__float2int_rn`).
- **Schedule.** `qes_start_it = ceil(--qes-start * iters)`; at the top of that iteration (or before
  anything decodes when it is 0) the per-channel min/max ranges are fitted from the shadow and
  snapping begins. With `--qes-refit N > 0` and `--qes-freeze F2 > --qes-start` the ranges are
  refitted every N iterations until `ceil(F2 * iters)`, then frozen; otherwise frozen at the start.
  Once frozen, the shadow is clamped to `[lo, hi]` inside every refresh (after each Adam step; on
  the device inside `k_snap`).
- **Reporting.** `bitrate_stats` charges a live `--qes` level its bits per value plus 64 header
  bits per channel, takes its entropy from the grid indices of the shadow, and returns the snapped
  values unchanged (no post-hoc re-quantization). Before the start iteration such a level is still
  continuous and gets the post-hoc quantization at its `--qes` depth (so the quoted raw bpp is the
  run's final format throughout). Other levels keep the `--qbits` behaviour. The level-0
  context-entropy figure keeps working on a `--qes` level 0 (indices from the qes grid). The
  banner has a `qes      :` line; the progress line labels the quantized PSNR `q<spec>`
  (e.g. `q0,8`); the final line says `--qes 0,8 latent + fp16 mlp`.
- **Model file v12** (magic `0x4E54433C`): after the v11 ints and the extra-level dims, per level
  (level 0 included): `int qes_bits` (0 = continuous), then `C` floats `lo`, then `C` floats `hi`
  (zeros when off or not yet started); then the hidden widths as before. The latent payload is
  `zdec()`: on-grid values for `--qes` levels. The loader accepts v2..v12. Same `--qes` spec:
  ranges restored, frozen, no refit (`--load --iters 0` reproduces the run). Different or no
  `--qes`: warm start with a note. The mismatch message gains ` --qes B0,B1` / ` (no --qes)`.
- **CUDA.** `ModelDesc::qes_bits[MAX_LEVELS]`; `QesDesc` (bits, live, frozen, lo/hi/range and
  grid offsets per level and channel, up to `MAX_QES_CH = 64` channels per level) in
  `__constant__ c_qes`; the grid table in global memory `d_qgrid`. `Trainer::set_qes()`,
  `snap_latent()`, `download_zq()`. Persistent `d_zq` and the pointer `d_zdec` (= `d_z` when no
  level has `--qes`, so no copies and no behaviour change). `k_snap` runs after `k_adam` in
  `lat_step`, in `upload_model` and in `set_qes`. `k_decode`, `k_batch_features`, `k_lat_pair` and
  `k_qat_search` read `d_zdec`; `qat_search` copies the level-0 slice `d_zdec -> d_z` when the
  buffers differ. The old scratch `d_zq` of `decode_full` is now `d_ztmp`.
- **`--cuda-check`.** Checks 1-5 read `zdec()` where the CPU trainer does; new check 6 downloads
  the device decode buffer and compares it with the host snap of the downloaded shadow (exact, 0
  differing values), then again after one `lat_step` with the run's `--lat-lr`.
- **ntc_decode.cpp.** Accepts v6..v12; reads the per-level bits and ranges; `q8_quantize` skips
  every level the file marks as quantized (`--qat` level 0, `--qes` levels) and keeps the post-hoc
  8-bit step for the others; the summary prints `qes B-bit, range [lo, hi] ...` per channel; a
  note says when `--q8` and `--fp32-latent` decode the same values.

## 2. Flags

| flag | meaning |
|---|---|
| `--qes B` | every level without `--qat` on a 2^B-value per-channel min/max grid (B = 2..12) |
| `--qes B0,B1[,B2]` | per-level depths, 0 = continuous; must name every level; nonzero on a `--qat` level is an error; a level that does not exist is an error |
| `--qes-start F` | fit the ranges and start snapping at `ceil(F * iters)` (default 0.5; 0 = from the first iteration) |
| `--qes-refit N` | refit every N iterations between start and freeze (default 0 = never) |
| `--qes-freeze F` | freeze at `ceil(F * iters)` (default = `--qes-start`); shadow clamped from then on |

Refusals verified: `--qat 2 --qes 2,8`, `--qat 2 --qes 0,8,4` (3 levels listed, model has 2),
`--qes 1`, `--qes 13`, `--qes-start 1.5`, `--qes-freeze 0.5` with `--qes-start 0.6`. Notes:
`--qes 0,0` (names no level), schedule flags without `--qes`.

## 3. Verification (commands and output)

Builds: `cmake --build build --config Release` (0 warnings) and
`cmake --build build_cuda --config Release` (only the pre-existing `LNK4098` defaultlib
warning from `cudart_static`).

### Regression pin (flag off)
```
build\Release\ntc.exe --crop 512 --mlp-pairs 32 --iters 200 --print-every 200 --save-every 200 --out out_reg
iter    200  mse 0.004140  psnr  23.83 dB  best  23.83 | q8 psnr  23.83 @ 0.552 bpp (ent 0.500, ctx 0.084) | ...
```

### `--cuda-check`, new layouts (chief1.png 512x512, `--iters 5 --print-every 5 --out out_chk`)
(a) `--block 8 --latent 0 0 1 --latent2 0 0 4 --filter nearest,bilinear --pos lv1local --qat 2 --qes 0,8 --mlp 17,17 --leak 0.0009765625 --qes-start 0`
```
  decode          : max |gpu - cpu| = 1.192e-07  PASS
  mlp ES dl       : max |gpu - cpu| / rms(dl) = 3.991e-07 over 8 pairs  PASS
  mlp FD dl       : max |gpu - cpu| / rms(dl) = 1.096e-04 over 24 weights  PASS
  latent ES grad  : max |gpu - cpu| / rms(grad) = 3.821e-05 over 278528 values  PASS
  qat search      : 1 of 262144 values differ in 1 texels: gpu worse 0, better 0, ties 1; loss after search cpu 6.581854e-02 gpu 6.581854e-02  PASS
  qes snap        : device snapped copy vs host snap of the downloaded shadow: 0 of 278528 values differ, 0 after a latent step with lr 0.02  PASS
cuda-check: all passed
```
(b) `--latent 128 128 4 --latent2 32 32 4 --filter bilinear,bilinear --pos uv --qes 6,8 --qes-start 0 --mlp 17,17`
```
  decode          : max |gpu - cpu| = 1.192e-07  PASS
  mlp ES dl       : max |gpu - cpu| / rms(dl) = 2.424e-07 over 8 pairs  PASS
  mlp FD dl       : max |gpu - cpu| / rms(dl) = 1.721e-04 over 24 weights  PASS
  latent ES grad  : max |gpu - cpu| / rms(grad) = 3.931e-05 over 69632 values  PASS
  qes snap        : device snapped copy vs host snap of the downloaded shadow: 0 of 69632 values differ, 0 after a latent step with lr 0.02  PASS
cuda-check: all passed
```
(c) same as (b) with `--qes 8`
```
  decode          : max |gpu - cpu| = 1.192e-07  PASS
  mlp ES dl       : max |gpu - cpu| / rms(dl) = 4.793e-07 over 8 pairs  PASS
  mlp FD dl       : max |gpu - cpu| / rms(dl) = 1.765e-04 over 24 weights  PASS
  latent ES grad  : max |gpu - cpu| / rms(grad) = 3.144e-05 over 69632 values  PASS
  qes snap        : device snapped copy vs host snap of the downloaded shadow: 0 of 69632 values differ, 0 after a latent step with lr 0.02  PASS
cuda-check: all passed
```

### `--cuda-check`, the README's existing three lines (unchanged)
```
ntc model.png --cuda --cuda-check --latent 512 512 2 --latent2 128 128 4 --filter nearest,nearest --pos lv1local --qat 3,1 --mlp 36,36 --iters 1 --out out_chk
  decode 1.788e-07 PASS | mlp ES 5.599e-07 PASS | mlp FD 1.096e-04 PASS | latent ES grad 2.126e-05 over 589824 PASS
  qat search: 3 of 524288 values differ in 2 texels: gpu worse 0, better 1, ties 1; loss cpu 5.724406e-02 gpu 5.724406e-02 PASS
  qes snap: 0 of 589824 values differ, 0 after a latent step (no live --qes level) PASS
ntc --cuda --cuda-check --iters 1 --out out_chk
  decode 1.788e-07 PASS | mlp ES 3.334e-07 PASS | mlp FD 5.628e-05 PASS | latent ES grad 7.586e-06 over 16384 PASS | qes snap 0 of 16384 PASS
ntc m1.png m2.png m3.png m4.png --cuda --cuda-check --latent 512 512 3 --latent2 128 128 4 --filter nearest,nearest --pos lv1local --qat 2 --mlp 36,36 --iters 1 --out out_chk
  decode 1.192e-07 PASS | mlp ES 2.439e-07 PASS | mlp FD 5.807e-05 PASS | latent ES grad 5.538e-05 over 851968 PASS
  qat search: 3 of 786432 values differ in 2 texels: gpu worse 2, better 0, ties 0; loss cpu 7.811870e-02 gpu 7.811872e-02 PASS
  qes snap: 0 of 851968 values differ, 0 after a latent step PASS
```
All three: `cuda-check: all passed`.

### CPU vs GPU, 500 iterations, default `--qes-start 0.5`
Common options: `chief1.png --rng hash --iters 500 --lr-anneal 0.5 0.05 --mlp-fd 0.5 --mlp-pairs 32 --print-every 100`.

Layout (a) `--block 8 --latent 0 0 1 --latent2 0 0 4 --filter nearest,bilinear --pos lv1local --qat 2 --qes 0,8 --mlp 17,17 --leak 0.0009765625`:
```
banner   : bitrate  : fp32 34.061 bpp (latent 34.000 + mlp 0.061); 8-bit latent + fp16 mlp 2.531 bpp raw (level 0 at its --qat depth, --qes levels at their --qes depth)
           qes      : level 1 at 8 bits: quantization-aware ES (...); ranges fitted at iteration 250, then frozen (shadow clamped)
CPU  iter 250  qes: ranges fitted ... level 1 8-bit [-1.774, 1.122] [-1.041, 1.763] [-1.169, 1.291] [-1.786, 1.748]
CPU  iter 500  psnr  31.30 dB | q0,8 psnr  31.30 @ 2.531 bpp (ent 2.438, ctx 1.410)   (out_qes_a_cpu.log)
GPU  iter 250  qes: ranges fitted ... level 1 8-bit [-1.774, 1.126] [-1.043, 1.763] [-1.159, 1.292] [-1.786, 1.747]
GPU  iter 500  psnr  31.29 dB | q0,8 psnr  31.29 @ 2.531 bpp (ent 2.438, ctx 1.413)   (out_qes_a_gpu.log)
```
Hand computation of the raw bpp for (a): selector 512x512x1 at 2 bits = 2.000; latent2 64x64x4 at
8 bits = 131072 / 262144 = 0.500; headers 4 x 64 = 256 bits = 0.00098; MLP 496 params x 16 bits =
7936 bits = 0.03027; total 2.531 bpp. Matches the banner and the progress lines.

Layout (b) `--latent 128 128 4 --latent2 32 32 4 --filter bilinear,bilinear --pos uv --qes 6,8 --mlp 17,17`:
```
banner   : bitrate  : fp32 8.567 bpp (latent 8.500 + mlp 0.067); 8-bit latent + fp16 mlp 1.660 bpp raw (--qes levels at their --qes depth)
           qes      : level 0 at 6 bits, level 1 at 8 bits: ... ranges fitted at iteration 250, then frozen (shadow clamped)
CPU  iter 500  psnr  26.95 dB | q6,8 psnr  26.95 @ 1.660 bpp (ent 1.435, ctx 0.919)   (out_qes_b_cpu.log)
GPU  iter 500  psnr  26.94 dB | q6,8 psnr  26.94 @ 1.660 bpp (ent 1.435, ctx 0.919)   (out_qes_b_gpu.log)
```
Hand computation for (b): 128x128x4 at 6 bits = 393216 bits = 1.500; 32x32x4 at 8 bits = 32768
bits = 0.125; headers 8 x 64 = 512 bits = 0.00195; MLP 547 x 16 = 8752 bits = 0.03339; total
1.660 bpp. Before the start iteration the progress line already charges 1.660 bpp and reports
the post-hoc quantization at the `--qes` depths (GPU iteration 100: fp32 22.95 dB, `q6,8` 22.94 dB
at 1.660 bpp); all four logs were produced with the final binaries.

### ntc_decode on the v12 files
```
build\Release\ntc_decode.exe out_qes_a_gpu\model.bin --compare out_qes_a_gpu\recon_q_final.png --verify -o dec_q8
note: every level was quantized in training; --q8 and --fp32-latent decode the same values
model    : model.bin (v12)
level 0  : 512x512x1 nearest, qat 2, fp32 planar
level 1  : 64x64x4 bilinear, qes 8-bit, range [-1.774, 1.126] [-1.043, 1.763] [-1.159, 1.292] [-1.786, 1.747]
compare t0: recon_q_final.png (512x512): PSNR 90.85 dB, max |diff| 1, |diff| histogram: 0: 786390 (99.9947%), 1: 42 (0.0053%), 2: 0, >2: 0
verify   : pre-nonlinearity: max 0 ulp (0 of 786432 values differ); output fp32: max 42 ulp, max |diff| 7.75e-07; RGB8 mismatches: 41 of 786432 (0.00521%)
--fp32-latent -> dec_fp32.png: identical bytes to dec_q8.png (cmp)

build\Release\ntc_decode.exe out_qes_b_cpu\model.bin --compare out_qes_b_cpu\recon_q_final.png --verify -o dec_q8
level 0  : 128x128x4 bilinear, qes 6-bit, range [-1.493, 2.038] [-2.565, 2.333] [-1.555, 1.737] [-1.904, 1.922]
level 1  : 32x32x4 bilinear, qes 8-bit, range [-1.304, 1.784] [-1.368, 0.8429] [-1.231, 1.159] [-1.021, 1.163]
compare t0: PSNR 89.61 dB, max |diff| 1, 0: 786376 (99.9929%), 1: 56 (0.0071%)
verify   : pre-nonlinearity: max 0 ulp (0 of 786432 values differ); RGB8 mismatches: 56 of 786432 (0.00712%)
--fp32-latent output identical bytes to --q8 output (cmp)

v11 file: build\Release\ntc_decode.exe out_model12_b8_c1q2_c4bilin_mlp27_cuda8k_fd50\model.bin --compare ...\recon_q_final.png
model    : model.bin (v11)
level 1  : 256x144x4 bilinear, q8 per channel
compare t0: (2048x1152): PSNR 89.53 dB, max |diff| 1, 0: 7077375 (99.9928%), 1: 513 (0.0072%)
```

### `--load --iters 0`
```
CPU: ntc chief1.png <layout a options> --qes 0,8 --load out_qes_a_gpu/model.bin --iters 0
qes      : ranges restored from the file (frozen, no refit)
iter      0  mse 0.000742  psnr  31.29 dB  (initial)
done: final psnr 31.29 dB ... --qes 0,8 latent + fp16 mlp: psnr 31.29 dB at 2.531 bpp raw, 2.438 bpp entropy-coded, 1.413 bpp ...
GPU (--cuda): iter 0 psnr 31.29 dB; done: final psnr 31.29 dB (same line)
v12 loaded with --qes 0,6: note: the model was saved with --qes 0,8; this run uses --qes 0,6: warm start ... -> 31.26 dB at 2.406 bpp
v12 loaded with no --qes:  note: the model was saved with --qes 0,8; this run uses no --qes: warm start ... -> 31.29 dB
v12 loaded with --mlp 18,18: "... --leak 0.000977 --qat 2 --qes 0,8 --mlp 17,17 --act leaky --pos lv1local and 1 texture(s); pass the same options"
v11 in the trainer: ntc model12.png <its options> --load out_model12_b8_c1q2_c4bilin_mlp27_cuda8k_fd50/model.bin --iters 0
  iter 0 psnr 40.54 dB; done: final psnr 40.54 dB ... 8-bit latent + fp16 mlp: psnr 40.35 dB at 2.507 bpp raw (the log's final line: 40.54 / 40.35 / 2.507)
```

### Bit depth flowing through (layout (a), CPU, 500 iterations, same options as above)
```
--qes 0,6  : bitrate 2.406 bpp raw; iter 500 psnr 31.32 dB | q0,6 psnr 31.32 @ 2.406 bpp (ent 2.314, ctx 1.293); file: level 1 qes 6-bit;  ntc_decode --compare: PSNR 90.85 dB, max |diff| 1
--qes 0,10 : bitrate 2.656 bpp raw; iter 500 psnr 31.30 dB | q0,10 psnr 31.30 @ 2.656 bpp (ent 2.554, ctx 1.530); file: level 1 qes 10-bit; ntc_decode --compare: PSNR 89.09 dB, max |diff| 1
```
(2 + 4 x 6 / 64 + 0.031 = 2.406; 2 + 4 x 10 / 64 + 0.031 = 2.656.)

## 4. Deviations from the plan, with reasons

1. **Position of the v12 fields.** The per-level `qes_bits, lo[C], hi[C]` block is written after the
   extra-level `(W, H, C)` triples and before the hidden widths (plan section 3), i.e. after every
   v11 field in file order, because the loader needs each level's channel count before it can
   read that level's ranges.
2. **Bitrate before the start iteration.** A `--qes` level that has not started yet is charged the
   post-hoc quantization at its `--qes` depth rather than at `--qbits`, so the raw bpp printed on
   every progress line is the run's final format (the plan did not say what to print before the
   start; the amendment asks for the level to be charged its `--qes` bits). Consequence: a
   `--save-every` checkpoint written before the start iteration stores the level continuous
   (`qes_bits = 0`), so `ntc_decode` re-quantizes it at 8 bits while the trainer's progress line
   charged it at the `--qes` depth; only mid-run checkpoints are affected, the final file is
   always live.
3. **Loaded ranges are frozen** regardless of `--qes-refit` / `--qes-freeze` (the plan only covers
   `--iters 0`; continuing training from a v12 file keeps the file's grid).
4. **No refit at the freeze iteration:** the last refitted ranges are the frozen ones.
5. **Check 6 also runs without `--qes`** (then it is a copy check: the device decode buffer aliases
   the shadow) rather than being skipped, so the README's existing lines print six PASS lines.
6. **No "off-grid" invariant print** for `--qes` levels on the progress line: the snapped copy is
   recomputed from the shadow on every refresh, so the count would be zero by construction; the
   meaningful device-versus-host comparison is check 6.
7. **`--qes B` with `--qat`:** the single value silently applies to every level except the `--qat`
   level (amendment: "every continuous level"); only an explicit nonzero entry for the `--qat`
   level is an error.
8. **Grid table in global memory, ranges in `__constant__`** (`QesDesc`, `MAX_QES_CH = 64`
   channels per level, refused above that with `--cuda`); the table is sized for 12 bits on every
   channel of every level (`MAX_QES_GRID = 3 x 64 x 4096` floats, 3 MB) and allocated only when
   some level has `--qes` bits.
9. **`STRICT_FP_BEGIN` on MSVC adds `fp_contract(off)`** to the pragmas ntc_decode.cpp uses so the
   host grid value `lo + k / levels * range` cannot be contracted (the index expression has no
   multiply-add pair either way).
10. The `iter 0 qes: ranges fitted ...` line for `--qes-start 0` prints before the banner (the fit
    must precede `--cuda-check` and the CUDA init).

## 5. Files and functions touched

- `main.cpp` (CRLF kept): DEPENDENCY notes; `Options` (`qes_ch`, `qes`, `qes_start`, `qes_refit`,
  `qes_freeze`); `usage()`; new QES block after `LatentSet` (`STRICT_FP_*`, `QesLevel`,
  `qes_index`, `qes_fit`, `qes_rebuild`, `qes_snap_level`, `qes_spec`); `Decoder` (`qes`, `zq`,
  `qes_on/live/frozen`, `zdec()`, `zdec_mut()`, `qes_refresh()`, `qes_fit_all()`);
  `MlpTrainer::step` / `step_fd`; `LatentTrainer::step`; `qat_search`; `save_model` (v12);
  `bitrate_stats` (new signature `(const Decoder&, int qbits, zq)`); `eval_filter`; `cuda_check`
  (checks 1-5 on `zdec()`, check 6); `main()` (parser, validation, level resolution, loader v12
  and warm-start notes, `qes_start_it/qes_freeze_it`, `qes_begin`, `qes_sync_device`, ModelDesc
  fill, banner, `decode_with`, `qlabel`, the schedule at the top of each iteration, the print and
  final blocks; the local `zq` renamed `zpost`).
- `cuda/ntc_cuda.h` (LF): `MAX_QES_CH`, `ModelDesc::qes_bits`, `QesDesc`, `Trainer::set_qes`,
  `snap_latent`, `download_zq`.
- `cuda/ntc_cuda.cu` (LF): DEPENDENCY note; `MAX_QES_GRID`, `c_qes`, `k_snap`; `Impl` (`d_ztmp`,
  `d_zq`, `d_zdec`, `d_qgrid`, `qes`); destructor; `init` (envelope check, `c_qes` upload,
  buffers, `d_zdec` aliasing); `snap_latent_impl`, `upload_model`, `set_qes`, `snap_latent`,
  `download_zq`; `blocks_for` moved up; `decode_full`, `draw_batch`, `lat_step`, `qat_search`
  read `d_zdec` (copy-back in `qat_search`).
- `ntc_decode.cpp` (CRLF kept): header comments; `Model` (`qes_bits`, `qes_lo`, `qes_hi`);
  `load_model` (v12); `level_quantized`, `q8_quantize`; `usage()` (`--q8`, `--fp32-latent`);
  `main` (note, summary line).
- `README.md` (CRLF kept): `--qes` paragraph under "A learned block format", four flag rows, the
  v12 note in the `--qat` row, check 6 and two new lines under "Regression testing".
- New: `QES_NOTES.md` (this file). Test outputs: `out_qes_a_cpu`, `out_qes_a_gpu`, `out_qes_b_cpu`,
  `out_qes_b_gpu`, `out_qes_a6_cpu`, `out_qes_a10_cpu` and their `.log` files, `out_chk`, `out_reg`,
  `out_smoke_a`, `out_load_*`, `out_val` (scratch).
