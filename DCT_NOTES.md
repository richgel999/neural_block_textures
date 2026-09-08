# DCT_NOTES.md — the experimental DCT-coded selector plane (`--dct-q`) implementation notes

September 6, 2026. Implements DCT_MVP_PLAN.md (sections 0-9 as amended by its sections 12 and 13,
which override the earlier text where they conflict) on top of tag `pre-dct-mvp`. All three
hand-written copies of the decode math (main.cpp, cuda/ntc_cuda.cu, ntc_decode.cpp) were changed
together. Every number below is copied from the program output of the commands shown; nothing is
estimated. Nothing is committed (the parent commits).

## 1. What was built

With `--dct-q Q` (Q = 1..100) on a `--latent 0 0 1 --filter nearest,...` layout, level 0 is an
8x8 DCT-coded selector plane. With the flag absent nothing changes (section 3, "Firewall").

- **Units and transform.** Selector `s` in [-1,1], plane `w = (s + 1) * 32` in [0,64] (the
  XUASTC / XUBC7 normalization), reconstruction `s' = clamp(w' / 32 - 1, -1, 1)`. Orthonormal
  2D DCT-II / DCT-III ported from basisu `dct2f` with alpha folded into the basis table:
  `DCT_BASIS_BITS[64]` are 64 float bit patterns (the double products `alpha(u) cos(pi (2x+1) u / 16)`
  rounded once), the same literals in main.cpp and ntc_decode.cpp and uploaded to the device;
  never recomputed with `cosf`. Fixed evaluation order (sequential float sums, inner index
  ascending), `w[x * 8 + y]` with `x` the row, `C[u * 8 + v]` with `u` the vertical frequency.
- **Quantizer** (XUASTC's, `basisu_transcoder.cpp` L27167-27252 and `_internal.h` L1920-1962):
  libjpeg quality scaling `S = (q < 50 ? 5000 / q : 200 - 2q) / 100`, Annex K luminance table
  `g_baseline_jpeg_y` sampled bilinearly at `(v * 8/N, u * 8/N)` (the identity at N = 8; `step[u][v]
  = K.1[u][v]`, not the transpose), `step = max(1, (int)(base * level_scale + 0.5))`, every AC step 1
  at q >= 100. DC (coefficient (0,0) = 8 x block mean, range [0,512]): uniform, no dead zone,
  round-half-even (`nearbyintf` / `rintf`), step `--dct-dc-step` (default 4). AC: dead zone alpha
  0.5 with the first-order exemption: (1,0) and (0,1) are `roundf(d / L)` (half away from zero on
  both sides); the others `s = |d|, tau = 0.5 L, s <= tau -> 0, else floor((s - tau) / L + 0.5)` with
  the sign, so the zero bin is `|d| < L`, `|q| = 1` covers [L, 2L) and dequantizes to the bin centre
  `tau + |q| L`. Every `(int)` cast is preceded by a float clamp that maps NaN to the lower bound.
- **Per-block scale (decision 4, option (b)).** `level_scale_k = S(q) * A_k`, `A_k = 1 / g_k`,
  `g_k = (14/64) 2^(0.6 k)`, k = 0..15 (`A_0 = 64/14`, the XUBC7 / XUASTC span floor over 64;
  `g_15 = 112` LSB per plane unit); `DCT_AK_BITS[16]` are float bit patterns shared with
  ntc_decode.cpp. The 16 x 64 integer step tables per channel are built once on the host
  (`dct_build_steps`, strict FP) and copied to the device; ntc_decode rebuilds them from
  (q, N, dc_step) with the same source text. The code of a block is the trainer-side probe
  `dct_probe_codes`: at 4 pixel positions `(bx*8 + {2,5}, by*8 + {2,5})` the MLP input is built once
  and the MLP evaluated with the selector at -1, -0.5, 0, 0.5, 1; segment gain
  `g_j = (255/16) sqrt(sum_o cw[o] (out_{j+1}[o] - out_j[o])^2 / wsum)`, block gain = RMS over the 16
  (position, segment) pairs, `k = clamp(lround(log2(max(g, 14/64) / (14/64)) / 0.6), 0, 15)` in
  double. 20 MLP evaluations per block. Codes are fitted once at `dct_start_it = ceil(--dct-start *
  iters)` and frozen; a v13 file restores them frozen.
- **Training through the `--qes` mechanism, spatial shadow.** `Decoder::zdec()` returns the snapped
  copy `zq` once `dct.live`; `qes_refresh()` clamps the shadow's level 0 to [-1,1] at its top whenever
  `dct.on && !dct.live` (before either branch, so it also runs when `--qes` is already live), and
  after the qes level loop (which copies level 0 as a plain copy) overwrites level 0 by
  `dct_snap_level0` (clamp in place, forward DCT, quantize, store the symbols, dequantize, inverse
  DCT, clamp; blocks independent, OpenMP). `LatentTrainer::step` is unchanged: level 0 is ES-active
  because `o.qat == 0`; the perturbation is added to the snapped plane and the update goes to the
  shadow. No soft rate penalty, no truncation RDO, no perceptual metric.
- **Rate = bit simulator** (`dct_analyze`): per channel, blocks in raster order, the DC symbol then
  the 63 ACs in the JPEG zigzag as (run, sign, |q| - 1) tokens with EOB (run symbol 64) when the
  rest is zero (none when the last nonzero is at zigzag 63). `raw`: `dc_raw_bits` (= ceil(log2(floor(512 /
  dc_step) + 1)), 8 at step 4) per block + 7 + 8 + 1 per nonzero + 7 per EOB + 4 per scale code +
  96 header bits per channel; `h0`: order-0 entropy of the run stream (EOB included), of the |q|
  stream, of the DC residual against the left block (first column: above; first block: 0), 1 bit
  per sign, order-0 entropy of the code plane; `ctx`: run and |q| conditioned on the zigzag
  position of the previous nonzero (0 for a block's first token), DC and signs as h0, the code
  plane under `context_entropy_bits` (up, left). Every token is charged its code length under its
  stream's model, so the per-class split (dc / run / mag / sign / eob / code / hdr) sums exactly to
  the total. `bitrate_stats` folds these into the raw / entropy / context totals (the branch is the
  first statement of the level body; no per-texel bits, no 64-bit min/max header for level 0).
- **Statistics and PNGs** (plan 12.B / 12.C / 13.8 / 13.9). Progress line: `| dct nz lnz eob0%
  clamp% bits/blk raw/h0/ctx chg sym N zq M` (symbols and level-0 values changed since the last
  print). Final block before `done:` (and at every print with `--dct-stats`): `dct cfg / nz /
  nzhist / lnz / run / dc / pnz / pnz_lo / codes / blkbits / bits raw|h0|ctx / plane` lines plus
  `dct reprobe` (the probe re-run at the end, not applied: would-be histogram and the fraction of
  codes that would move by >= 1 / >= 2). The switch line prints the PSNR before and after the first
  snap and forces a progress print. `dct_map_NNNNNN.png` (one colour per 8x8 block; `--dct-map nz |
  code | bits | lnz | dconly | all`), `latent_q_NNNNNN.png` (the snapped plane), `dct_resid_NNNNNN.png`
  ((shadow - snapped) * 8 + 0.5), `dct_map.png` and `dct_side_by_side.png` at the end.
- **Model file v13** (magic `0x4E54433D`), written only when `dct.live`: the v12 layout through the
  per-level `--qes` fields, then `int dct_N, dct_dc_step, dct_C`, `dct_C` ints `q[c]`, the hidden
  widths and spec as v12, then `int16 sym[BH*BW][dct_C][64]` (natural order), `uint8 code[BH*BW][dct_C]`,
  the floats of levels >= 1 only, the MLP. Everything is indexed `[block][channel]` with
  `MAX_DCT_CH = 4`; the MVP refuses `C0 != 1` (lifted in section 7: 1..4 channels, `--dct-q Q0,Q1,...`). Both loaders accept v9..v13. A v13 file needs the
  same `--dct-q` and `--dct-dc-step` (no fp32 shadow is stored); a v9..v12 file loaded with
  `--dct-q` is the warm start (continuous level 0 snapped at `--dct-start`), which with `--iters 0`
  is the fast-sweep mode of plan 12.E. Level 0 of a loaded v13 is `dct_recon_level0` of the symbols,
  the shadow's level 0 is set equal to it, no probe, no re-snap.
- **CUDA.** `ModelDesc::dct / dct_C / dct_N / dct_dc_step / dct_q[] / dct_basis`, `DctDesc`,
  `Trainer::set_dct` (uploads the description, the step tables into `__constant__ c_dct_step[4][16][64]`,
  the codes and symbols, then `k_dct_recon` rebuilds level 0 of `d_zq` from the symbols; the shadow
  `d_z` is never written by it), `download_dct`. `d_zq` exists when `I.qes || I.dct`; `snap_latent_impl`
  runs `k_snap` and then `k_dct_snap` (live) or `k_dct_clamp0` (not live: clamps level 0 of the shadow
  and the copy). Every product and sum in the kernels is `__fmul_rn / __fadd_rn / __fsub_rn / __fdiv_rn`
  so nvcc cannot contract a multiply-add pair; `rintf` for the DC, `roundf` for the first-order pair.
  `--cuda-check` check 7 compares symbols, plane and codes exactly.
- **ntc_decode.** Reads v13, reconstructs level 0 with the same strict-FP code (its `dct_inv8` skips
  zero coefficients in the first pass, bit-exact: adding `0.0f * B` to a float sum leaves it
  unchanged and the sum never becomes -0.0f), prints the level-0 line as `dct 8x8, q 50, dc step 4,
  4-bit scale codes, N nonzero ACs/block, P% EOB-only`, a `dct` line with the cheap statistics and
  raw bpp, and an `idct` line with its own timing and the MAC/texel count.

## 2. Flags

| flag | meaning |
|---|---|
| `--dct-q Q` | level 0 is an 8x8 DCT-coded selector plane, Q = 1..100 (100 = every AC step 1, near lossless; 1 = coarsest; 0 = off, the default) |
| `--dct-q Q0,Q1[,Q2,Q3]` | since section 7: one quality per level-0 channel (`--latent 0 0 C`, C = 1..4); a single value applies to every channel |
| `--dct-start F` | fit the scale codes and start snapping at `ceil(F * iters)` (default 0.5; 0 = from the start); the shadow is clamped to [-1,1] from the first iteration either way |
| `--dct-dc-step N` | uniform DC step in [0,64]-plane DC units, 1..64 (default 4); independent of Q |
| `--dct-block N` | DCT block size; only 8 is accepted in the MVP |
| `--dct-stats` | print the full `dct <keyword>` block at every progress print |
| `--dct-map KIND` | `nz` (default) / `code` / `bits` / `lnz` / `dconly` / `all` |
| `--dct-selftest` | run the self-test and exit 0 / 1 |

Refusals verified (each prints the message and exits 1): `--dct-q 101`, `--dct-dc-step 0`,
`--dct-block 4` ("only 8 in the MVP"), `--dct-start 1.5`, `--dct-map foo`, `--dct-q 50 --qat 2`
("replaces --qat on level 0; drop one"), `--dct-q 50 --qes 8,8` ("use --qes 0,B"), `--filter
bilinear,bilinear`, `--latent 256 256 1` (not full resolution), `--latent 0 0 2` (multi-channel was a follow-up, lifted in section 7; this refusal is gone) (multi-channel is a
follow-up), a 500x500 image without `--block` ("needs an image size that is a multiple of 8: pad
with --block"). Notes: `--dct-start 0.2` without `--dct-q` ("ignored without --dct-q"); `--dct-q 50
--qes 4` applies the single value to level 1 only (rc 0). `--block 6` with `--dct-q` pads
500x500 to 504x504 ("a multiple of 24x24, the lcm of the latent cell and the --dct-block 8 DCT
block"); with the flag off the padding multiple is unchanged.

## 3. Verification (commands and output; every figure from the final binaries)

Builds: `cmake --build build --config Release` and `cmake --build build_cuda --config Release`,
both with 0 compiler warnings after every step; the CUDA build prints only the pre-existing
`LNK4098` defaultlib warning.

### Firewall (flag off): the regression pin and a byte comparison against the `pre-dct-mvp` binaries
```
build\Release\ntc.exe --crop 512 --mlp-pairs 32 --iters 200 --print-every 200 --save-every 200 --out out_reg
iter    200  mse 269.314  psnr  23.83 dB  best  23.83 | q8 psnr  23.83 @ 0.552 bpp (ent 0.500, ctx 0.084) | mlp batch 0.00442 dstd 9.65e-05 | lat mean -0.051 sd 0.551 max 2.08 | 8.6s
xxd -l 4 out_reg/model.bin -> 3c43 544e (v12)
sha256 out_reg/model.bin: cb88470777e1d041c8803bcf3ca531434b9a2b2b5bce2ef88e031fbc86d69600, identical before and after the change
build\Release\ntc_decode.exe out_reg/model.bin --compare out_reg/recon_q_final.png --verify
compare t0: PSNR 88.70 dB, max |diff| 1, |diff| histogram: 0: 786363 (99.9912%), 1: 69 (0.0088%), 2: 0, >2: 0
verify   : pre-nonlinearity: max 0 ulp (0 of 786432 values differ); output fp32: max 42 ulp, max |diff| 7.75e-07; RGB8 mismatches: 68 of 786432 (0.00865%); rc=0
```
The `pre-dct-mvp` sources were built in the scratchpad (`git archive`, same CMake flags) and
compared with the new binaries: the regression log (timing fields excluded), `model.bin`,
`recon_q_final.png`, `side_by_side.png` and `latent_000200.png` are byte-identical; `ntc_decode`'s
PNG for `out_reg/model.bin` and for the v12 `out_image3_b8_c1q4_c4bilin_qes8_mlp27_cuda8k_fd50`
file is byte-identical between the baseline and the new decoder; the V08 R4 loads (below) print
the same lines with both binaries.

### `--dct-selftest` (CPU and CUDA builds: `dct-selftest: all passed`, rc 0)
```
  basis literals               PASS  0 of 64 literals differ from the double product rounded to float
  A_k literals                 PASS  0 of 16 A_k literals differ from 1 / g_k rounded to float
  round trip                   PASS  max |inv(fwd(w)) - w| = 2.289e-05 over 1000 blocks
  q 100 quantizer              PASS  steps all 1: yes; max |err| DC/first-order 0.4998 (<= 0.5), other ACs 0.9995 (<= 1), spatial rms 0.3050 (< 1)
  zigzag                       PASS  0 of 64 entries differ from the JPEG literal
  table K.1 at q 50            PASS  S(50) = 1; 0 AC entries differ from K.1[u][v] (54 from the transpose, which must be nonzero); [0] = 4 (dc_step 4)
  scale codes                  PASS  0 of 16 codes do not round-trip; monotone yes; g_0 0.2188 g_15 112.0
  dead zone                    PASS  (1,0): 0.4L->0 0.6L->1; (2,0): 0.6L->0 0.99L->0 1.0L->1 1.1L->1; (0,2): -1.5L->-1; dequant 1@(2,0) = 1.5L, 3@(1,0) = 3.0L, -2@(2,3) = -2.5L
  DC rounding / step range     PASS  DC 2/4 -> 0, 6/4 -> 2, 10/4 -> 2 (half-even 0, 2, 2); q 1 steps 4..27657, q 90 k 7 step(0,1) = 1
```

### `--cuda-check`: the README's five lines (unchanged figures) and the new line at q 50 / 90 / 20
Lines 1-5 (`model.png --qat 3,1`, default, `m1..m4 --qat 2`, chief1 `--qat 2 --qes 0,8`, chief1 `--qes
6,8`): each `cuda-check: all passed`, no check 7 printed; decode 1.788e-07 / 1.788e-07 / 1.192e-07 /
1.192e-07 / 1.192e-07, latent ES grad 2.126e-05 / 7.586e-06 / 5.538e-05 / 3.821e-05 / 3.931e-05, qat
search `3 of 524288` (worse 0, better 1, ties 1) / `3 of 786432` (worse 2) / `1 of 262144` (ties 1),
qes snap 0 differing values before and after the step on every line.
```
ntc chief1.png --cuda --cuda-check --block 8 --latent 0 0 1 --latent2 0 0 4 --filter nearest,bilinear --pos lv1local --dct-q 50 --dct-start 0 --qes 0,8 --qes-start 0 --mlp 17,17 --leak 0.0009765625 --iters 5 --print-every 5 --out out_chk
iter      0  dct: scale codes fitted ... psnr 11.28 -> 11.28 dB; nz 0.04/blk, eob-only 96.0%, clamped 0.00%; histogram k0..k15 0 0 294 3802 0 0 0 0 0 0 0 0 0 0 0 0
  decode          : max |gpu - cpu| = 1.192e-07  PASS
  mlp ES dl       : max |gpu - cpu| / rms(dl) = 3.442e-07 over 8 pairs  PASS
  mlp FD dl       : max |gpu - cpu| / rms(dl) = 8.361e-05 over 24 weights  PASS
  latent ES grad  : max |gpu - cpu| / rms(grad) = 3.085e-05 over 278528 values  PASS
  qes snap        : device snapped copy vs host snap of the downloaded shadow: 0 of 278528 values differ, 0 after a latent step with lr 0.02  PASS
  dct snap        : symbols 0 of 262144 differ, plane 0 of 262144 values differ (max 0 ulp), codes 0 of 4096 differ (device plane from k_dct_recon on the host's symbols: IDCT + clamp parity); after a latent step with lr 0.02: 0 / 0 (max 0 ulp) / 0 (k_dct_snap: forward DCT + quantizer + IDCT parity)  PASS
cuda-check: all passed
--dct-q 90: nz 3.95/blk, eob-only 1.4% at the fit; the same six PASS lines (mlp ES 3.746e-07, FD 7.959e-05, latent 3.974e-05), dct snap 0 / 0 / 0 before and after; all passed
--dct-q 20: nz 0.00/blk, eob-only 100.0% at the fit; mlp ES 3.670e-07, FD 8.316e-05, latent 3.083e-05; dct snap 0 / 0 / 0 before and after; all passed
```
(At iteration 0 the plane is the initial `N(0, 0.1)` draw, so few ACs are nonzero; the trained-model
parity is covered by the 500-iteration comparison and the v13 load equality below.)

### Step-3 smoke runs (CPU, chief1 layout of plan 8.7, `--rng hash --iters 100 --print-every 20 --mlp-pairs 32`)
```
--dct-q 100: iter 50 dct: psnr 16.82 -> 16.82 dB; nz 55.23/blk; codes k2 3 k3 799 k4 2990 k5 304; step1 by k all 1 (all-ones 100.0%)
             iter 100 psnr 24.22 dB | dct100,q8 psnr 24.22 @ 14.767 bpp (ent 5.998, ctx 5.723) | dct nz 56.12 lnz 62.8 eob0 0.0% clamp 5.75% bits/blk 911.1/351.8/334.2
             dct bits raw dc 32768 run 1609013 mag 1838872 sign 229859 eob 4949 code 16384 hdr 96 total 3731941 /blk 911.12 = 14.2362 bpp
             (hand check: 4096 x 8 + 229859 x (7 + 8 + 1) + 707 x 7 + 4096 x 4 + 96 = 3731941)
--dct-q 30:  iter 50 dct: psnr 16.82 -> 16.62 dB; nz 3.81/blk; step1 by k 84 55 36 24 16 10 7 5 3 2 1 1 1 1 1 1; step63 by k 754 ... 1
             iter 100 psnr 22.93 dB | dct30,q8 psnr 22.93 @ 2.190 bpp (ent 1.089, ctx 1.025) | dct nz 5.45 lnz 11.3 eob0 0.3% clamp 6.16%
             done: ... 22.93 dB at 2.190 bpp raw, 1.089 bpp entropy-coded, 1.025 bpp ... [level 0 alone: 0.523] [dct: raw 1.659 h0 0.587 ctx 0.523 code 0.062/0.011 bpp; nz 5.45/blk eob0 0.3%]
             model.bin magic 3d43 544e (v13); --dct-map all wrote dct_map_{code,bits,lnz,dconly}_000100.png, dct_map_*.png, latent_q_*, dct_resid_*, dct_side_by_side.png
```

### CPU vs GPU, 500 iterations, `--rng hash` (plan 8.7)
`chief1.png --rng hash --iters 500 --lr-anneal 0.5 0.05 --mlp-fd 0.5 --mlp-pairs 32 --print-every 100
--block 8 --latent 0 0 1 --latent2 0 0 4 --filter nearest,bilinear --pos lv1local --dct-q 50 --qes 0,8
--mlp 17,17 --leak 0.0009765625`, `out_dct_a_cpu.log` and `out_dct_a_gpu.log` (`--cuda`):
```
both     iter 200 psnr 31.84 dB | q0,8 psnr 31.83 @ 8.531 bpp (ent 8.066, ctx 3.263)
both     iter 250 dct: psnr 33.86 -> 29.48 dB; nz 14.02/blk, eob-only 0.0%, clamped 4.68%; histogram k0..k15 0 3 9 24 82 373 550 3055  (identical on CPU and GPU)
both     iter 250 psnr 28.45 dB | dct50,q8 psnr 28.45 @ 4.336 bpp (ent 1.975, ctx 1.809) | dct nz 14.03 lnz 24.4 ... bits/blk 243.5/95.6/85.0
CPU      iter 500 psnr 29.95 dB | dct50,q8 psnr 29.95 @ 5.407 bpp (ent 2.359, ctx 2.099) | dct nz 18.32 lnz 39.1 eob0 0.0% clamp 6.22% bits/blk 312.0/119.8/103.1 chg sym 43966 zq 252643
GPU      iter 500 psnr 29.98 dB | dct50,q8 psnr 29.98 @ 5.398 bpp (ent 2.356, ctx 2.096) | dct nz 18.29 lnz 39.1 eob0 0.0% clamp 6.24% bits/blk 311.5/119.6/103.0 chg sym 43985 zq 252461
CPU      done: final psnr 29.95 dB (best 31.84) ... 29.95 dB at 5.407 bpp raw, 2.359 bpp entropy-coded, 2.099 bpp ... [level 0 alone: 1.611] [dct: raw 4.875 h0 1.871 ctx 1.611 code 0.062/0.011 bpp; nz 18.32/blk eob0 0.0%] | 36.7s
GPU      done: final psnr 29.98 dB (best 31.84) ... 29.98 dB at 5.398 bpp raw, 2.356 bpp entropy-coded, 2.096 bpp ... [level 0 alone: 1.608] [dct: raw 4.867 h0 1.868 ctx 1.608 code 0.062/0.011 bpp; nz 18.29/blk eob0 0.0%] | 1.2s
reprobe  CPU: codes that would move >= 1: 65.9%, >= 2: 13.4%;  GPU: 65.3%, 13.4%
```
The two runs are identical through the switch (same code histogram, same switch line, same
iteration-250 line) and agree to display precision at 500 (0.03 dB, 0.009 bpp raw), the QES_NOTES.md
pattern. The PSNR drop at the switch (33.86 -> 29.48 dB at q 50) and the partial recovery in the
remaining 250 annealed iterations are the run's real behaviour, not a wiring fault (`changed`
at the switch = 262144 of 262144).

### v13 `--load --iters 0` equality, refusals, decoder (plan 8.6)
```
ntc chief1.png --cuda --block 8 --latent 0 0 1 --latent2 0 0 4 --filter nearest,bilinear --pos lv1local --dct-q 50 --dct-start 0 --qes 0,8 --qes-start 0 --mlp 17,17 --leak 0.0009765625 --iters 5 --print-every 5 --out out_chk
iter      5  psnr 11.55 dB | dct50,q8 psnr 11.55 @ 0.914 bpp (ent 0.564, ctx 0.561) | dct nz 0.34 lnz 1.8 eob0 67.4% ...; out_chk/model.bin magic 3d43 544e
--load out_chk/model.bin --iters 0, CPU and --cuda: iter 0 psnr 11.55 dB; done: ... 11.55 dB at 0.914 bpp raw, 0.564 bpp entropy-coded, 0.561 bpp ... [level 0 alone: 0.074] [dct: raw 0.382 h0 0.077 ctx 0.074 code 0.062/0.011 bpp]  (both backends, same line as the run)
--load out_dct_a_gpu/model.bin --iters 0, CPU and --cuda: done: final psnr 29.98 dB ... 29.98 dB at 5.398 bpp raw, 2.356 bpp entropy-coded, 2.096 bpp ... [level 0 alone: 1.608] [dct: raw 4.867 h0 1.868 ctx 1.608 ...]  (= the run's final line)
without --dct-q, with --dct-q 30, with --dct-dc-step 8: "out_chk/model.bin was saved with --latent 512 512 1 --latent2 64 64 4 --filter nearest,bilinear --leak 0.000977 --qes 0,8 --dct-q 50 --dct-dc-step 4 --mlp 17,17 --pos lv1local and 1 texture(s); pass the same options", rc=1
ntc_decode out_chk/model.bin --compare out_chk/recon_q_final.png --verify
level 0  : 512x512x1 nearest, dct 8x8, q 50, dc step 4, 4-bit scale codes, 0.3 nonzero ACs/block, 67.4% EOB-only, fp32 planar
dct      : nz mean 0.34/blk, eob0 67.4%, lnz mean 1.8, clamped 0.00%, codes k0..k15 0 0 294 3802 0 ...; raw 24.5 bits/blk = 0.3826 bpp
idct     : 1.13 ms single-threaded (4096 blocks; 8.2 MAC/texel with zero-skipping, 16 dense; nnz 1399)
compare t0: PSNR 88.40 dB, max |diff| 1, 0: 786358 (99.9906%), 1: 74 (0.0094%); verify: pre-nonlinearity max 0 ulp (0 of 786432 differ); output fp32 max 24 ulp; RGB8 mismatches 75; rc=0
--q8 and --fp32-latent outputs: cmp identical
ntc_decode out_dct_a_gpu/model.bin --compare ... --verify: level 0 dct 8x8, q 50, ..., 18.3 nonzero ACs/block, 0.0% EOB-only; idct 2.44 ms (10.4 MAC/texel, nnz 74897)
compare t0: PSNR 89.23 dB, max |diff| 1, 1: 61; verify max 0 ulp (0 of 786432); output fp32 max 42 ulp; RGB8 mismatches 60; --q8 = --fp32-latent (cmp)
ntc_decode out_dct_a_cpu/model.bin ...: compare PSNR 89.31 dB, max |diff| 1, 1: 60; verify max 0 ulp; idct 2.23 ms (10.4 MAC/texel, nnz 75026); --q8 = --fp32-latent
```

### `ntc_decode --compare --verify` on the v10 / v11 / v12 files (plan 8.3; V08_NOTES.md R3 figures)
```
out_model10_b6_c1q2_c4bilin_mlp17_cuda8k_fd50 (v10, 3a43544e): PSNR 90.42 dB, max |diff| 1, 1: 149; verify max 0 ulp (0 of 2524500); RGB8 mismatches 150; rc=0
out_model12_b8_c1q2_c4bilin_mlp27_cuda8k_fd50 (v11, 3b43544e): PSNR 89.53 dB, max |diff| 1, 1: 513; verify max 0 ulp (0 of 7077888); mismatches 513; rc=0
out_image3_b8_c1q4_c4bilin_qes8_mlp27_cuda8k_fd50 (v12, 3c43544e): PSNR 90.66 dB, max |diff| 1, 1: 353; verify max 0 ulp (0 of 6322560); mismatches 339; rc=0
out_m1234 (v12, 3c43544e, 4 textures): t0..t3 PSNR 89.45 / 88.28 / 87.84 / 92.04 dB, max |diff| 1; verify max 0 ulp (0 of 3145728); mismatches 243; rc=0
```

### Old files in the trainer (V08_NOTES.md R4 commands)
```
v9  out_2lv_512c1_128c4_qat2:  done: final psnr 30.80 dB ... 30.76 dB at 4.107 bpp raw, 3.737 bpp entropy-coded
v10 out_chief1_512c1q2_128c2_cuda8k_fd50 (--mlp 36,36 --leak 0.0009765625): done: 34.55 dB ... 34.52 dB at 3.102 bpp raw, 2.954 bpp entropy-coded
v12 out_m1234: done: final psnr 25.45 dB tex 23.17 31.47 23.22 29.55 ... 25.45 dB qtex 23.17 31.47 23.22 29.54 at 2.635 bpp raw, 2.213 bpp entropy-coded
v12 out_image3_..._qes8 (--qat 4 --qes 8): qes ranges restored; done: 48.94 dB ... 48.94 dB at 4.508 bpp raw, 4.146 bpp entropy-coded, 1.295 bpp ... [level 0 alone: 0.866]
```
V08_NOTES.md R4 quotes 34.57 / 34.53 and tex 31.48 / 29.56 for the v10 and m1234 lines; those
were recorded before the v0.8.1 basisu PSNR definition. The `pre-dct-mvp` baseline binary prints
exactly the lines above, so the difference predates this change.

### Fast-sweep mode (plan 12.E) on a float-selector v12 reference
Reference: the 8.7 command without `--dct-q` (`out_dct_ref_gpu.log`): `iter 500 psnr 41.25 dB |
q0,8 psnr 41.14 @ 8.531 bpp (ent 7.112, ctx 4.279)`, level 0 `lat max 3.67` (a float selector
trained without the [-1,1] clamp). `chief1.png --block 8 --latent 0 0 1 --latent2 0 0 4 --filter
nearest,bilinear --pos lv1local --qes 0,8 --mlp 17,17 --leak 0.0009765625 --load
out_dct_ref_gpu/model.bin --iters 0 --dct-q Q`, CPU:
```
note: the model's continuous level 0 is the warm start for --dct-q Q: the scale codes are fitted and the plane snapped at --dct-start (iteration 0)
codes (all Q): histogram k0..k15 0 1 7 21 38 406 3084 539
Q 100: psnr 31.99 -> 31.91 dB; nz 51.62/blk; done 31.91 dB at 13.661 bpp raw, 5.580 entropy-coded, 5.247 ctx [level 0 alone: 4.762] [dct: raw 13.129 h0 5.095 ctx 4.762]  0.3s
Q 50 : psnr 31.99 -> 28.37 dB; nz 12.47/blk; done 28.37 dB at 3.945 / 1.824 / 1.681 [level 0 alone: 1.196] [dct: raw 3.414 h0 1.339 ctx 1.196]  0.2s   (--cuda: identical line)
Q 10 : psnr 31.99 -> 24.92 dB; nz 3.87/blk, eob-only 2.9%; done 24.92 dB at 1.797 / 0.944 / 0.901 [level 0 alone: 0.416] [dct: raw 1.265 h0 0.459 ctx 0.416]  0.2s
Q 1  : psnr 31.99 -> 20.12 dB; nz 0.18/blk, eob-only 83.1%; done 20.12 dB at 0.873 / 0.611 / 0.606 [level 0 alone: 0.121] [dct: raw 0.341 h0 0.126 ctx 0.121]  0.2s
Q 50 --dct-stats --resave: resave : wrote out_sweep_q50_resave/model.bin as v13 (level 0 DCT-coded, q 50, DC step 4; codes fitted and symbols snapped in this run from the loaded continuous level 0); magic 3d43 544e;
      reloading it with --dct-q 50 --iters 0: done 28.37 dB at 3.945 / 1.824 / 1.681 (same line); dct reprobe: codes that would move >= 1: 0.0%
      dct nz mean 12.47 median 12 max 29 zero% 80.2 (interior% 16.4 trailing% 63.8); dct lnz mean 22.8 median 21 max 60 ...; dct run mean 0.83 run0% 71.0 (nonzero tokens 51069; EOB tokens 4096)
      dct dc range 6..114 mean 57.11 h0 6.05 hres 5.85 bits/blk; dct blkbits raw mean 214.5 median 207.0 max 479 | h0 mean 84.6 median 81.7 max 205.6
      dct bits raw dc 32768 run 357483 mag 408552 sign 51069 eob 28672 code 16384 hdr 96 total 895024 /blk 218.51 = 3.4142 bpp
      dct bits h0  dc 23957 run 87234 mag 168718 sign 51069 eob 15366 code 4686 hdr 96 total 351126 /blk 85.72 = 1.3394 bpp
      dct bits ctx dc 23957 run 72169 mag 155056 sign 51069 eob 7742 code 3499 hdr 96 total 313588 /blk 76.56 = 1.1962 bpp
      dct plane rms 4.502 max 35.66 (units of 64) clamped% 6.60 changed% 94.5
```
Note the "before" PSNR of 31.99 dB against the reference's 41.25 dB: the reference's float selector
exceeds [-1,1] (max 3.67) and the pre-start clamp alone costs 9.3 dB on it. A reference for the
Q = 100 point of a real sweep should be trained with a clamped level 0 (which `--dct-q 100` from
the start is), or the clamp loss must be read off the switch line as here.

## 4. Deviations from the plan, with reasons

1. **Region placement.** The plan puts the region after `qes_spec` (before the tap section); it is
   placed after `mlp_forward` and before the positional-encoding section instead, because the probe
   calls `mlp_forward` and `MLP`, which are defined between the two points. To keep one region, the
   probe (`dct_probe_codes`) is a function template taking the feature builder as a callable;
   `Decoder::dct_probe` supplies `features(zdec(), px, py, f)`. Consequently `context_entropy_bits`
   (defined later, with `bitrate_stats`) is forward-declared inside the region.
2. **Basis literal sanity check.** Plan 13.5 asks for a check "within 1 ulp of `sqrtf(...) * cosf(...)`".
   The float product differs from the double-rounded literal by up to 82 ulp near the small entries
   (cosf of a float angle near a zero crossing), so the self-test compares each literal with the
   double product rounded to float and requires exact equality (0 of 64 differ), the same for A_k.
3. **`--dct-dc-step` mismatch on a v13 load is refused** as well as a different q (the symbols'
   DC is in units of the file's step; without a shadow it cannot be re-quantized); the mismatch
   message prints ` --dct-q 50 --dct-dc-step 4`.
4. **Check 7's pre-step comparison** re-uploads the host's symbols (`set_dct`) before comparing, so
   the device plane really comes from `k_dct_recon` (13.12): check 6 has already run a latent step
   by then, and without the re-upload the device plane would come from `k_dct_snap`. The post-step
   comparison after `lat_step(3, ...)` is the full forward-path test.
5. **`--load --iters 0` of a v13 with a different `--qes`** re-snaps (the qes warm start calls
   `qes_fit_all` -> `qes_refresh` at iteration 0); the "no re-snap" guarantee holds for the same
   `--qes` (the verified case) and for any resumed training, where the first latent step re-snaps
   anyway.
6. **Progress line before the switch** charges level 0 at `--qbits` (the continuous fall-through),
   the banner prints the levels >= 1 raw figure only and says so (12.B over 13.17's alternative).
7. **The `plane` line's `changed%`** is the fraction of level-0 values whose snapped value differs
   from the clamped shadow (the codec's own effect); the per-print churn asked for in 13.8 is the
   separate `chg sym N zq M` token on the progress line (symbols / level-0 values changed since the
   last print).
8. **`firstorder%` / `<=2nd%`** are defined as the fraction of blocks with >= 1 nonzero whose last
   nonzero sits at zigzag <= 2 (the first-order pair) / <= 5 (through the second diagonal).
9. **The idct line is printed whenever a v13 file is decoded**, not only under `--bench` (it is one
   line and its timing is measured anyway); the timing note names the exclusion.
10. **`fp-contract=off` for GCC / Clang** was added to `STRICT_FP_BEGIN` in main.cpp as plan section 3
    asks; per 13.7 this is not arithmetic-neutral for the qes grid on those (unpinned) builds and
    changes nothing on MSVC, which already had `fp_contract(off)`.
11. **A stray byte in main.cpp.** Line 162 (`std::string load; // load model.bin instead of random
    init`) ended in `\r\r\n` at the tag, which makes git classify the file as `-text`; the Edit tool
    normalized it and it was put back byte for byte so the committed diff stays the real one
    (`git diff --ignore-cr-at-eol --numstat`: main.cpp 974 added / 21 removed lines, cuda/ntc_cuda.cu
    155 / 2, cuda/ntc_cuda.h 17 / 0, ntc_decode.cpp 185 / 6, README.md 39 / 1).
12. The default `--dct-map` for the `all` kind writes the extra maps (`code`, `bits`, `lnz`, `dconly`)
    at the final iteration and in the final block only; the per-snapshot map is always `nz`.

## 5. Files, regions and hooks (the removal list)

Every added line outside the two regions carries `[DCT]` (`grep -n "\[DCT\]"` finds all of them;
verified by a script over `git diff -U0 --ignore-cr-at-eol` that no added line outside a region lacks
the tag). Removal = delete the region between its two markers and every `[DCT]`-tagged line or
hunk; the tagged hooks that replace an existing line say `was ...` or `hook: ...` in their comment.

- `main.cpp` (CRLF kept): the two DEPENDENCY bullets; `Options` (`dct_q`, `dct_start`, `dct_dc_step`,
  `dct_block`, `dct_selftest`, `dct_stats`, `dct_map`, `dct_opts_given`); `usage()` (eight lines and
  the `--resave` text); the `STRICT_FP_BEGIN` contraction pragmas for GCC / Clang; the region
  `// ---- DCT: transform-coded level 0 [DCT]` ... `// ---- end of the DCT region [DCT]` after
  `mlp_forward` (tables, `DctLevel`, the strict-FP step builder / transform / quantizer / snap /
  recon / clamp, `dct_code_of_gain`, `dct_probe_codes`, `DctStats` / `dct_analyze`, `dct_plane_stats`,
  `dct_spec`, `dct_print_codes` / `dct_print_stats`, the map / residual PNG writers, `dct_selftest`);
  `Decoder` (`dct`, the `zdec()` / `zdec_mut()` conditions, the two `qes_refresh()` hooks,
  `dct_probe`, `dct_fit_all`); `save_model` (magic, header fields, payload branch); `bitrate_stats`
  (the first-statement branch); `cuda_check` check 7; `main()`: the seven parser lines, the
  `--dct-selftest` dispatch, the range checks and note, the lcm padding and its note, the
  `|| o.dct_q > 0` in the single-value `--qes` rule, the layout checks and level-0 setup block,
  `dct_from_file`, the loader (v13 magic, `saved_dct_*`, the header reads, `dct_ok`, the mismatch
  text, the payload branch, the restore block, the warm-start note), the post-load clamp call,
  `dct_start_it` / `dct_begin`, `dct_sync_device`, the `ModelDesc` fills and the `dct_sync_device()`
  call after `qes_sync_device()`, the banner (`continue` in the raw loop, the `bitrate` /
  `bpp/tex` variants, the `dct` line), `qlabel_base` / the `qlabel` lambda / `dct_last_sym` /
  `dct_last_zq`, the in-loop switch block and `|| dct_switch`, the progress-line token group and
  `--dct-stats` print, the snapshot PNG block, the `--resave` line, the final statistics / reprobe /
  map block, the `done:` line prefix and the `[dct: ...]` bracket.
- `cuda/ntc_cuda.h` (LF kept): `MAX_DCT_CH`, the `ModelDesc` fields, `DctDesc`, `set_dct`,
  `download_dct`.
- `cuda/ntc_cuda.cu` (LF kept): the DEPENDENCY paragraph; the region (`c_dct`, `c_dct_basis`,
  `c_dct_step`, `dev_dct_*`, `k_dct_clamp0`, `k_dct_snap`, `k_dct_recon`); `Impl` fields; the two
  `cudaFree`s; the `init` envelope checks, the constant uploads, the `|| I.dct` allocation condition
  and the two buffers; `snap_latent_impl` (`&& !I.dct` and the kernel dispatch); `set_dct`,
  `download_dct`.
- `ntc_decode.cpp` (CRLF kept): the header comment lines; `Model` fields; the v13 loader
  (version range, header fields, payload branch); the region after `q8_quantize` (literals, tables,
  `dct_build_steps`, the zero-skipping `dct_inv8`, dequantizers, `DctReconStats` / `dct_recon_level0`);
  the `level_quantized` clause; the timed reconstruction call; the `--pack-selectors` reason; the
  level-0 summary branch; the `dct` / `idct` lines (since September 8, 2026 the inverse DCT runs on the decode stage's thread pool, block rows split across threads with per-thread stats summed, output bit-identical at any thread count; the `idct` line prints the thread count, and earlier `single-threaded` idct figures in this file are not comparable to it); the timing-note `%s`.
- `README.md` (CRLF kept): the `--dct-q` paragraph after the `--qes` paragraph, seven flag rows, the
  sixth `--cuda-check` line and the check-7 sentence. New: `DCT_NOTES.md` (this file).
- Test outputs (untracked, like the qes ones): `out_dct_s100`, `out_dct_s30`, `out_dct_a_cpu(.log)`,
  `out_dct_a_gpu(.log)`, `out_dct_ref_gpu(.log)`, `out_sweep_q{100,50,10,1}`, `out_sweep_q50_gpu`,
  `out_sweep_q50_resave`, `out_load_dct_{cpu,gpu}`, `out_chk`, `out_reg`, `out_val`, `out_load_a`.
- Multi-channel additions (section 7, September 7, 2026; every line tagged `[DCT]`, most with `hook: was ...`):
  `Options::dct_q_ch`; the `--qat`-style parser loop for `--dct-q` (restore the one-line `atoi`); the validation
  lines (a)-(d) after the old range check; the `MAX_DCT_CH` refusal that replaced the `o.LC != 1` refusal;
  `D.dct.q = o.dct_q_ch` (restore `q.assign(o.LC, o.dct_q)`); `DctLevel::last_clamped_c` and the per-channel
  `nclamp_c` atomics in `dct_snap_level0` / `dct_recon_level0`; the `only_c` parameter and the `bi()` index of
  `dct_analyze`, the `only_c` filter of `dct_plane_stats`, the channel argument of `dct_print_codes`,
  `dct_print_codes_all`, `dct_print_channel` and the `dct[c]` loop at the end of `dct_print_stats`; the
  `dct_print_codes_all` calls in `dct_begin` / `dct_refit_now` and the `(ch ...)` clause with the `m1c` argument of
  `Decoder::dct_refit`; the side-by-side width (`mp.w`, restore `D.W`); the `%s` warm-start note and raw-bpp
  banner (restore `%d`, `o.dct_q`) and the probe clause of the `dct` banner; the channel count in the check-7
  PASS line; the usage text and the two DEPENDENCY comment lines; ntc_decode's `nnz_c` / `eob0_c` and its
  `per channel` line (and its `bits/(blk,ch)` label); the README rows and the seventh `--cuda-check` line; the DCT-region
  header comment, the `MAX_DCT_CH` comment, the conditional-sensitivity comment before `dct_probe_codes`, the `--dct-q`
  help block, the `dct_gain_clause` helper and its call in the switch / refit lines, and the `of N each` denominator of the refit clause.
- Rate proxy in the loss (`--dct-lambda`, section 8, September 7, 2026; DCT_RATE_PLAN.md section 9 has the same list): `Options::dct_lambda`, `dct_rate`,
  `dct_rate_given`; the two parser lines, the four usage lines, the three validation lines and the two note texts; the `DctLevel` rate fields
  (`lambda`, `rate_es`, `rate_trunc`, `lam_t16`, `bits`, `last_truncated`, `last_truncated_c`, `last_nz_before`, `rate_hits`, `rate_evals`); the cost
  model (`DCT_C_SIGN`, `DCT_C_EOB`, `dct_floor_log2`, `dct_c_run`, `dct_c_mag`), `dct_block_bits`, `dct_truncate_block`, `dct_quantize_block`,
  `dct_rate_block` (strict FP), `dct_proxy_totals`, `dct_rate_mode`, `dct_print_rate`; the `dct_snap_block` hook (restore the one-line quantize
  loop and the two-argument signature), the `dct_recon_block` bits line and its non-const `Q`, the `dct_snap_level0` reductions and per-channel
  truncation atomics; the `dct_print_channel` clause and the `dct_print_rate` call in `dct_print_stats`; the three self-test items; `NTC_NOINLINE`
  and `dct_rate_pass`, the `LatentTrainer::step` / `step_impl<RATE>` split (restore `void step(...)` with the original body; `dbits`); `dct_desc_of`,
  `cuda_check_rate_replica`, the check-4 call and the `dct rate` line, the `bits` clause (`lam`, `bitsd`, `dbits_all`, `bcl`, `bcl2`) of the
  `dct snap` / `dct refit` lines and the three `dct_desc_of` call sites (restore the two-line `DctDesc` fills); the level-setup lines; the
  switch / refit `trunc` clause; the banner clause; the progress-line `proxy` / `trunc` / `hits` (restore the one-line printf with the
  `chg` clause); the `--cuda` `download_dct_rate` call at print time; the `done:` line's `lamcl`; the `md.dct_zigzag` fill; the DEPENDENCY
  bullet. `cuda/ntc_cuda.h`: `ModelDesc::dct_zigzag`, the `DctDesc` rate fields, `download_dct_rate`. `cuda/ntc_cuda.cu`: the DEPENDENCY lines,
  `c_dct_zigzag`, `dev_dct_floor_log2` / `dev_dct_c_run` / `dev_dct_c_mag` / `dev_dct_block_bits` / `dev_dct_truncate_block` /
  `dev_dct_quantize_block`, the `k_dct_snap` hook (restore the one-line quantize loop and the four-argument signature) and the `k_dct_recon`
  bits argument, `k_dct_rate_pair`, `k_dct_rate_gather`, the `Impl` fields, the three buffers and their frees, the zigzag check and upload in
  `init`, the `snap_latent_impl` counter reset and arguments, the `set_dct` flag copies, `download_dct_rate`, the `lat_step` hook. README: the
  two flag rows, the `--dct-lambda` paragraph, the two check lines and the `dct rate` sentence. ntc_decode.cpp, the v13 file and the NTCB
  container are untouched (a truncated block is a legal block).

## 6. Open items

1. The 8000-iteration q sweep of plan 8.8 / 12.D (model15 and image3, bracket 100 / 50 / 10 first,
   `--dct-start 0.25`, `--dct-dc-step 8` and `2`) has not been run; the float-selector reference per
   layout should be trained with a clamped level 0 (see the fast-sweep note in section 3).
2. The 500-iteration run loses 4.4 dB at the switch and recovers 0.5 dB in 250 annealed iterations;
   the reprobe says 13.4% of the codes would move by >= 2 by the end of the run. Whether a refit
   option or the per-texel FD gradient (plan 11.1) is next is for the sweep to show.
3. `dct_analyze` runs twice per progress print (once inside `bitrate_stats`, once for the token
   group); cheap at 512x512, unmeasured at 2048x1152.
4. Multi-channel level 0, `--dct-block 4`, the soft rate penalty and the truncation RDO are the
   follow-ups of plan section 11; the layouts, file fields and kernels are already `[block][channel]`
   with `MAX_DCT_CH = 4`.

## Post-review changes (September 6, 2026)

Two reviews of the diff found no correctness defect; the math, quantizer and parity were verified
line by line against the Basis Universal sources, the firewall walked hook by hook (byte-identical
regression outputs against a pre-dct-mvp build), and an independent Python parse of a v13 file
reproduced the trainer's bit counts. Applied afterwards: `fp_contract(off)` added to
ntc_decode's MSVC strict-FP block (the DCT / table code has real multiply-add pairs; the build
has no /arch flag so this is a source guarantee rather than a flag dependency); the decoder's
raw bits figure no longer includes the 96-bit per-channel header, so it equals the trainer's
`[dct: raw ...]` bracket; `dct_analyze` counts magnitudes above 256 and prints them on the
`dct nz` line (must stay 0); the residual PNG gain is 2 instead of 8 (x8 saturated below q ~50);
the done-line code figure is labelled `code raw/ctx`; the `--dct-stats` block is not printed on
the last progress line (the final block follows); `--help` documents the fast
`--load <v12> --dct-q Q --iters 0` sweep; ntc_decode's help mentions the v13 DCT level. Open:
`dct_analyze` runs up to four times per progress print (harmless).


## Addendum: periodic scale-code refits (`--dct-refit`, September 7, 2026)

The codes were fitted once at the switch and frozen while the decoder kept adapting; the end-of-run
`dct reprobe` line of the 8000-iteration logs said 55-98% of the blocks would get a different code by
the end (`out_image3_b8_dct100_...`: 98.6% / 89.2%; the q 50 references below: 90.0% / 54.7% on image3,
62.1% / 7.3% on model15). Built on top of tag `dct-mvp-v1`; every number below is from the program
output of the commands shown, produced by the final binaries (both builds 0 warnings besides the
pre-existing LNK4098; the four 8000-iteration runs were made by a build of the same source before the
last comment-only edit, and the final binary reproduces the `--rng hash` GPU run below byte for byte:
`out_dct_r_gpu2.log` = `out_dct_r_gpu.log` apart from the timing fields, `model.bin` identical).

### What changed

- `--dct-refit N` (default 500; 0 = fit once and freeze, the previous behaviour): at every multiple of
  N after the switch (not the switch iteration, not the last iteration, so at least one latent step
  follows every refit) `Decoder::dct_refit` re-runs `dct_probe` on the current decoder, counts the codes
  that move by >= 1 / >= 2, replaces `dct.code`, and `qes_refresh()` re-snaps level 0 from the kept
  shadow (each block's rounding re-rolls by at most half of its new step). The host lambda
  `dct_refit_now` measures the PSNR before and after the re-snap with two host decodes; on `--cuda` the
  model is downloaded first (`download_model` + `qes_refresh`, the host snap with the old codes equals
  the device plane bit for bit, check 7) and the new codes / symbols / steps go up afterwards through
  the existing `dct_sync_device` -> `set_dct` -> `k_dct_recon` route; the device shadow `d_z` is never
  written. `k_dct_snap` reads the uploaded code buffer, so the next latent step quantizes with the
  refitted codes. No CUDA source change was needed.
- `--dct-refit-until F` (default 1.0): no refit after `ceil(F * iters)`. `DctLevel` carries
  `refit_every`, `refit_until_it`, `refits`, `refitted_at`; the `dct cfg` line prints
  `refit-every N refit-until IT refits R last-refit-at IT`; the banner says `codes fitted then,
  refitted every N iterations through iteration IT, then frozen`. The file stores the codes and symbols
  of the last refit (the v13 payload is unchanged); a loaded v13 with `--iters 0` prints `(codes restored
  from the file, frozen)`; a resumed training run refits the restored codes on the same schedule.
- Log line per refit: `iter N  dct: codes refitted (R); moved >= 1: X%, >= 2: Y% (of B); psnr A -> B dB;
  nz ..., eob-only ..., clamped ...; histogram k0..k15 ...`. The end-of-run `dct reprobe` line is kept.
- `--cuda-check` check 7 gained a `dct refit` line: `D.dct_refit` on the host, `set_dct`, compare
  (symbols / plane / codes must be 0 / 0 / 0), one more `lat_step`, host snap, compare again.
- Refusals: `--dct-refit -1` ("--dct-refit needs N >= 0 (0 = fit once and freeze)", rc 1),
  `--dct-refit-until 1.5` ("--dct-refit-until must be in [0, 1]", rc 1); both flags are listed in the
  "ignored without --dct-q" note. README: two flag rows, the `--dct-q` paragraph, the check-7 sentence.

Default justification. 500 iterations: on the 8000-iteration runs the first refit carries nearly all of
the movement (90.9% of the image3 codes, 26.8% of model15's) and by the fourth the rate is at or below
13.5% / 8.9%, so 7 host probes + 14 host decodes per run (image3: 84.8 s against the reference's 76.9 s) are
enough; 250 / 0.9 (12 refits) gained nothing (below). `--dct-refit-until 1.0`: with N = 500 the
last refit is at 7500 and 500 annealed iterations follow it; stopping at 0.9 (last refit 7000) left
more staleness at the end (13.6% / 5.8% of the codes would still move versus 5.6% / 2.3%) and 0.05 /
0.06 dB less PSNR, so 1.0 stays the default.

### Verification (commands and output)

Firewall: `build\Release\ntc.exe --crop 512 --mlp-pairs 32 --iters 200 --print-every 200 --save-every 200 --out out_reg`
```
iter    200  mse 269.314  psnr  23.83 dB  best  23.83 | q8 psnr  23.83 @ 0.552 bpp (ent 0.500, ctx 0.084)
xxd -l 4 out_reg/model.bin -> 3c43 544e (v12); sha256 cb88470777e1d041c8803bcf3ca531434b9a2b2b5bce2ef88e031fbc86d69600 (= section 3)
```
`--dct-selftest`: `dct-selftest: all passed` on both builds. Every added line carries `[DCT]`
(`git diff -U0 --ignore-cr-at-eol main.cpp`: 73 added / 3 removed lines, 0 added lines without the
tag; README 15 / 3; line endings unchanged, the `\r\r\n` of line 162 kept).

Frozen path unchanged: the section-3 GPU command (`chief1.png --rng hash --iters 500 ... --dct-q 50
--qes 0,8 --cuda`) with the default flags (no refit fires: 500 is the last iteration) and with
`--dct-refit 0` both print the section-3 lines (`iter 250 dct: psnr 33.86 -> 29.48 dB`, `iter 500 ...
29.98 dB @ 5.398 bpp (ent 2.356, ctx 2.096)`, reprobe 65.3% / 13.4%); `out_dct_a_gpu3/model.bin` is
byte-identical to `out_dct_a_gpu/model.bin` (`cmp`); the logs differ from the September 6 log only in
the two post-review labels (`mag>256 0`, `code raw/ctx`).

`--cuda-check`: the six README lines all print `cuda-check: all passed` (lines 1-5 as in section 3:
decode 1.788e-07 / 1.490e-07 / 1.192e-07 / 1.192e-07 / 1.192e-07, qat search `3 of 524288` (worse 0,
better 1, ties 1) / `3 of 786432` (worse 2) / `1 of 262144` (ties 1), qes snap 0 / 0 everywhere). The
DCT line at q 50 / 90 / 20:
```
ntc chief1.png --cuda --cuda-check --block 8 --latent 0 0 1 --latent2 0 0 4 --filter nearest,bilinear --pos lv1local --dct-q 50 --dct-start 0 --qes 0,8 --qes-start 0 --mlp 17,17 --leak 0.0009765625 --iters 5 --print-every 5 --out out_chk
  dct snap        : symbols 0 of 262144 differ, plane 0 of 262144 values differ (max 0 ulp), codes 0 of 4096 differ (...); after a latent step with lr 0.02: 0 / 0 (max 0 ulp) / 0 (...)  PASS
  dct refit       : host re-probe moved 195 of 4096 codes (>= 2: 0), uploaded through set_dct: symbols 0 differ, plane 0 (max 0 ulp), codes 0 (k_dct_recon with the refitted codes); after a latent step: 0 / 0 (max 0 ulp) / 0 (k_dct_snap quantizes with the refitted codes)  PASS
cuda-check: all passed
q 90: dct refit moved 193 of 4096 (>= 2: 0), 0 / 0 (max 0 ulp) / 0 before and after the step, all passed
q 20: dct refit moved 196 of 4096 (>= 2: 0), 0 / 0 (max 0 ulp) / 0 before and after the step, all passed
```

CPU vs GPU, 500 iterations, `--rng hash`, refits every 50 from the switch at 125: `chief1.png --rng hash
--iters 500 --lr-anneal 0.5 0.05 --mlp-fd 0.5 --mlp-pairs 32 --print-every 100 --block 8 --latent 0 0 1
--latent2 0 0 4 --filter nearest,bilinear --pos lv1local --dct-q 50 --dct-refit 50 --dct-start 0.25
--qes 0,8 --mlp 17,17 --leak 0.0009765625` (`out_dct_r_cpu.log`, `out_dct_r_gpu.log` with `--cuda`):
```
both  iter 125  dct: scale codes fitted ... psnr 26.44 -> 25.36 dB; nz 12.09/blk; histogram 0 1 27 272 384 228 3183 1 (identical)
CPU   iter 150 refitted (1) moved 13.5% / 0.0%, psnr 26.38 -> 26.42 | 200 (2) 43.0% / 0.0%, 26.92 -> 25.06 | 250 (3) 19.2%, 27.70 -> 27.78 | 300 (4) 24.4%, 28.58 -> 28.09 | 350 (5) 11.1%, 28.95 -> 28.93 | 400 (6) 6.2%, 29.19 -> 29.01 | 450 (7) 6.8%, 29.32 -> 29.32
GPU   iter 150 refitted (1) moved 13.5% / 0.0%, psnr 26.38 -> 26.42 | 200 (2) 43.8% / 0.0%, 26.92 -> 25.00 | 250 (3) 20.1%, 27.74 -> 27.83 | 300 (4) 23.4%, 28.53 -> 27.90 | 350 (5) 11.5%, 28.95 -> 28.97 | 400 (6) 6.2%, 29.15 -> 29.05 | 450 (7) 6.0%, 29.27 -> 29.20
CPU   iter 500 psnr 29.55 dB | dct50,q8 psnr 29.55 @ 4.570 bpp (ent 2.024, ctx 1.850) | dct nz 14.97 ...; reprobe 3.2% / 0.0%; done ... [level 0 alone: 1.361] | 42.3s
GPU   iter 500 psnr 29.54 dB | dct50,q8 psnr 29.54 @ 4.624 bpp (ent 2.040, ctx 1.862) | dct nz 15.18 ...; reprobe 3.9% / 0.0%; done ... [level 0 alone: 1.375] | 1.3s
```
Identical through the switch and the first refit's move figures; the code histogram differs by one
block from refit 1 on (275 / 3118 vs 274 / 3119: a block gain on a rounding boundary of the two
backends' 1e-7-level decode difference), and the runs agree to display precision at the end (0.01 dB,
0.012 bpp ctx), the section-3 pattern. The refit at 200 costs 1.9 dB on both backends (43% of the
codes move in one step at an unannealed learning rate) and is recovered within the next 50 iterations;
the refits after 350 change the PSNR by <= 0.2 dB. The frozen section-3 run of the same length ends at
29.98 dB @ 5.398 bpp raw (2.096 ctx, level 0 alone 1.608) with 65.3% of the codes stale; this run ends
at 29.54 dB @ 4.624 (1.862 ctx, level 0 alone 1.375) with 3.9% stale, i.e. 0.44 dB less at 11% fewer
context bits (a different `--dct-start`, so not a controlled pair; the 8000-iteration pairs below are).

v13 round trip of the refitted GPU model (`--load out_dct_r_gpu/model.bin --iters 0`, magic 3d43 544e):
CPU and `--cuda` both print `done: final psnr 29.54 dB ... 29.54 dB at 4.624 bpp raw, 2.040 bpp
entropy-coded, 1.862 bpp ... [level 0 alone: 1.375] [dct: raw 4.092 h0 1.554 ctx 1.375 ...]` (= the run's
line; the file carries the codes of refit 7: `dct cfg ... codes-fitted-at -1 refit-every 500 refit-until 0
refits 0`, reprobe 3.9% / 0.0% as at the end of the run). `ntc_decode out_dct_r_gpu/model.bin --compare
out_dct_r_gpu/recon_q_final.png --verify`: `level 0 : ... dct 8x8, q 50, dc step 4, 4-bit scale codes,
15.2 nonzero ACs/block`, codes k0..k15 `2 4 9 21 111 823 2253 866 7 ...` (= the run's final histogram),
raw 261.9 bits/blk = 4.0920 bpp (= the trainer's `[dct: raw 4.092`), idct 2.49 ms (10.0 MAC/texel, nnz
62186); compare PSNR 88.70 dB, max |diff| 1 (69 texels); verify pre-nonlinearity max 0 ulp (0 of 786432
differ), output fp32 max 42 ulp, RGB8 mismatches 69; rc 0.

### The 8000-iteration pairs (GPU, `build_cuda\Release\ntc.exe`)

`IMG --cuda --block 8 --latent 0 0 1 --latent2 0 0 C --filter nearest,bilinear --pos lv1local --dct-q 50
--qes 0,8 --mlp 27,27 --mlp-pairs 256 --iters 8000 --lr-anneal 0.5 0.05 --mlp-fd 0.5 --leak 0.0009765625
--dct-map all` with `--dct-refit 500` (`..._refit500_...`) and `--dct-refit 250 --dct-refit-until 0.9`
(`..._refit250u90_...`), against the frozen-code references of the sweep (same command without the
refit flags, built before this change). All runs share the switch line (image3 `psnr 44.26 -> 39.47 dB`,
histogram `55 621 551 1922 5533 9863 11877 2277 231`; model15 `37.00 -> 33.26 dB`, `52 9202 8971`).

| run | PSNR | bpp raw / h0 / ctx | level 0 alone | nz/blk | refits (moved >= 1 %) | reprobe >= 1 / >= 2 |
|---|---|---|---|---|---|---|
| image3 c4, frozen (`out_image3_b8_dct50_c4bilin_qes8_mlp27_cuda8k_fd50.log`) | 45.64 | 2.145 / 1.071 / 0.925 | 0.482 | 5.36 | none | 90.0% / 54.7% |
| image3 c4, `--dct-refit 500` | 45.24 | 1.458 / 0.794 / 0.717 | 0.274 | 2.61 | 7: 90.9 (>= 2: 48.1), 33.0, 17.3, 13.5, 11.7, 9.4, 6.3 | 5.6% / 0.0% |
| image3 c4, `--dct-refit 250 --dct-refit-until 0.9` | 45.19 | 1.491 / 0.807 / 0.728 | 0.285 | 2.74 | 12: 91.0 (>= 2: 48.6), 20.0, 25.1, 14.3, 19.6, 12.8, 17.3, 12.7, 11.2, 8.4, 14.9, 13.6 | 13.6% / 0.0% |
| model15 c3, frozen (`out_model15_b8_dct50_c3bilin_qes8_mlp27_cuda8k_fd50.log`) | 33.57 | 4.272 / 1.865 / 1.726 | 1.371 | 14.34 | none | 62.1% / 7.3% |
| model15 c3, `--dct-refit 500` | 32.55 | 3.941 / 1.676 / 1.551 | 1.197 | 13.02 | 7: 26.8, 28.1, 14.1, 8.9, 8.1, 5.8, 5.1 (>= 2 at most 0.2) | 2.3% / 0.0% |
| model15 c3, `--dct-refit 250 --dct-refit-until 0.9` | 32.49 | 3.949 / 1.680 / 1.554 | 1.200 | 13.05 | 12: 20.9, 8.5, 17.6, 13.8, 8.3, 5.5, 5.8, 5.6, 7.0, 4.2, 4.1, 4.0 | 5.8% / 0.0% |

PSNR before -> after each re-snap (image3, refit 500): 43.53 -> 43.14, 44.16 -> 43.39, 44.24 -> 44.06,
44.54 -> 43.48, 44.82 -> 44.67, 44.94 -> 44.43, 45.10 -> 44.97 dB; model15, refit 500: 32.96 -> 32.91,
32.68 -> 32.63, 32.56 -> 32.41, 32.52 -> 32.44, 32.48 -> 32.38, 32.45 -> 32.25, 32.47 -> 32.42 dB (the
full lines are in the logs). Timing: image3 84.8 / 79.5 s against the reference's 76.9 s; model15
51.5 / 52.1 s against 50.2 s (the host probe and the two host decodes per refit).

Reading. The refits move the codes down (coarser steps: image3's mode moves from k6 to k3-k4, model15's
from k7-k8 to k6-k7), because the trained decoder is less sensitive to the selector than the decoder
at the switch, so every refit run is a lower-rate, lower-PSNR point than its frozen reference rather
than the same point improved; the comparison has to be made against the frozen sweep curve. On image3
the refit-500 point (45.24 dB at 0.717 bpp ctx, level 0 alone 0.274) sits at the rate of the frozen q 25
reference (`out_image3_b8_dct25_c4bilin_...`: 44.41 dB at 0.727, level 0 alone 0.284) with 0.83 dB more
PSNR, and 0.40 dB below the frozen q 50 point at 22% fewer context bits. On model15 the refit-500 point
(32.55 dB at 1.551, level 0 alone 1.197) lies between the frozen q 25 (31.66 dB at 1.450, 1.099) and
q 50 (33.57 dB at 1.726, 1.371) references: 0.89 dB above q 25 for 0.101 bpp more, 1.02 dB below q 50
for 0.175 bpp less. The residual staleness at the end drops from 90.0% / 62.1% to 5.6% / 2.3% (refit
500) and the >= 2 movers to 0.0% on both images. The 250 / 0.9 variant is 0.05 / 0.06 dB below the
500 / 1.0 one at slightly more bits on both images and leaves more staleness, hence the defaults.
Open: the drop at each re-snap (up to 1.06 dB at image3's refit 4) is recovered but not free; a rate
term in the probe, or moving only the codes whose gain crossed the bin centre by a margin, are the
obvious follow-ups, and a matched-rate comparison needs the sweep re-run with the refits on.

### Refit review (September 7, 2026), applied

The review found the refit route correct (device shadow never written; byte-identical model.bin
with the flag off or `--dct-refit 0`; check 7's refit line exercising the upload) and three edge
cases, fixed: a run resumed from a v13 file no longer skips the refit that lands on the would-be
switch iteration (the exclusion applies only when the switch happens in this run);
`--dct-refit-until` at or below `--dct-start` prints a note and disables refits instead of a
misleading banner; the load-path line says "no re-snap at load; the codes refit on the run's
schedule" (or "codes frozen") instead of always "codes frozen". Resumed 20-iteration run with
`--dct-refit 10`: the refit at iteration 10 now fires.

## 7. Multi-channel level 0 (September 7, 2026)

DCT_MULTICHANNEL_PLAN.md sections 1-7 on top of tag `v0.10-ntcb-container` (commit fda17fe): `--latent 0 0 C` with
`--dct-q` for C = 1..4, each channel its own 8x8 DCT-coded plane with its own per-block 4-bit scale codes, symbol
stream and q (`--dct-q Q` = every channel, `--dct-q Q0,Q1[,Q2,Q3]` per channel, the `--qat B1,B2` convention). As the
plan's audit found, every array, kernel, file field and reader was `[block][channel]` already; the change is the option
syntax and validation (`Options::dct_q_ch`, `o.dct_q` = the maximum so every existing gate is untouched), the
`MAX_DCT_CH` refusal in place of the `o.LC != 1` refusal, `D.dct.q = o.dct_q_ch`, the per-channel statistics
(`dct_analyze(Q, only_c)`, `dct_plane_stats(..., only_c)`, `dct_print_codes(Q, S, c)` -> `dct codes[c]`, the `dct[c]` line
per channel after the summed block, the `(ch a% / b%)` clause of the refit line from `Decoder::dct_refit`'s `m1c`),
per-channel clamp counts (`DctLevel::last_clamped_c`, one `omp atomic` per (block, channel) in `dct_snap_level0` /
`dct_recon_level0`), `dct_side_by_side.png` = reconstruction | the map of every channel (W x (1 + C) wide), the `%s`
warm-start note and raw-bpp banner, the probe-cost clause of the `dct` banner, the channel count in the check-7 PASS line,
ntc_decode's `nnz_c` / `eob0_c` and its `dct      : per channel` line, the usage / README text. No CUDA source change
(`git diff --stat cuda/` is empty); the device already runs one thread per (block, channel) with `c_dct_step[c]`.
Diff (`git diff --ignore-cr-at-eol --numstat`): main.cpp 124 added / 47 removed, ntc_decode.cpp 8 / 1, README.md 27 / 11,
DCT_NOTES.md this section and the section-5 list; 0 added source lines without `[DCT]` (script over `git diff -U0`);
line endings unchanged (the `\r\r\n` of the `std::string load;` line, now line 172, put back byte for byte after the
Edit tool normalized it, as in section 4 item 11). Both builds 0 compiler warnings (the CUDA link's pre-existing
`LNK4098` only). Every figure below is from the final binaries; nothing is committed.

### Firewall (C == 1 byte-identical) and the regression pin

`--dct-selftest`: `dct-selftest: all passed` on both builds, the nine section-3 lines unchanged (CPU and CUDA output
`diff`-identical). Regression pin:
```
build\Release\ntc.exe --crop 512 --mlp-pairs 32 --iters 200 --print-every 200 --save-every 200 --out out_reg
iter    200  mse 269.314  psnr  23.83 dB  best  23.83 | q8 psnr  23.83 @ 0.552 bpp (ent 0.500, ctx 0.084) | mlp batch 0.00442 dstd 9.65e-05 | lat mean -0.051 sd 0.551 max 2.08 | 7.6s (26.19 it/s)
done: final psnr 23.83 dB (best 23.83) at fp32 2.103 bpp | 8-bit latent + fp16 mlp: psnr 23.83 dB at 0.552 bpp raw, 0.500 bpp entropy-coded, 0.084 bpp with an (up, left) context on level 0 [level 0 alone: 0.032] | 7.8s
sha256 out_reg/model.bin cb88470777e1d041c8803bcf3ca531434b9a2b2b5bce2ef88e031fbc86d69600 (= section 3)
ntc_decode out_reg/model.bin --compare --verify: PSNR 88.76 dB, max |diff| 1, 1: 68; pre-nonlinearity max 0 ulp (0 of 786432); RGB8 mismatches 68 (= NTCB_NOTES section 9)
```
The 1-channel v13 loads (`build_cuda\Release\ntc.exe ... --load <dir>/model.bin --iters 0`, the options of the file's log):
```
out_m1_b4_dct30_c4bilin_qes8_mlp27_refit500_cuda8k_fd50/model.bin:
dct bits raw dc 32768 run 172914 mag 197616 sign 24702 eob 28672 code 16384 hdr 96 total 473152 /blk 115.52 = 1.8049 bpp
dct reprobe (end of run, not applied) hist k0..k15 0 30 1100 1449 551 949 17 0 0 0 0 0 0 0 0 0 | codes that would move >= 1: 4.6%, >= 2: 0.0% (of 4096)
done: final psnr 26.08 dB (best -1.00) at fp32 40.129 bpp | --dct-q 30 level 0 (bit simulator) + --qes 0,8 latent + fp16 mlp: psnr 26.08 dB at 3.870 bpp raw, 2.443 bpp entropy-coded, 2.397 bpp with an (up, left) context on level 0 [level 0 alone: 0.548] [dct: raw 1.805 h0 0.594 ctx 0.548 code raw/ctx 0.062/0.021 bpp; nz 6.03/blk eob0 3.6%] | 0.2s
out_image3_b8_dct50_c4bilin_qes8_mlp27_refit500_cuda8k_fd50/model.bin:
dct bits raw dc 263440 run 602294 mag 688336 sign 86042 eob 230503 code 131720 hdr 96 total 2002431 /blk 60.81 = 0.9501 bpp
done: final psnr 45.24 dB (best -1.00) at fp32 34.016 bpp | --dct-q 50 level 0 (bit simulator) + --qes 0,8 latent + fp16 mlp: psnr 45.24 dB at 1.458 bpp raw, 0.794 bpp entropy-coded, 0.717 bpp with an (up, left) context on level 0 [level 0 alone: 0.274] [dct: raw 0.950 h0 0.351 ctx 0.274 code raw/ctx 0.062/0.011 bpp; nz 2.61/blk eob0 7.1%] | 1.3s
```
On both files the `dct nz` .. `dct reprobe` block and the `done:` line are `diff`-identical to the run's log apart from
`best -1.00` and the `dct plane rms 0.000 ... changed% 0.0` line (a `--load --iters 0` property: the shadow's level 0 is
set equal to the IDCT, section 1), i.e. the same lines the pre-change binary prints. A byte comparison of the whole
m1 load output with the September 7 container conversion of the same file (`conv_dct-b4.txt`, the review-fix binary)
differs in two lines only: the banner's new `; probe: 20 MLP evaluations per block and channel (0.31 decode-equivalents
per probe for 1 channel)` clause, and the reprobe line (`1449 551 ... 4.6%` here vs `1448 552 ... 4.7%` there), which is
`--write-ntcb`'s fp16 rounding of the MLP before the reprobe, not the change: today's binary with `--write-ntcb` prints
the conversion's `1448 552 ... 4.7%` again (the `dct-b4` matrix row below), and the plain load prints the log's `4.6%`.

The eight README `--cuda-check` lines (five general + `--dct-q 50 / 90 / 20`): every one `cuda-check: all passed`,
rc 0, with the section-3 / addendum figures (decode 1.788e-07 / 1.788e-07 / 1.192e-07 / 1.192e-07 / 1.192e-07; qat search
`3 of 524288` (worse 0, better 1, ties 1) / `3 of 786432` (worse 2) / `1 of 262144` (ties 1); qes snap 0 / 0 everywhere;
the DCT line's mlp ES 3.442e-07 / 3.746e-07 / 3.670e-07, FD 8.361e-05 / 7.959e-05 / 8.316e-05, `dct refit` moved
195 / 193 / 196 of 4096, dct snap 0 / 0 (max 0 ulp) / 0 before and after the step). The PASS line now names the count:
```
  dct snap        : symbols 0 of 262144 differ, plane 0 of 262144 values (1 channel) differ (max 0 ulp), codes 0 of 4096 differ (device plane from k_dct_recon on the host's symbols: IDCT + clamp parity); after a latent step with lr 0.02: 0 / 0 (max 0 ulp) / 0 (k_dct_snap: forward DCT + quantizer + IDCT parity)  PASS
```
The three C == 1 DCT rows of the NTCB matrix (`matrix_all.sh` rows dct50, dct-b6, dct-b4 through `matrix.sh`): every
step rc 0, `[OK]` + `self-check OK`, the ntcb and model.bin decoder PNGs `cmp` identical, reload done line equal; the
conversion, decode and reload outputs are `diff`-identical to the September 7 review-fix run's (NTCB_NOTES section 9)
apart from the banner's probe clause and ntc_decode's bench timing row:
```
file: out_ntcb_dct50/model.ntcb 565301 bytes = 2.146 bpp (simulator raw 2.145 bpp); content 4521126 bits = simulator 4521478 - 352 side-info bits (stored in the header) [OK]; container header 1040 + section headers 192 + sync 42 + padding 8 bits; bpp over the 1480x1424 decode size, as every bpp in this log; self-check OK
file: out_ntcb_dct-b6/model.ntcb 155248 bytes = 1.355 bpp (simulator raw 1.354 bpp); content 1240701 bits = simulator 1241053 - 352 side-info bits (stored in the header) [OK]; container header 1040 + section headers 192 + sync 42 + padding 9 bits; bpp over the 888x1032 decode size, as every bpp in this log; self-check OK
file: out_ntcb_dct-b4/model.ntcb 126941 bytes = 3.874 bpp (simulator raw 3.870 bpp); content 1014240 bits = simulator 1014592 - 352 side-info bits (stored in the header) [OK]; container header 1040 + section headers 192 + sync 42 + padding 14 bits; bpp over the 512x512 decode size, as every bpp in this log; self-check OK
```

### Refusals (each prints the line and exits 1)

`--dct-q 30,50,70` on `--latent 0 0 2`: `--dct-q lists 3 qualities but level 0 has 2 channels`; `--dct-q 30,x`:
`--dct-q: malformed quality list (use Q or Q0,Q1,... with Q = 1..100)`; `--dct-q 30,0`: `--dct-q: every channel needs
1..100 when the DCT plane is on`; `--dct-q 30,101`: `--dct-q needs 1..100 (0 = off)`; `--latent 0 0 5 --dct-q 30`:
`--dct-q: level 0 may have at most 4 channels`. `--dct-q 0,0` is the flag off (rc 0, no `dct` line, `8-bit latent` label).

### `--cuda-check` with 2 and 4 channels (plan 8.2)

`build_cuda\Release\ntc.exe m1.png --cuda --cuda-check --block 4 --latent 0 0 2 --latent2 0 0 4 --filter nearest,bilinear
--pos lv1local --dct-q 50 --dct-start 0 --qes 0,8 --qes-start 0 --mlp 27,27 --leak 0.0009765625 --iters 5 --print-every 5
--out out_chk`, the same with `--dct-q 50,20`, with `--latent 0 0 4 --dct-q 30,50,70,90`, and `chief1.png --block 8
--latent 0 0 2 --latent2 0 0 4 ... --dct-q 50 --mlp 17,17`: each `cuda-check: all passed`, rc 0, seven PASS lines.
```
m1 --latent 0 0 2 --dct-q 50:    decode 1.192e-07; mlp ES 2.992e-07; FD 3.670e-05; latent ES grad 2.095e-05 over 589824 values; qes snap 0 of 589824, 0 after the step
  dct snap        : symbols 0 of 524288 differ, plane 0 of 524288 values (2 channels) differ (max 0 ulp), codes 0 of 8192 differ (device plane from k_dct_recon on the host's symbols: IDCT + clamp parity); after a latent step with lr 0.02: 0 / 0 (max 0 ulp) / 0 (k_dct_snap: forward DCT + quantizer + IDCT parity)  PASS
  dct refit       : host re-probe moved 5 of 8192 codes (>= 2: 0), uploaded through set_dct: symbols 0 differ, plane 0 (max 0 ulp), codes 0 (k_dct_recon with the refitted codes); after a latent step: 0 / 0 (max 0 ulp) / 0 (k_dct_snap quantizes with the refitted codes)  PASS
m1 --latent 0 0 2 --dct-q 50,20: decode 1.192e-07; mlp ES 3.037e-07; FD 3.719e-05; latent 2.095e-05; the same dct snap line (0 / 0 (max 0 ulp) / 0 before and after, 2 channels); dct refit moved 5 of 8192, 0 / 0 / 0 before and after
m1 --latent 0 0 4 --dct-q 30,50,70,90: decode 1.192e-07; mlp ES 1.971e-07; FD 4.382e-05; latent 2.382e-05 over 1114112 values; qes snap 0 of 1114112
  dct snap        : symbols 0 of 1048576 differ, plane 0 of 1048576 values (4 channels) differ (max 0 ulp), codes 0 of 16384 differ (...); after a latent step with lr 0.02: 0 / 0 (max 0 ulp) / 0 (...)  PASS
  dct refit       : host re-probe moved 1931 of 16384 codes (>= 2: 0), uploaded through set_dct: symbols 0 differ, plane 0 (max 0 ulp), codes 0 (...); after a latent step: 0 / 0 (max 0 ulp) / 0 (...)  PASS
chief1 --block 8 --latent 0 0 2 --dct-q 50: decode 1.192e-07; mlp ES 4.796e-07; FD 1.398e-04; latent 4.184e-05 over 540672 values; dct snap 0 of 524288 / 0 of 524288 (2 channels) / 0 of 8192, 0 / 0 / 0 after the step; dct refit moved 974 of 8192, 0 / 0 / 0 before and after  PASS
```
The switch at iteration 0 on the 4-channel line: `histogram k0..k15 0 0 5119 7169 4096 0 ...` (the four q give the four
channels different tables: 16 KB of `c_dct_step`); with `--dct-q 50,20` the two channels' tables differ, so a `[c]` /
`[0]` mix-up in either mirror would have shown as symbol differences.

### The 2-channel training runs (plan 8.3)

`build_cuda\Release\ntc.exe m1.png --cuda --block 4 --latent 0 0 2 --latent2 0 0 4 --filter nearest,bilinear --pos lv1local
--dct-q 30 --dct-refit 500 --qes 0,8 --mlp 27,27 --mlp-pairs 256 --iters 8000 --lr-anneal 0.5 0.05 --mlp-fd 0.5 --leak
0.0009765625 --dct-map all --out out_m1_b4_dct30x2_c4bilin_qes8_mlp27_refit500_cuda8k_fd50` (28.4 s) and the `--dct-q 30,50`
variant (`out_m1_b4_dct30x50_...`, 28.5 s), against the 1-channel `out_m1_b4_dct30_c4bilin_qes8_mlp27_refit500_cuda8k_fd50.log`
(26.08 dB at 3.870 / 2.443 / 2.397 bpp, level 0 alone 0.548, nz 6.03/blk, switch 30.24 -> 23.82 dB, reprobe 4.6%).

| run | PSNR | bpp raw / h0 / ctx | level 0 alone | [dct: ...] | switch psnr | refits moved >= 1 (ch 0 / ch 1) | reprobe |
|---|---|---|---|---|---|---|---|
| 1 channel, q 30 (reference log) | 26.08 | 3.870 / 2.443 / 2.397 | 0.548 | raw 1.805 h0 0.594 ctx 0.548 code raw/ctx 0.062/0.021; nz 6.03/blk eob0 3.6% | 30.24 -> 23.82 | 100.0, 43.0, 13.5, 9.3, 9.1, 6.6, 5.7 | 4.6% / 0.0% |
| 2 channels, q 30,30 | 26.69 | 5.064 / 2.894 / 2.813 | 0.944 | raw 2.996 h0 1.024 ctx 0.944 code raw/ctx 0.125/0.049; nz 4.80/blk eob0 13.5% | 36.88 -> 24.71 | 99.7 (99.3 / 100.0), 24.4 (22.7 / 26.1), 16.9 (17.3 / 16.5), 11.8 (11.7 / 11.8), 12.0 (11.1 / 12.8), 9.8 (10.1 / 9.4), 7.3 (7.6 / 7.1) | 5.7% / 0.0% |
| 2 channels, q 30,50 | 27.08 | 6.074 / 3.224 / 3.109 | 1.240 | raw 4.006 h0 1.355 ctx 1.240 code raw/ctx 0.125/0.047; nz 6.82/blk eob0 8.8% | 36.88 -> 25.44 | 99.9 (99.8 / 100.0), 22.7 (21.3 / 24.1), 14.8 (16.5 / 13.1), 12.0 (13.1 / 10.9), 10.1 (10.9 / 9.4), 9.4 (11.0 / 7.7), 7.4 (8.1 / 6.7) | 5.8% / 0.0% |

The done lines and the per-channel lines of the final block:
```
done: final psnr 26.69 dB (best 36.51) at fp32 72.132 bpp | --dct-q 30,30 level 0 (bit simulator) + --qes 0,8 latent + fp16 mlp: psnr 26.69 dB at 5.064 bpp raw, 2.894 bpp entropy-coded, 2.813 bpp with an (up, left) context on level 0 [level 0 alone: 0.944] [dct: raw 2.996 h0 1.024 ctx 0.944 code raw/ctx 0.125/0.049 bpp; nz 4.80/blk eob0 13.5%] | 28.4s
dct bits raw dc 65536 run 275534 mag 314896 sign 39362 eob 57344 code 32768 hdr 192 total 785632 /blk 191.80 = 2.9969 bpp
dct bits h0  dc 40509 run 102750 mag 46496 sign 39362 eob 20781 code 18470 hdr 192 total 268560 /blk 65.57 = 1.0245 bpp
dct bits ctx dc 40509 run 93106 mag 44059 sign 39362 eob 17414 code 12905 hdr 192 total 247547 /blk 60.44 = 0.9443 bpp
dct plane rms 27.906 max 64.00 (units of 64) clamped% 2.42 changed% 98.4
dct[0] q 30 nz mean 4.64 eob0 15.5% lnz 13.2 codes k0..k15 16 711 1395 728 477 758 11 0 0 0 0 0 0 0 0 0 mean_k 2.80 | bits raw 382064 h0 131608 ctx 121353 (/blk 93.28/32.13/29.63 = 1.4575/0.5020/0.4629 bpp) | dc range 28..81 hres 4.99 | plane rms 28.035 max 64.00 clamped% 2.43 changed% 98.4
dct[1] q 30 nz mean 4.97 eob0 11.5% lnz 14.2 codes k0..k15 3 450 1275 1078 554 732 4 0 0 0 0 0 0 0 0 0 mean_k 2.96 | bits raw 403568 h0 136952 ctx 126194 (/blk 98.53/33.44/30.81 = 1.5395/0.5224/0.4814 bpp) | dc range 41..84 hres 4.90 | plane rms 27.777 max 64.00 clamped% 2.42 changed% 98.4
dct reprobe (end of run, not applied) hist k0..k15 17 1195 2660 1789 1040 1473 18 0 0 0 0 0 0 0 0 0 | codes that would move >= 1: 5.7%, >= 2: 0.0% (of 8192)

done: final psnr 27.08 dB (best 36.89) at fp32 72.132 bpp | --dct-q 30,50 level 0 (bit simulator) + --qes 0,8 latent + fp16 mlp: psnr 27.08 dB at 6.074 bpp raw, 3.224 bpp entropy-coded, 3.109 bpp with an (up, left) context on level 0 [level 0 alone: 1.240] [dct: raw 4.006 h0 1.355 ctx 1.240 code raw/ctx 0.125/0.047 bpp; nz 6.82/blk eob0 8.8%] | 28.5s
dct bits raw dc 65536 run 391314 mag 447216 sign 55902 eob 57337 code 32768 hdr 192 total 1050265 /blk 256.41 = 4.0064 bpp
dct bits h0  dc 40474 run 133638 mag 83479 sign 55902 eob 23759 code 17966 hdr 192 total 355409 /blk 86.77 = 1.3558 bpp
dct bits ctx dc 40474 run 119703 mag 78672 sign 55902 eob 17979 code 12332 hdr 192 total 325254 /blk 79.41 = 1.2407 bpp
dct[0] q 30 nz mean 4.49 eob0 16.3% lnz 13.1 codes k0..k15 27 706 1474 688 491 707 3 0 0 0 0 0 0 0 0 0 mean_k 2.74 | bits raw 371952 h0 128569 ctx 118476 (/blk 90.81/31.39/28.92 = 1.4189/0.4905/0.4519 bpp) | dc range 22..88 hres 4.97 | plane rms 28.106 max 64.00 clamped% 2.37 changed% 98.4
dct[1] q 50 nz mean 9.16 eob0 1.3% lnz 21.1 codes k0..k15 0 215 1226 1307 574 772 2 0 0 0 0 0 0 0 0 0 mean_k 3.11 | bits raw 678313 h0 226840 ctx 206778 (/blk 165.60/55.38/50.48 = 2.5876/0.8653/0.7888 bpp) | dc range 42..88 hres 4.91 | plane rms 25.812 max 64.00 clamped% 3.83 changed% 97.5
```
Hand check of the per-channel sums against the summed lines (the section-4 requirement): q 30,30: raw 382064 + 403568 =
785632, h0 131608 + 136952 = 268560, ctx 121353 + 126194 = 247547; q 30,50: raw 371952 + 678313 = 1050265, h0 128569 +
226840 = 355409, ctx 118476 + 206778 = 325254; codes 16 + 3 = 19, 711 + 450 = 1161, ... = the summed `dct codes[c]`
histograms' sum = the reprobe-line population 8192; `nz mean` (4.64 + 4.97) / 2 = 4.805 -> 4.80; `clamped%` (2.43 + 2.42) / 2
-> 2.42. The switch and refit lines (q 30,30; the q 30,50 run's are in its log):
```
iter   4000  dct: scale codes fitted from the decoder sensitivity probe (frozen), level 0 decodes from the DCT-snapped plane from here; psnr 36.88 -> 24.71 dB; nz 9.27/blk, eob-only 0.1%, clamped 1.89%; histogram k0..k15 0 0 0 2 297 3739 3100 1052 2 0 0 0 0 0 0 0
dct codes[0] k0 0 k1 0 k2 0 k3 2 k4 279 k5 2610 k6 1173 k7 32 k8 0 ... mean_k 5.23 | step1 by k 84 55 36 24 16 10 7 5 3 2 1 1 1 1 1 1 | step63 by k 754 498 328 217 143 94 62 41 27 18 12 8 5 3 2 1 | all-ones 0.0% dc-only-by-table 0.0%
dct codes[1] k0 0 k1 0 k2 0 k3 0 k4 18 k5 1129 k6 1927 k7 1020 k8 2 k9 0 ... mean_k 5.97 | (the same tables: both channels at q 30)
iter   4500  dct: codes refitted (1); moved >= 1: 99.7%, >= 2: 83.8% (of 8192) (ch 99.3% / 100.0%); psnr 27.26 -> 26.06 dB; nz 4.41/blk, eob-only 11.5%, clamped 1.26%; histogram k0..k15 0 100 2254 2910 1052 1799 77 0 ...
iter   5000 (2) 24.4% (ch 22.7% / 26.1%) 26.39 -> 26.24 | 5500 (3) 16.9% (17.3 / 16.5) 26.48 -> 26.38 | 6000 (4) 11.8% (11.7 / 11.8) 26.55 -> 26.49 | 6500 (5) 12.0% (11.1 / 12.8) 26.63 -> 26.56 | 7000 (6) 9.8% (10.1 / 9.4) 26.64 -> 26.60 | 7500 (7) 7.3% (7.6 / 7.1) 26.64 -> 26.61
q 30,50: iter 4000 psnr 36.88 -> 25.44 dB (the same code histogram: the probe does not see q); dct codes[1] step1 by k 50 33 22 14 10 6 4 3 2 1 1 1 1 1 1 1 | step63 by k 453 299 197 130 86 57 37 25 16 11 7 5 3 2 1 1 (the q 50 tables)
q 30,50 refits: 4500 (1) 99.9% (>= 2: 81.7%) (ch 99.8 / 100.0) 27.63 -> 26.30 | 5000 (2) 22.7% (21.3 / 24.1) 26.72 -> 26.56 | 5500 (3) 14.8% (16.5 / 13.1) 26.84 -> 26.75 | 6000 (4) 12.0% (13.1 / 10.9) 26.92 -> 26.84 | 6500 (5) 10.1% (10.9 / 9.4) 26.97 -> 26.91 | 7000 (6) 9.4% (11.0 / 7.7) 27.03 -> 26.96 | 7500 (7) 7.4% (8.1 / 6.7) 27.03 -> 26.98
```
Reading. The trainer uses the second channel: in the q 30,30 run channel 1 has nz 4.97/blk against channel 0's 4.64,
dc range 41..84 against 28..81, plane rms 27.8 against 28.0, a code histogram as populated as channel 0's (`3 450 1275
1078 554 732 4` vs `16 711 1395 728 477 758 11`) and 51% of the level's context bits (126194 of 247547); `latent_q_008000.png` (1024 x 512,
the two snapped planes side by side) shows the stone structure in both halves by eye; the numeric backing is the per-channel plane rms / dc range / changed% above, and the decoder-sensitivity proxy `mean_k` per channel, which says the decoder listens to channel 1 rather than merely that it carries bits: 2.96 (ch 1) vs 2.80 (ch 0) at the end of the q 30,30 run, 5.97 vs 5.23 at its switch, 3.11 vs 2.74 in the q 30,50 run (from the `dct codes[c]` lines); the per-channel level-0 ctx rates are 0.4629 / 0.4814 bpp (q 30,30) and 0.4519 / 0.7888 bpp (q 30,50), the rate the second plane costs;
`dct_side_by_side.png` is 1536 x 512 (reconstruction | map of channel 0 | map of channel 1; the 1-channel run's is
1024 x 512, unchanged). The refit percentages track each other per channel (the first refit moves nearly every code on
both, as in the 1-channel run's 100.0%), and the drop at each re-snap is 1.20 / 1.33 dB at refit 1 (q 30,30 / q 30,50) and
0.03-0.16 dB after that, the 1-channel run's pattern (0.86 dB at refit 1, then <= 0.19). The switch costs more (36.88 -> 24.71 / 25.44 dB against
30.24 -> 23.82): the float 2-channel model is 6.6 dB better before the switch and lands 0.9 / 1.6 dB higher after it.
Against the 1-channel point (26.08 dB at 2.443 bpp order-0, level 0 alone 0.548): q 30,30 gives +0.61 dB at 2.894 bpp
(+0.451 bpp, level 0 alone +0.396) and q 30,50 gives +1.00 dB at 3.224 bpp (+0.781, level 0 alone +0.692); neither is a
matched-rate comparison (the plan's section 9: the fair comparisons are a same-layout `--latent 0 0 2` float / `--qes`
run and the 1-channel curve at the same rate, e.g. `--dct-q 50` alone, not run here). Nothing was tuned.

### `--load --iters 0`, refusals, decoder, container on the 2-channel model (plan 8.4-8.6)

`--load out_m1_b4_dct30x2_.../model.bin --iters 0` with the run's options, CPU (`build\Release\ntc.exe`) and `--cuda`: both
print the run's `done:` line (`26.69 dB at 5.064 bpp raw, 2.894 bpp entropy-coded, 2.813 bpp ... [level 0 alone: 0.944]
[dct: raw 2.996 h0 1.024 ctx 0.944 code raw/ctx 0.125/0.049 bpp; nz 4.80/blk eob0 13.5%]`, `best -1.00`) and the run's
`dct nz` .. `dct reprobe` block including both `dct[c]` lines (`diff`-identical apart from the two `plane rms 0.000 ...
changed% 0.0` figures of a load, as for C = 1); `dct cfg q 30,30 N 8 dc_step 4 blocks 4096 (64x64) ch 2 codes-fitted-at -1
...`. Refusals (rc 1, the same message): `--latent 0 0 1 --dct-q 30`, `--latent 0 0 2 --dct-q 30,50`, and without `--dct-q`:
```
out_m1_b4_dct30x2_c4bilin_qes8_mlp27_refit500_cuda8k_fd50/model.bin was saved with --latent 512 512 2 --latent2 128 128 4 --filter nearest,bilinear --leak 0.000977 --qes 0,8 --dct-q 30,30 --dct-dc-step 4 --mlp 27,27 --pos lv1local and 1 texture(s); pass the same options
```
and the q 30,50 file loaded with `--dct-q 30,30`: `... --dct-q 30,50 --dct-dc-step 4 ...; pass the same options`, rc 1.
```
build\Release\ntc_decode.exe out_m1_b4_dct30x2_.../model.bin --compare out_m1_b4_dct30x2_.../recon_q_final.png --verify
model    : out_m1_b4_dct30x2_c4bilin_qes8_mlp27_refit500_cuda8k_fd50/model.bin (v13)
level 0  : 512x512x2 nearest, dct 8x8, q 30,30, dc step 4, 4-bit scale codes, 4.8 nonzero ACs/block, 13.5% EOB-only, fp32 planar
level 1  : 128x128x4 bilinear, qes 8-bit, range [-5.296, 3.629] [-7.974, 6.497] [-5.848, 7.258] [-4.496, 5.473]
dct      : nz mean 4.80/blk, eob0 13.5%, lnz mean 13.7, clamped 2.42%, codes k0..k15 19 1161 2670 1806 1031 1490 15 0 0 0 0 0 0 0 0 0; raw 95.9 bits/blk = 2.9962 bpp (fixed-length simulator: 8 DC + 16 per nonzero + 7 per EOB + 4 code bits)
dct      : per channel [0] q 30 nz 4.64/blk eob0 15.5% nnz 19009; [1] q 30 nz 4.97/blk eob0 11.5% nnz 20353
idct     : 3.43 ms single-threaded (8192 blocks; 8.7 MAC/texel with zero-skipping, 16 dense; nnz 39362)
compare t0: out_m1_b4_dct30x2_c4bilin_qes8_mlp27_refit500_cuda8k_fd50/recon_q_final.png (512x512): PSNR 89.53 dB, max |diff| 1, |diff| histogram: 0: 786375 (99.9928%), 1: 57 (0.0072%), 2: 0 (0.0000%), >2: 0 (0.0000%)
verify   : pre-nonlinearity: max 0 ulp (0 of 786432 values differ); output fp32: max 41 ulp, max |diff| 7.75e-07; RGB8 mismatches: 56 of 786432 (0.00712%)
--q8 and --fp32-latent PNGs: cmp identical to the default output; rc 0
```
(`nnz 19009 + 20353 = 39362` = the idct line's nnz; the decoder's `nz .../blk` is per (block, channel) = the trainer's.)
Container round trip (`matrix.sh` row `dct30x2`: `--load model.bin --iters 0 --write-ntcb --resave --out out_ntcb_dct30x2`,
`ntc_decode` on the ntcb and on the resaved model.bin, `cmp`, then `--load model.ntcb --iters 0`):
```
ntcb     : mlp weights rounded to fp16 (1083 of 1083 changed, max |dw| 4.839e-04); psnr 26.69 -> 26.69 dB; figures from here on are post-rounding
resave   : wrote out_ntcb_dct30x2/model.bin as v13 (level 0 DCT-coded, q 30,30, DC step 4; symbols and codes from the loaded v13 file)
file: out_ntcb_dct30x2/model.ntcb 166045 bytes = 5.067 bpp (simulator raw 5.064 bpp); content 1327056 bits = simulator 1327504 - 448 side-info bits (stored in the header) [OK]; container header 1056 + section headers 192 + sync 48 + padding 8 bits; bpp over the 512x512 decode size, as every bpp in this log; self-check OK
ntc_decode out_ntcb_dct30x2/model.ntcb: model : out_ntcb_dct30x2/model.ntcb (ntcb v1); level 0 : 512x512x2 nearest, dct 8x8, q 30,30, ...; file : 166045 bytes = 5.067 bpp (simulator raw 5.064 bpp: 1309728 level bits + 448 side-info (stored in the header) + 17328 mlp bits; content 1327056 bits matches the simulator exactly; container header 1056 + section headers 192 + sync 48 + padding 8 bits; ...)
compare t0: PSNR 90.01 dB, max |diff| 1, 1: 51 (0.0065%); verify: pre-nonlinearity max 0 ulp (0 of 786432 values differ); RGB8 mismatches 53   (the same two lines on the resaved model.bin); cmp n_dct30x2.png b_dct30x2.png: identical
reload (--load out_ntcb_dct30x2/model.ntcb --iters 0, --cuda): done: final psnr 26.69 dB (best -1.00) ... psnr 26.69 dB at 5.064 bpp raw, 2.894 bpp entropy-coded, 2.813 bpp ... [level 0 alone: 0.944] [dct: raw 2.996 h0 1.024 ctx 0.944 code raw/ctx 0.125/0.049 bpp; nz 4.80/blk eob0 13.5%]   RELOAD QUANTIZED DONE-LINE EQUAL: yes
```
Side info 448 = 2 x 96 (the two DCT channels' headers) + 4 x 64 (level 1's lo / hi), as the plan's section 6 (c) asks;
`sync 48` = the level / channel markers of a two-level file with two DCT channels (36 + 2 x 6). The q 30,50 model's row
(`dct30x50`): `file: out_ntcb_dct30x50/model.ntcb 199125 bytes = 6.077 bpp (simulator raw 6.074 bpp); content 1591689 bits =
simulator 1592137 - 448 side-info bits (stored in the header) [OK]; ... sync 48 + padding 15 bits; ... self-check OK`,
`level 0 : 512x512x2 nearest, dct 8x8, q 30,50, ...`, PNGs `cmp: identical`, reload done line equal, every rc 0.

### Deviations from the plan, with reasons

1. The per-channel `clamped%` of the `dct[c]` line needed a per-channel clamp count that the level did not keep
   (`last_clamped` is one total); `DctLevel::last_clamped_c[MAX_DCT_CH]` is filled by one `#pragma omp atomic` per
   (block, channel) inside the existing `reduction(+:...)` loops (integer sums, order-independent; the C = 1 figures are
   byte-identical, above). The `dct[c]` line also prints the plane `max` and `changed%` next to the rms (the same three
   figures as the summed `dct plane` line), which the plan's format did not list.
2. The refit line's per-channel clause sits after `(of N)`: `moved >= 1: 24.4%, >= 2: 0.0% (of 8192) (ch 22.7% / 26.1%)`
   (the plan wrote it directly after the first percentage); each figure is that channel's movers over that channel's codes.
3. Two helpers the plan did not name: `dct_print_codes_all` (the summed `dct codes` line when C == 1, one `dct codes[c]`
   line per channel from `dct_analyze(Q, c)` otherwise) and `dct_print_channel` (the `dct[c]` line), so the switch line,
   the refit line and the final block share one implementation; `dct_analyze`'s (block, channel) vectors use a `bi()`
   index (`b * C + c` summed, `b` per channel) instead of resizing, so the summed path is the old code.
4. The banner's probe figure is printed as `probe: 20 MLP evaluations per block and channel (0.62 decode-equivalents
   per probe for 2 channels)` (the plan's `%d MLP evaluations` with the constant 20 written out); it is the only C = 1
   banner text that changed (the `bitrate` and `dct` lines print the list through `dct_spec`, which is `30` for C = 1).
5. ntc_decode's per-channel line is `dct      : per channel [0] q 30 nz 4.64/blk eob0 15.5% nnz 19009; [1] ...` (the
   plan's "per-channel nz" line with the q and the raw nonzero count added so it can be checked against the idct line).
6. The README's seventh `--cuda-check` line is the m1 `--latent 0 0 2 --dct-q 50,20` line (distinct tables per channel,
   the check that would catch a channel mix-up), with `--dct-q 50` and the 4-channel line named in its comment.
7. The `--lat-pairs 8` variant of plan 8.3 was not run. Channel 1 is not near-flat (above), but the plan's other predicted signature of ES on two planes is in the logs: the post-switch recovery is slower (+2.26 dB from the switch to the end with one channel, 23.82 -> 26.08; +1.98 and +1.64 dB with two, 24.71 -> 26.69 and 25.44 -> 27.08, over the same 4000 annealed iterations). That is confounded with being nearer the ceiling; `--lat-pairs 8` on the q 30,30 layout is the run that would separate the two, still to do.
8. cuda/ntc_cuda.cu and cuda/ntc_cuda.h are untouched (the plan allowed 0-4 comment lines).
### Review fixes (September 7, 2026), applied

Two Fable 5.1 reviews (correctness / parity / firewall; reporting / usability / experimental reading) found no
correctness defect. Applied: (1) the `dct` banner's probe clause and the `(N channels)` wording of check 7's PASS line
print only when C > 1, so the C == 1 output is literally unchanged; (2) an all-zero `--dct-q` list of any length
(`0,0`) means off (the list is cleared before the length check); (3) the per-channel statistics recompute the
simulator 2C + 1 times per print (cost only, left as is); (4) `/blk` wording: the `dct bits` lines and the progress line
are per block position summed over channels, `dct dc`, `dct blkbits` and ntc_decode's line are per (block, channel), and
ntc_decode now labels its figure `bits/(blk,ch)` when C > 1; (5) stale one-channel text in the full-resolution refusal,
the `DctLevel` comment, this file's sections 1-3 and the README's check-7 sentence; (6) the refit clause prints its
denominator: `(ch 22.7% / 26.1% of 4096 each)`; (7) the switch and refit lines print the probe's mean gain per channel
when C > 1 (`(mean gain ch 0.77 / 0.85)`), and the reading above cites `mean_k` per channel and the per-channel level-0
ctx rates; (8) the "stone structure" sentence is marked as by eye with the numeric backing named; (9) the removal list
names the comment lines and the help block; (10) one list notation (`Q | Q0,Q1[,Q2,Q3]`) across `--qat`, `--qes` and
`--dct-q`, and the help says the DC step is shared by the channels. (The fix agent applying these was cut off by a usage
limit after the code edits and builds; the remaining edits and the verification below were done by hand.)

Verification on the rebuilt binaries (both builds 0 warnings):
- Regression pin: `done: final psnr 23.83 dB (best 23.83) at fp32 2.103 bpp`, `out_reg/model.bin` sha256 `cb88470777e1d041...` (unchanged).
- The README `--cuda-check` lines (five general, `--dct-q 50`, the 2-channel `--dct-q 50,20` line) plus the `--dct-q 90` and `20` variants: nine times `cuda-check: all passed`.
- Firewall: `--load out_m1_b4_dct30_c4bilin_qes8_mlp27_refit500_cuda8k_fd50/model.bin --iters 0` with the run's options prints every `dct nz / nzhist / lnz / run / dc / pnz / codes / blkbits / bits` line and the `done:` figures identical to the log (the only diffs are the load path's own: the restore note, `codes-fitted-at -1`, `plane rms 0.000` with no shadow in the file, `best -1.00`); no `probe:` clause is printed at C == 1.
- 2-channel check line output: `iter 0 dct: scale codes fitted ... (mean gain ch 0.77 / 0.85); psnr 11.50 -> 11.48 dB; nz 0.02/blk ...`, `dct snap : symbols 0 of 524288 differ, plane 0 of 524288 values (2 channels) differ (max 0 ulp), codes 0 of 8192 differ ... PASS`, `dct refit : host re-probe moved 5 of 8192 codes (>= 2: 0) ... PASS`, `cuda-check: all passed`.
- `--dct-q 0,0` on `--latent 0 0 2`: no `dct` banner, `done: ... --qes 0,8 latent + fp16 mlp: psnr 11.70 dB at 18.068 bpp raw` (the plane is off, the 2-channel level 0 is post-hoc 8-bit).


## 8. Rate proxy in the loss (September 7, 2026)

DCT_RATE_PLAN.md sections 1-10 on top of commit a29d987 (tags `v0.10-ntcb-container`, `v0.11-dct-multichannel`): `--dct-lambda L`
(0 = off) adds (A) a rate term to the latent ES objective, the integer Exp-Golomb token cost (plan 2.2) of the symbols of the
*perturbed shadow* (plan 2.3), attributed to the 64 texels of the (block, channel), and (B) deterministic truncation at snap time with
`lambda_t(k) = L / g_k^2` (plan 2.4 / 2.5), identical in `dct_snap_block` and `k_dct_snap`; `--dct-rate es | trunc | both`. The plan's
"decisions to confirm" are adopted as written: the term enters only the latent ES, it applies once `dct.live`, lambda is not stored in
the file, the exact distortion `c^2 - (c - v)^2` is used. ntc_decode.cpp, the v13 file and the NTCB container are untouched.
Diff (`git diff --ignore-cr-at-eol --numstat`): main.cpp 305 added / 26 removed, cuda/ntc_cuda.cu 130 / 5, cuda/ntc_cuda.h 8 / 0,
README.md 36 / 1, DCT_NOTES.md the section-5 bullet and this section (after the review fixes at the end of this section: main.cpp 323 / 26,
cuda/ntc_cuda.cu 131 / 5, cuda/ntc_cuda.h 8 / 0, README.md 45 / 1); 0 added source lines without `[DCT]` (script over `git diff -U0`);
line endings unchanged (the `\r\r\n` of the `std::string load;` line put back byte for byte after the Edit tool normalized it, as in
section 4 item 11; the final binaries were built before that one-byte comment-line restore). Both builds 0 compiler warnings (the CUDA
link's pre-existing `LNK4098` only). Every figure below is from program output; the experiments and the 500-iteration pair were run by
a build of the same source before the last edit (the host re-snap in check 4b of `cuda_check`, not on the training path), and the final
binary reproduces the lambda-80 m1 experiment byte for byte (`cmp` of `model.bin` and `model.ntcb`: identical) and the firewall run below.

### What was built

- `Options::dct_lambda`, `dct_rate`, `dct_rate_given`; `--dct-lambda F`, `--dct-rate S` (both set `dct_opts_given`); refusals
  `--dct-lambda -1` -> `--dct-lambda needs L >= 0 (0 = off)` (rc 1), `--dct-rate foo` -> `--dct-rate: es | trunc | both` (rc 1);
  notes `--dct-rate es` without lambda -> `note: --dct-rate es has no effect without --dct-lambda L > 0` (rc 0), `--dct-lambda 80` without
  `--dct-q` -> the existing "ignored without --dct-q" note, which now lists `--dct-lambda / --dct-rate` (rc 0).
- `DctLevel`: `lambda`, `rate_es`, `rate_trunc`, `lam_t16[16]` (= `L / (16 g_k^2)` in double, rounded once), `bits` ([block][channel]
  proxy bits in 1/16 bit), `last_truncated`, `last_truncated_c[4]`, `last_nz_before`, `rate_hits`, `rate_evals`.
- Cost model (strict FP block, integer): `dct_c_run(r) = 16 (2 floor_log2(r + 1) + 1)`, `dct_c_mag(m)` the same on `m = min(256, |q|)`,
  sign 16, EOB 32 (absent when the last nonzero is at zigzag 63); `dct_block_bits(sym)` = the zigzag walk of `dct_analyze` pass 2 with
  these costs. `dct_truncate_block(Cf, st, lam16, sym)`: plan 2.4 verbatim (`saved` with the run merge and the EOB that appears when
  zigzag 63 empties; `dD = c * c - (c - v) * (c - v)`, contraction off; `if (dD < lam16 * (float)saved)`). `dct_quantize_block(Cf, st,
  dc_step, lam16, with_bits, sym, ntrunc, nz_before)`: the old one-line quantize loop, then the truncation when `lam16 > 0`, returning
  the bits only when `with_bits` (the ES evaluations); `dct_snap_block` calls it and `dct_recon_block` fills `Q.bits` when `lambda > 0`
  (so `--load --iters 0` prints the `dct rate` line for the file's symbols). `dct_rate_block`: gather `clamp(z + sign * (sg * eps))` from
  the shadow, `dct_fwd8`, `dct_quantize_block` with the block's `lam_t16[code]`, the bits. `dct_snap_level0` reduces `ntrunc` and
  `nz_before` (integer, order-independent) into `last_truncated`, `last_truncated_c` and `last_nz_before`.
- Part A on the host: `dct_rate_pass(D, eps, grad, dbits, K, sg, hits)` (`NTC_NOINLINE`), the per-pair rate pass: `dbits[b][c]` =
  `dct_rate_block(+1) - dct_rate_block(-1)` (OpenMP over blocks), `grad[i] += rate_scale * dbits * scale * eps[i]` for the block's
  texels, `rate_scale = L / (16 * 65025 * npix)`, `scale = 1 / (2 K sg)`; `hits` counts the nonzero differences.
  `LatentTrainer::step` became a one-line dispatcher on `step_impl<RATE>`: `RATE = false` is the pre-change function text (the pass is an
  `if constexpr`), `RATE = true` calls the pass after the mse scatter of every pair (deviation 1 below says why).
- CUDA: `DctDesc` gains `rate_es, rate_trunc, rate_scale, lam_t16[16]` (filled by `dct_desc_of`, the one helper the three `set_dct`
  call sites now share); `ModelDesc::dct_zigzag` uploads `DCT_ZIGZAG` into `c_dct_zigzag`; `dev_dct_c_run / c_mag / block_bits /
  truncate_block / quantize_block` mirror the host (`dD = __fsub_rn(__fmul_rn(c, c), __fmul_rn(__fsub_rn(c, v), __fsub_rn(c, v)))`,
  threshold `__fmul_rn(lam16, (float)saved)`); `k_dct_snap` calls `dev_dct_quantize_block` and, when a rate flag is set, writes `bits[i]`
  and `atomicAdd`s the two counters; `k_dct_recon` writes `bits[i]`; `k_dct_rate_pair` (one thread per (block, channel), `s =
  clamp(__fadd_rn / __fsub_rn(z[idx], __fmul_rn(sg, ntc_gauss(seed, NS_LAT, step, pair, z_off + off + idx))))`, two forward DCTs, `dbits =
  bp - bm`, `atomicAdd(hits)`) and `k_dct_rate_gather` (one thread per level-0 texel) run inside `lat_step`'s pair loop after
  `k_lat_gather` when `I.dct_live && I.dct_rate_es`; `d_dct_bits`, `d_dct_dbits`, `d_dct_cnt[4]` (ntrunc, nz_before, hits, evals);
  `download_dct_rate(bits, dbits_last_pair, ntrunc, nz_before, hits, evals)`. With lambda 0 `lam_t16` is all zero, the flags are 0, no
  new kernel is launched and `k_dct_snap` runs the same `_rn` instructions plus a not-taken branch.
- `--cuda-check`: check 4's host estimate gains the rate pass through `cuda_check_rate_replica` (called once after the mse loop, the
  hash noise regenerated per pair); a `dct rate` line follows check 4 when part A is on: the device's pair K-1 `dbits` and hit count
  against the replica's, and its snapped-plane proxy bits and truncation counts against a host re-snap of the downloaded shadow (as
  check 7 does; deviation 2); the `dct snap` / `dct refit` lines gain `; proxy bits N of M differ` only when lambda > 0 (the `chs` pattern).
- Prints (all gated on `lambda > 0`; the formats are the ones after the review fixes at the end of this section, the quotes in the
  verification subsections above them are the pre-fix formats `proxy P`, `trunc N hits X%`, `trunc N of M nonzero`): the progress line's
  `| dct ...` group gains ` (pre-trunc X)` after `nz` (the nonzero ACs per (block, channel) before the last snap's truncation), ` proxy P
  (Rx h0ac)` after `bits/blk` (proxy bits per block position and the ratio to the h0 AC tokens) and ` trunc N/M hits X%` at its end (N
  truncated of M pre-truncation nonzeros); a part left out by `--dct-rate` prints `trunc off` / `hits off`; the switch / refit lines
  append `; trunc N of M pre-truncation nonzero ACs` (`; trunc off` in `es` mode); the final block gains `dct rate lambda L mode M proxy
  total B /blk P = bpp | h0 ac (run+mag+sign+eob) /blk H ratio proxy/h0 R (eob excluded R2) | lambda_t16 by k v0..v15 | trunc last-snap N
  of M (x%) | hits x% (of E)` before `dct plane` (`lambda_t16 off` / `trunc off` / `hits off` for a part that is off), and the `dct[c]`
  lines ` proxy P (x/blk) trunc N`; the `done:` line says `(bit simulator, lambda 80)` / `lambda 80 es-only` / `lambda 80 trunc-only`; the
  `dct` banner appends the rate clause. On `--cuda` the bits and truncation counts come from the host re-snap after `download_model` (=
  the device's by check 7), `hits` is downloaded at print time and `evals` is host state (`Impl::dct_evals`, no device copy).
- Self-test: three new items (below). README: two flag rows, the `--dct-lambda` paragraph after the `--dct-q` one, two check lines and
  the `dct rate` sentence.

### Verification 1: builds, self-test, firewall (plan 8.1)

`--dct-selftest` on both builds (`diff`-identical output, rc 0): the nine section-3 lines unchanged, plus
```
  rate cost table              PASS  0 of 16 table values differ from 1,3,3,5,5,5,5,7 bits; monotone yes; c_run(62) = 11 c_mag(256) = 17 bits; sign 1 EOB 2 bits
  rate proxy bits              PASS  hand-built block 448/16 = 28.00 bits (hand count 28); lone |q| = 1 at zigzag 63: 13.00 bits (13, no EOB); empty block 2.00 bits (2)
  rate truncation              PASS  (a) lone 63 at lambda_t 0.7: q 1 trunc 0 (of 1), at 0.8: q 0 trunc 1; (b) interior p3 at 2: q 1 trunc 0 (of 3; p1 3 p10 3), at 3: q 0 trunc 1 (p1 3 p10 3); (c) lambda 0: trunc 0 nz_before 0 q 1; (d) first-order c = 0.5L: q 0 trunc 1 (of 1)
dct-selftest: all passed
```
(hand-built block: tokens (run 0, |q| 3) (0, 1) (2, 7) (14, 1) + EOB = (1+3+1) + (1+1+1) + (3+5+1) + (7+1+1) + 2 = 28 bits; (a) dD = 12^2 -
(12 - 15)^2 = 135 against saved = 176 + 16 + 16 - 32 = 176: kept at 0.7 (123.2), dropped at 0.8 (140.8); (b) the p3 saving is 80 + c_run(6) -
c_run(8) = 80 + 80 - 112 = 48 rather than its 80-bit token, so lambda_t 2 keeps it (135 >= 96) and 3 drops it (135 < 144).)

Regression pin (final CPU binary):
```
build\Release\ntc.exe --crop 512 --mlp-pairs 32 --iters 200 --print-every 200 --save-every 200 --out out_reg
iter    200  mse 269.314  psnr  23.83 dB  best  23.83 | q8 psnr  23.83 @ 0.552 bpp (ent 0.500, ctx 0.084) | mlp batch 0.00442 dstd 9.65e-05 | lat mean -0.051 sd 0.551 max 2.08 | 8.5s (23.42 it/s)
done: final psnr 23.83 dB (best 23.83) at fp32 2.103 bpp | 8-bit latent + fp16 mlp: psnr 23.83 dB at 0.552 bpp raw, 0.500 bpp entropy-coded, 0.084 bpp with an (up, left) context on level 0 [level 0 alone: 0.032] | 8.7s
sha256 out_reg/model.bin cb88470777e1d041c8803bcf3ca531434b9a2b2b5bce2ef88e031fbc86d69600 (= sections 3 and 7)
```
The pin was lost twice on the way and is the reason for deviation 1: with the rate pass written inside `LatentTrainer::step` (the plan's
hook) the pin printed the same `psnr 23.83 dB` line but `model.bin` hashed `5fe8b3f8...` (14948 of the 17262 floats differ, max relative
2.0e-3 after 200 iterations: a rounding-level divergence amplified by training); with the pass moved to a `noinline` function called
under `if (rate_on)` inside the pair loop, `cf5357ba...`. The trainer is built `/fp:fast /arch:AVX2`, so the numerics of a function
depend on how the compiler vectorizes and contracts *its own* loops, and any new code in the function (an OpenMP region, an opaque call
in the loop) changes that. A HEAD binary built in the scratchpad from `git archive` reproduces `cb884707...` today, and so does the
`step_impl<false>` instantiation, whose text is the original function's.

CPU lambda-0 path with `--dct-q` (2 channels, refits): `m1.png --rng hash --block 4 --latent 0 0 2 --latent2 0 0 4 --filter
nearest,bilinear --pos lv1local --dct-q 30,30 --dct-start 0.25 --dct-refit 50 --qes 0,8 --mlp 27,27 --mlp-pairs 32 --iters 200
--print-every 100 --lr-anneal 0.5 0.05 --mlp-fd 0.5 --leak 0.0009765625 --dct-map all` with the new and the HEAD CPU binary:
`model.bin`, `dct_map.png`, `dct_map_dconly.png`, `recon_q_final.png` `cmp`-identical, the logs identical apart from the timing fields
(`done: final psnr 24.77 dB ... 24.77 dB at 6.565 bpp raw, 3.244 bpp entropy-coded, 3.080 bpp ... [level 0 alone: 1.240]`).

The nine README `--cuda-check` lines (the seven `ntc ... --cuda-check` lines plus the `--dct-q 90` / `20` variants of the DCT line),
final CUDA binary: every one `cuda-check: all passed`, rc 0, 6 / 5 / 6 / 6 / 5 / 7 / 7 / 7 / 7 PASS lines. Figures: decode 1.788e-07 /
1.788e-07 / 1.192e-07 / 1.192e-07 / 1.192e-07 / 1.192e-07 / 1.192e-07; latent ES grad 2.174e-05 / 7.003e-06 / 4.645e-05 / 3.884e-05 /
3.669e-05 / 3.305e-05 / 2.095e-05 (q 90: 3.974e-05, q 20: 3.304e-05); qat search `3 of 524288` (worse 0, better 1, ties 1) / `3 of
786432` (worse 2) / `1 of 262144` (ties 1); qes snap 0 / 0 everywhere; dct snap 0 / 0 (max 0 ulp) / 0 before and after the step; dct
refit moved 195 / 193 / 196 / 5 of 4096 / 4096 / 4096 / 8192. Four of the latent-ES-grad figures differ from the ones quoted in the
addendum (2.126e-05, 7.586e-06, 3.821e-05, 3.085e-05): the HEAD binary built today prints today's values (`2.174e-05` on line 1,
`3.305e-05` on the DCT line, checked), so they are the current toolchain's and not this change's; the DCT lines' other figures (mlp ES
3.442e-07 / 3.746e-07 / 3.670e-07, FD 8.361e-05 / 7.959e-05 / 8.316e-05) are the addendum's.

The strongest firewall (final CUDA binary), the 2-plane start-0.25 run without `--dct-lambda`:
```
build_cuda\Release\ntc.exe m1.png --cuda --block 4 --latent 0 0 2 --latent2 0 0 4 --filter nearest,bilinear --pos lv1local --dct-q 30,30 --dct-start 0.25 --dct-refit 500 --qes 0,8 --mlp 27,27 --mlp-pairs 256 --iters 8000 --lr-anneal 0.5 0.05 --mlp-fd 0.5 --leak 0.0009765625 --dct-map all --write-ntcb --out out_rate_fw
fc /b out_rate_fw\model.bin out_m1_b4_dct30x30_c4bilin_qes8_mlp27_refit500_start25_cuda8k_fd50_ntcb\model.bin -> FC: no differences encountered
fc /b out_rate_fw\model.ntcb ...\model.ntcb -> FC: no differences encountered
done: final psnr 26.82 dB (best 35.03) at fp32 72.132 bpp | --dct-q 30,30 level 0 (bit simulator) + --qes 0,8 latent + fp16 mlp: psnr 26.82 dB at 4.974 bpp raw, 2.840 bpp entropy-coded, 2.765 bpp with an (up, left) context on level 0 [level 0 alone: 0.919] [dct: raw 2.906 h0 0.993 ctx 0.919 code raw/ctx 0.125/0.048 bpp; nz 4.63/blk eob0 12.5%]
file: out_rate_fw/model.ntcb 163101 bytes = 4.977 bpp (simulator raw 4.974 bpp); content 1303504 bits = simulator 1303952 - 448 side-info bits (stored in the header) [OK]; ...; self-check OK
```
= the log's `26.82 dB at 4.974 / 2.840 / 2.765`; the switch line, every refit line and the whole final block are identical to the log;
the only differing lines are the wall-time-paced progress prints (different iteration numbers, hence a different `best` sample before
the switch: 35.03 here, 33.89 in the log, from an `iter 1383` print the log had and this run did not) and the `file:` path.

### Verification 2: the lambda check lines (plan 8.3)

```
ntc chief1.png --cuda --cuda-check --block 8 --latent 0 0 1 --latent2 0 0 4 --filter nearest,bilinear --pos lv1local --dct-q 50 --dct-start 0 --qes 0,8 --qes-start 0 --mlp 17,17 --leak 0.0009765625 --iters 5 --print-every 5 --dct-lambda 80 --out out_chk
iter      0  dct: scale codes fitted ... psnr 11.28 -> 11.28 dB; nz 0.00/blk, eob-only 100.0%, clamped 0.00%; histogram k0..k15 0 0 294 3802 0 0 0 0 0 0 0 0 0 0 0 0; trunc 165 of 165 nonzero
  latent ES grad  : max |gpu - cpu| / rms(grad) = 3.304e-05 over 278528 values  PASS
  dct rate        : pair 3 bit differences 0 of 4096 differ, hits 0 = 0 (of 16384 = 16384); host re-snap of the downloaded shadow vs k_dct_snap: proxy bits 0 of 4096 differ, truncated 165 = 165 (of 165 = 165 nonzero)  PASS
  dct snap        : symbols 0 of 262144 differ, plane 0 of 262144 values differ (max 0 ulp), codes 0 of 4096 differ (device plane from k_dct_recon on the host's symbols: IDCT + clamp parity); after a latent step with lr 0.02: 0 / 0 (max 0 ulp) / 0 (k_dct_snap: forward DCT + quantizer + IDCT parity); proxy bits 0 of 8192 differ (--dct-lambda, both compares)  PASS
  dct refit       : host re-probe moved 196 of 4096 codes (>= 2: 0), uploaded through set_dct: symbols 0 differ, plane 0 (max 0 ulp), codes 0 (k_dct_recon with the refitted codes); after a latent step: 0 / 0 (max 0 ulp) / 0 (k_dct_snap quantizes with the refitted codes); proxy bits 0 of 8192 differ  PASS
cuda-check: all passed
--dct-lambda 100000 on the same line: trunc 165 of 165 nonzero at the fit; dct rate: pair 3 bit differences 0 of 4096 differ, hits 0 = 0 (of 16384 = 16384); ... proxy bits 0 of 4096 differ, truncated 165 = 165 (of 165 = 165 nonzero)  PASS; dct snap / dct refit 0 / 0 (max 0 ulp) / 0, proxy bits 0 of 8192 differ; all passed
ntc m1.png --cuda --cuda-check --block 4 --latent 0 0 2 --latent2 0 0 4 --filter nearest,bilinear --pos lv1local --dct-q 50,20 --dct-start 0 --qes 0,8 --qes-start 0 --mlp 27,27 --leak 0.0009765625 --iters 5 --print-every 5 --dct-lambda 80 --out out_chk
iter      0  dct: ... (mean gain ch 0.77 / 0.85); psnr 11.50 -> 11.48 dB; nz 0.00/blk, eob-only 100.0%, clamped 0.00%; histogram k0..k15 0 0 0 8187 5 0 ...; trunc 173 of 173 nonzero
  latent ES grad  : max |gpu - cpu| / rms(grad) = 2.094e-05 over 589824 values  PASS
  dct rate        : pair 3 bit differences 0 of 8192 differ, hits 0 = 0 (of 32768 = 32768); host re-snap of the downloaded shadow vs k_dct_snap: proxy bits 0 of 8192 differ, truncated 173 = 173 (of 173 = 173 nonzero)  PASS
  dct snap        : symbols 0 of 524288 differ, plane 0 of 524288 values (2 channels) differ (max 0 ulp), codes 0 of 8192 differ (...); after a latent step with lr 0.02: 0 / 0 (max 0 ulp) / 0 (...); proxy bits 0 of 16384 differ (--dct-lambda, both compares)  PASS
  dct refit       : host re-probe moved 5 of 8192 codes (>= 2: 0), ... 0 / 0 (max 0 ulp) / 0 ...; proxy bits 0 of 16384 differ  PASS
cuda-check: all passed
```
At iteration 0 the plane is the initial `N(0, 0.1)` draw: every one of its 165 / 173 nonzero ACs is below the lambda-80 threshold, so
both lines exercise the truncation walk fully (`trunc 165 of 165`) but part A with `hits 0` (every pair evaluation sees a DC-only block on
both sides). The parity of part A with a real signal is the same check on a trained lambda model (the m1 lambda-80 run of the table):
```
ntc m1.png --cuda --cuda-check --block 4 --latent 0 0 1 --latent2 0 0 4 --filter nearest,bilinear --pos lv1local --dct-q 30 --qes 0,8 --mlp 27,27 --leak 0.0009765625 --dct-lambda 80 --load out_m1_b4_dct30_c4bilin_qes8_mlp27_refit500_start25_lam80_cuda8k_fd50_ntcb/model.bin --iters 5 --print-every 5 --out out_chk
  decode          : max |gpu - cpu| = 4.172e-07  PASS
  mlp ES dl       : max |gpu - cpu| / rms(dl) = 2.505e-06 over 8 pairs  PASS
  mlp FD dl       : max |gpu - cpu| / rms(dl) = 1.407e-03 over 24 weights  PASS
  latent ES grad  : max |gpu - cpu| / rms(grad) = 5.150e-05 over 327680 values  PASS
  dct rate        : pair 3 bit differences 0 of 4096 differ, hits 1132 = 1132 (of 16384 = 16384); host re-snap of the downloaded shadow vs k_dct_snap: proxy bits 0 of 4096 differ, truncated 1 = 1 (of 8778 = 8778 nonzero)  PASS
  qes snap        : device snapped copy vs host snap of the downloaded shadow: 0 of 327680 values differ, 0 after a latent step with lr 0.02  PASS
  dct snap        : symbols 0 of 262144 differ, plane 0 of 262144 values differ (max 0 ulp), codes 0 of 4096 differ (...); after a latent step with lr 0.02: 0 / 0 (max 0 ulp) / 0 (...); proxy bits 0 of 8192 differ (--dct-lambda, both compares)  PASS
  dct refit       : host re-probe moved 479 of 4096 codes (>= 2: 0), ...: symbols 0 differ, plane 0 (max 0 ulp), codes 0 (...); after a latent step: 0 / 0 (max 0 ulp) / 0 (...); proxy bits 0 of 8192 differ  PASS
cuda-check: all passed
```
(1132 of the 16384 (block, channel, pair) evaluations have a nonzero bit difference, all 4096 pair-3 differences equal on both sides.
The same on the model11 lambda-80 model: `dct rate ... hits 77 = 77 (of 57276 = 57276); ... proxy bits 0 of 14319 differ, truncated 1 = 1
(of 5862 = 5862 nonzero)  PASS` and every DCT line PASS, but check 1 reads `decode : max |gpu - cpu| = 1.031e-05  FAIL` against its 1e-5
tolerance; the HEAD binary on the lambda-0 model11 model fails too (`latent ES grad ... 1.768e-03 ... FAIL` against 1e-3, decode
8.792e-06): `--cuda-check` on a trained 17,17 model11 decoder is outside the tolerances the tool was set for, independently of this change.)

### Verification 3: CPU vs GPU, 500 iterations, `--rng hash`, `--dct-lambda 80` (plan 8.4)

`chief1.png --rng hash --iters 500 --lr-anneal 0.5 0.05 --mlp-fd 0.5 --mlp-pairs 32 --print-every 100 --block 8 --latent 0 0 1 --latent2 0 0 4
--filter nearest,bilinear --pos lv1local --dct-q 50 --dct-refit 50 --dct-start 0.25 --qes 0,8 --mlp 17,17 --leak 0.0009765625 --dct-lambda 80`
(`out_dct_lam_cpu.log`, `out_dct_lam_gpu.log` with `--cuda`):
```
both  iter    125  dct: scale codes fitted ... psnr 26.44 -> 25.35 dB; nz 11.68/blk, eob-only 0.0%, clamped 6.44%; histogram k0..k15 0 1 27 272 384 228 3183 1 0 ...; trunc 1689 of 49518 nonzero   (identical)
CPU   iter    150  dct: codes refitted (1); moved >= 1: 14.3%, >= 2: 0.0% (of 4096); psnr 26.30 -> 26.35 dB; nz 10.10/blk, eob-only 0.3%, clamped 5.84%; histogram ... 0 1 18 139 384 274 3087 193 ...; trunc 2203 of 43566 nonzero
GPU   iter    150  ... (identical through `clamped 5.84%` and the histogram) ...; trunc 2204 of 43566 nonzero
CPU   iter    200  mse 243.013  psnr  24.27 dB  best  25.40 | dct50,q8 psnr  24.27 @ 5.619 bpp (ent 2.307, ctx 2.078) | dct nz 19.18 lnz 40.6 eob0 0.2% clamp 7.01% bits/blk 325.6/116.5/101.9 proxy 111.1 chg sym 81389 zq 256106 trunc 2484 hits 90.8%
GPU   iter    200  mse 242.968  psnr  24.28 dB  best  25.40 | dct50,q8 psnr  24.27 @ 5.617 bpp (ent 2.307, ctx 2.078) | dct nz 19.17 lnz 40.6 eob0 0.2% clamp 7.02% bits/blk 325.5/116.5/101.9 proxy 111.1 chg sym 81337 zq 256121 trunc 2493 hits 90.7%
CPU   iter    500  mse 64.578  psnr  30.03 dB  best  30.03 | dct50,q8 psnr  30.03 @ 2.681 bpp (ent 1.465, ctx 1.380) | dct nz 7.41 lnz 17.9 eob0 5.7% clamp 1.95% bits/blk 137.6/62.7/57.3 proxy 57.4 chg sym 22280 zq 241896 trunc 3070 hits 84.5%
GPU   iter    500  mse 64.610  psnr  30.03 dB  best  30.03 | dct50,q8 psnr  30.03 @ 2.679 bpp (ent 1.465, ctx 1.380) | dct nz 7.40 lnz 17.8 eob0 5.4% clamp 1.93% bits/blk 137.5/62.7/57.2 proxy 57.3 chg sym 22092 zq 241143 trunc 3097 hits 84.6%
CPU   dct rate lambda 80 mode es+trunc proxy total 235072 /blk 57.39 = 0.8967 bpp | h0 ac (run+mag+sign+eob) /blk 56.15 ratio proxy/h0 1.022 | lambda_t16 by k 104.5 45.48 19.8 8.617 3.751 1.633 0.7107 0.3093 0.1346 0.05861 0.02551 0.0111 0.004833 0.002104 0.0009157 0.0003986 | trunc last-snap 3070 of 33432 (9.2%) | hits 84.5% (of 16384)
GPU   dct rate lambda 80 mode es+trunc proxy total 234554 /blk 57.26 = 0.8948 bpp | h0 ac (run+mag+sign+eob) /blk 56.05 ratio proxy/h0 1.022 | lambda_t16 by k (the same 16 values) | trunc last-snap 3097 of 33419 (9.3%) | hits 84.6% (of 16384)
CPU   done: final psnr 30.03 dB (best 30.03) ... (bit simulator, lambda 80) ... 30.03 dB at 2.681 bpp raw, 1.465 bpp entropy-coded, 1.380 bpp ... [level 0 alone: 0.894] [dct: raw 2.150 h0 0.980 ctx 0.894 ...; nz 7.41/blk eob0 5.7%] | 44.6s
GPU   done: final psnr 30.03 dB (best 30.03) ... (bit simulator, lambda 80) ... 30.03 dB at 2.679 bpp raw, 1.465 bpp entropy-coded, 1.380 bpp ... [level 0 alone: 0.893] [dct: raw 2.147 h0 0.979 ctx 0.893 ...; nz 7.40/blk eob0 5.4%] | 1.6s
```
Identical through the switch (same histogram, same `trunc 1689 of 49518`) and through refit 1's movers and PSNRs; one truncation apart
from refit 1 on (2203 / 2204: the addendum's one-block rounding divergence of the two backends' decodes, now visible as one coefficient's
`dD < lambda_t saved` decision), and display-precision agreement at 500 (0.00 dB, 0.002 bpp raw, 0.1% hits). The `lambda_t16` table
(host-computed, uploaded) is the same 16 values on both.

### Verification 4: loads, decoder, container (plan 8.5)

`--load out_m1_b4_dct30_c4bilin_qes8_mlp27_refit500_start25_lam80_cuda8k_fd50_ntcb/model.bin --iters 0` with the run's layout options
and `--dct-lambda 80`, CPU and `--cuda` (`diff`-identical `dct` / `done` lines):
```
dct      : symbols and scale codes restored from the file (level 0 = IDCT of the symbols, no re-snap at load; codes frozen)
dct rate lambda 80 mode es+trunc proxy total 58646 /blk 14.32 = 0.2237 bpp | h0 ac (run+mag+sign+eob) /blk 14.56 ratio proxy/h0 0.983 | lambda_t16 by k 104.5 ... 0.0003986 | trunc last-snap 0 of 0 (0.0%) | hits 0.0% (of 0)
done: final psnr 26.05 dB (best -1.00) at fp32 40.129 bpp | --dct-q 30 level 0 (bit simulator, lambda 80) + --qes 0,8 latent + fp16 mlp: psnr 26.05 dB at 2.898 bpp raw, 2.184 bpp entropy-coded, 2.161 bpp with an (up, left) context on level 0 [level 0 alone: 0.299] [dct: raw 0.833 h0 0.322 ctx 0.299 code raw/ctx 0.062/0.016 bpp; nz 2.14/blk eob0 54.0%] | 0.3s
```
(= the run's `done:` figures and its `dct rate` proxy / h0 figures; `trunc 0 of 0`, `hits 0.0% (of 0)`: a load does not re-snap and no
latent step ran.)
```
build\Release\ntc_decode.exe out_m1_..._lam80_..._ntcb/model.bin --compare out_m1_..._lam80_..._ntcb/recon_q_final.png --verify
level 0  : 512x512x1 nearest, dct 8x8, q 30, dc step 4, 4-bit scale codes, 2.1 nonzero ACs/block, 54.0% EOB-only, fp32 planar
dct      : nz mean 2.14/blk, eob0 54.0%, lnz mean 12.2, clamped 0.03%, codes k0..k15 0 0 0 0 46 2191 1429 430 0 ...; raw 53.3 bits/blk = 0.8326 bpp (...)
idct     : 1.27 ms single-threaded (4096 blocks; 8.4 MAC/texel with zero-skipping, 16 dense; nnz 8778)
compare t0: ... (512x512): PSNR 89.31 dB, max |diff| 1, |diff| histogram: 0: 786372 (99.9924%), 1: 60 (0.0076%), 2: 0 (0.0000%), >2: 0 (0.0000%)
verify   : pre-nonlinearity: max 0 ulp (0 of 786432 values differ); output fp32: max 42 ulp, max |diff| 7.75e-07; RGB8 mismatches: 57 of 786432 (0.00725%)
```
Container round trip (`--load model.bin --iters 0 --write-ntcb --resave --out out_ntcb_lam80`, `ntc_decode` on the ntcb and on the
resaved model.bin with `--compare m1.png`, `cmp`, `--load model.ntcb --iters 0 --cuda`):
```
ntcb     : mlp weights rounded to fp16 (0 of 1056 changed, max |dw| 0.000e+00); psnr 26.05 -> 26.05 dB; figures from here on are post-rounding
resave   : wrote out_ntcb_lam80/model.bin as v13 (level 0 DCT-coded, q 30, DC step 4; symbols and codes from the loaded v13 file)
file: out_ntcb_lam80/model.ntcb 95093 bytes = 2.902 bpp (simulator raw 2.898 bpp); content 759456 bits = simulator 759808 - 352 side-info bits (stored in the header) [OK]; container header 1040 + section headers 192 + sync 42 + padding 14 bits; bpp over the 512x512 decode size, as every bpp in this log; self-check OK
ntc_decode out_ntcb_lam80/model.ntcb: model : out_ntcb_lam80/model.ntcb (ntcb v1); file : 95093 bytes = 2.902 bpp (simulator raw 2.898 bpp: 742560 level bits + 352 side-info (stored in the header) + 16896 mlp bits; content 759456 bits matches the simulator exactly; ...)
compare t0: m1.png (512x512): PSNR 26.05 dB, max |diff| 114 ...; verify: pre-nonlinearity: max 0 ulp (0 of 786432 values differ); output fp32: max 42 ulp; RGB8 mismatches: 57 of 786432 (0.00725%)
cmp of the ntcb and the resaved-model.bin decoder PNGs: identical; reload done: final psnr 26.05 dB (best -1.00) ... (bit simulator, lambda 80) ... 26.05 dB at 2.898 bpp raw, 2.184 bpp entropy-coded, 2.161 bpp ... [level 0 alone: 0.299] [dct: raw 0.833 h0 0.322 ctx 0.299 ...]  (= the run's line)
```
The container's 95093 bytes are the run's own `file:` line (the writer only sees symbols; a truncated block is a legal block).

### Step-2 / step-3 smoke runs (plan section 10; CPU, final binary; the section-3 chief1 layout, `--rng hash --iters 100 --print-every 20 --mlp-pairs 32 --dct-q 30 --dct-start 0.5`)

```
lambda 0:      iter 50 dct: psnr 16.82 -> 16.62 dB; nz 3.81/blk, eob-only 0.1%; histogram k0..k15 0 0 3 799 2990 304 ...   (no trunc clause)
               iter 100 psnr 22.93 dB | dct30,q8 psnr 22.93 @ 2.190 bpp (ent 1.089, ctx 1.025) | dct nz 5.45 lnz 11.3 eob0 0.3% clamp 6.16% bits/blk 106.2/37.6/33.5 chg sym 16367 zq 244660   (= the section-3 `--dct-q 30` line; the done line's 2.190 / 1.089 / 1.025, level 0 alone 0.523, dct raw 1.659 h0 0.587 ctx 0.523 too)
lambda 80:     iter 50 dct: psnr 16.82 -> 16.61 dB; nz 3.58/blk, eob-only 0.7%, clamped 0.09%; histogram (the same); trunc 928 of 15607 nonzero
               iter 100 psnr 22.54 dB | dct30,q8 psnr 22.54 @ 1.733 bpp (ent 0.955, ctx 0.912) | dct nz 3.62 lnz 8.3 eob0 10.5% clamp 3.68% bits/blk 76.9/29.1/26.3 proxy 21.9 chg sym 11165 zq 243307 trunc 1758 hits 58.6%
               dct rate lambda 80 mode es+trunc proxy total 89881 /blk 21.94 = 0.3429 bpp | h0 ac (run+mag+sign+eob) /blk 22.02 ratio proxy/h0 0.996 | ... | trunc last-snap 1758 of 16571 (10.6%) | hits 58.6% (of 16384)
               done: ... (bit simulator, lambda 80) ... 22.54 dB at 1.733 bpp raw, 0.955 bpp entropy-coded, 0.912 bpp ... [level 0 alone: 0.411] [dct: raw 1.201 h0 0.454 ctx 0.411 ...]
lambda 100000: iter 50 dct: psnr 16.82 -> 16.07 dB; nz 0.00/blk, eob-only 100.0%, clamped 0.00%; ...; trunc 15607 of 15607 nonzero
               iter 100 psnr 19.27 dB | dct30,q8 psnr 19.27 @ 0.828 bpp (ent 0.607, ctx 0.601) | dct nz 0.00 lnz 0.0 eob0 100.0% clamp 0.00% bits/blk 19.0/7.0/6.6 proxy 2.0 chg sym 3513 zq 224832 trunc 24600 hits 0.0%
               dct rate lambda 100000 mode es+trunc proxy total 8192 /blk 2.00 = 0.0312 bpp | h0 ac (run+mag+sign+eob) /blk 0.00 ratio proxy/h0 0.000 | ... | trunc last-snap 24600 of 24600 (100.0%) | hits 0.0% (of 16384)
```
Lambda 100000 makes every block DC-only (`lnz 0.0`, `eob0 100.0%`, proxy exactly 2 bits per block = the EOB); lambda 80 reports fewer `nz`
than lambda 0 (3.62 vs 5.45) with `hits 58.6%`. The plan's step-3 sanity check, one build with `dct_rate_pass` pointed at the snapped copy
`D.zdec()` instead of the shadow (then reverted; the restored binary reproduces the pin `cb884707...`, the self-test and the line above):
`iter 60 ... dct nz 4.22 ... trunc 1078 hits 3.0%`, `iter 80 ... nz 4.92 ... hits 6.7%`, `iter 100 ... dct nz 5.26 lnz 11.2 eob0 1.4% clamp 6.05%
bits/blk 103.1/36.9/33.0 proxy 29.9 ... trunc 1029 hits 8.8%`, `dct rate ... trunc last-snap 1029 of 22561 (4.6%) | hits 8.8% (of 16384)`: on the
snapped copy the term has next to no signal (`nz` stays at the lambda-0 level, 5.26 vs 5.45; the 3-9% of hits are the clamped texels and the
small-step blocks section 2.3 of the plan excepts) against 58.6% and `nz 3.62` on the shadow.

### Experiments (plan section 7; GPU, every run `--write-ntcb`, then `build\Release\ntc_decode.exe <out>/model.ntcb -o <out>/ntcb_decoded --compare <image>`)

Commands: the plan's three lines verbatim (`m1.png --cuda --block 4 --latent 0 0 1 --latent2 0 0 4 --filter nearest,bilinear --pos lv1local
--dct-q 30 --dct-start 0.25 --dct-refit 500 --qes 0,8 --mlp 27,27 --mlp-pairs 256 --iters 8000 --lr-anneal 0.5 0.05 --mlp-fd 0.5 --leak
0.0009765625 --dct-map all --dct-lambda L --write-ntcb`; the same with `--latent 0 0 2 --dct-q 30,30`; `model11.png --cuda --block 6 --latent
0 0 1 --latent2 0 0 4 --filter nearest,bilinear --pos lv1local --dct-q 15 --dct-refit 500 --qes 0,8 --mlp 17,17 --mlp-pairs 256 --iters 8000
--lr-anneal 0.5 0.05 --mlp-fd 0.5 --leak 0.0009765625 --dct-map all --dct-lambda L --write-ntcb`), output directories
`out_m1_b4_dct30_c4bilin_qes8_mlp27_refit500_start25_lam{0,20,80,80_es,80_trunc,320}_cuda8k_fd50_ntcb`,
`out_m1_b4_dct30x30_..._start25_lam{20,80,320}_cuda8k_fd50_ntcb`, `out_model11_b6_dct15_c4bilin_qes8_mlp17_refit500_lam{0,80,320,1280}_cuda8k_fd50_ntcb`,
logs `<out>.log` and `<out>/ntcb_decode.log`. PSNR = the decoder's `compare t0` figure from the .ntcb; container bytes / bpp from the `file:` line;
order-0 bpp = the `done:` line's entropy-coded figure; level 0 h0 = the `[dct: ... h0 ...]` figure; nz / eob0 from the same bracket; trunc and
hits from the `dct rate` line. The q-curve column is the piecewise-linear interpolation of the plan's reference points on the (order-0 bpp,
PSNR) plane (m1 1 plane: q 30 26.08 @ 2.443, q 50 26.41 @ 2.810, q 70 26.88 @ 3.426; 2 planes: 20,20 26.11 @ 2.550, 30,30 26.82 @ 2.840,
30,50 27.08 @ 3.224; model11: q 5 35.13 @ 0.906, q 15 35.84 @ 0.972, q 50 39.56 @ 1.877), extrapolated with the nearest segment's slope
outside their range (marked). The plan's 1-plane q-30 reference used `--dct-start 0.5`; the lambda-0 run here (start 0.25) is the control.
Review fix: every m1 1-plane lambda point lies *below* the lowest plan reference (q 30 at 2.443), so the column's original entries for
those rows were extrapolations (25.96 / 25.85 / 25.88 / 26.05 / 25.70 dB at lambda 20 / 80 / 80 es / 80 trunc / 320, all "extrapolated
below q 30") and are replaced below by interpolations between the start-0.25 references run for the review (q 15 25.60 @ 2.130, q 20
25.89 @ 2.228, q 30 = the control 26.27 @ 2.425; the two reference rows and the lambda-320 trunc-only row are the review's runs, see
"Review fixes" at the end of this section); the lambda-320 point at 2.015 is still below q 15 and stays an extrapolation (marked).

| run | lambda | PSNR (decoder, .ntcb) | container bytes = bpp | order-0 bpp | level 0 h0 bpp | nz/(blk,ch) | eob0 | trunc (last snap) | hits | q curve at that order-0 bpp | above? |
|---|---|---|---|---|---|---|---|---|---|---|---|
| m1 1 plane q 30 | 0 | 26.27 | 124715 = 3.806 | 2.425 | 0.581 | 5.76 | 5.8% | - | - | 26.27 dB (the q 30 reference itself) | control (+0.00 dB) |
| m1 1 plane q 20 (reference) | 0 | 25.89 | 102249 = 3.120 | 2.228 | 0.364 | 3.02 | 26.9% | - | - | reference | - |
| m1 1 plane q 15 (reference) | 0 | 25.60 | 91393 = 2.789 | 2.130 | 0.260 | 1.69 | 45.4% | - | - | reference | - |
| m1 1 plane q 30 | 20 | 26.34 | 110023 = 3.358 | 2.308 | 0.461 | 3.97 | 28.0% | 922 of 17165 (5.4%) | 86.1% | 26.04 dB (between q 20 and q 30) | yes (+0.30 dB) |
| m1 1 plane q 30 | 80 | 26.05 | 95093 = 2.902 | 2.184 | 0.322 | 2.14 | 54.0% | 3183 of 11961 (26.6%) | 54.5% | 25.76 dB (between q 15 and q 20) | yes (+0.29 dB) |
| m1 1 plane q 30 | 80 es | 26.09 | 97587 = 2.978 | 2.218 | 0.355 | 2.45 | 48.9% | 0 of 0 (0.0%) | 57.8% | 25.86 dB (between q 15 and q 20) | yes (+0.23 dB) |
| m1 1 plane q 30 | 80 trunc | 26.27 | 123645 = 3.773 | 2.412 | 0.575 | 5.63 | 11.5% | 3182 of 26236 (12.1%) | 0.0% | 26.25 dB (26.245; between q 20 and q 30) | yes (+0.03 dB) |
| m1 1 plane q 30 | 320 | 25.33 | 80705 = 2.463 | 2.015 | 0.161 | 0.39 | 79.8% | 19476 of 21060 (92.5%) | 21.1% | 25.26 dB (extrapolated below q 15) | yes (+0.07 dB, extrapolated) |
| m1 1 plane q 30 | 320 trunc | 26.20 | 112697 = 3.439 | 2.351 | 0.488 | 4.29 | 31.2% | 16175 of 33755 (47.9%) | - | 26.13 dB (between q 20 and q 30) | yes (+0.07 dB) |
| m1 2 planes q 30,30 | 20 | 26.69 | 132481 = 4.043 | 2.567 | 0.723 | 2.76 | 40.0% | 1818 of 24398 (7.5%) | 76.0% | 26.15 dB (between q 20,20 and q 30,30) | yes (+0.54 dB) |
| m1 2 planes q 30,30 | 80 | 26.22 | 109681 = 3.347 | 2.358 | 0.511 | 1.36 | 61.0% | 7212 of 18392 (39.2%) | 41.5% | 25.64 dB (extrapolated below q 20,20) | yes (+0.58 dB) |
| m1 2 planes q 30,30 | 320 | 25.29 | 92889 = 2.835 | 2.146 | 0.314 | 0.34 | 86.2% | 34956 of 37740 (92.6%) | 11.8% | 25.12 dB (extrapolated below q 20,20) | yes (+0.17 dB) |
| model11 q 15 | 0 | 35.84 | 155590 = 1.358 | 0.972 | 0.184 | 0.65 | 72.8% | - | - | 35.84 dB (between q 5 and q 15) | no (+0.00 dB) |
| model11 q 15 | 80 | 35.97 | 148706 = 1.298 | 0.963 | 0.164 | 0.41 | 79.6% | 1115 of 6976 (16.0%) | 21.6% | 35.74 dB (between q 5 and q 15) | yes (+0.23 dB) |
| model11 q 15 | 320 | 35.54 | 142424 = 1.243 | 0.946 | 0.148 | 0.19 | 89.0% | 8989 of 11709 (76.8%) | 5.6% | 35.56 dB (between q 5 and q 15) | no (-0.02 dB) |
| model11 q 15 | 1280 | 35.00 | 138484 = 1.209 | 0.927 | 0.132 | 0.05 | 96.2% | 39068 of 39818 (98.1%) | 0.9% | 35.36 dB (between q 5 and q 15) | no (-0.36 dB) |

The lambda-0 model11 run reproduces the reference log to the digit (35.84 dB at 0.972 bpp order-0, nz 0.65/blk, eob0 72.8%: the GPU
path is deterministic and the flag off changes nothing). The `done:` lines and per-run figures:
```
m1 lam0:        done: ... (bit simulator) ... 26.27 dB at 3.802 bpp raw, 2.425 bpp entropy-coded, 2.381 bpp ... [level 0 alone: 0.537] [dct: raw 1.737 h0 0.581 ctx 0.537 code raw/ctx 0.062/0.022 bpp; nz 5.76/blk eob0 5.8%] | 27.7s; reprobe 4.0% / 0.0%; codes k1 67 k2 1228 k3 1316 k4 539 k5 920 k6 26 mean_k 3.27
m1 lam20:       done: ... (bit simulator, lambda 20) ... 26.34 dB at 3.354 bpp raw, 2.308 bpp entropy-coded, 2.278 bpp ... [level 0 alone: 0.431] [dct: raw 1.288 h0 0.461 ctx 0.431 code raw/ctx 0.062/0.016 bpp; nz 3.97/blk eob0 28.0%] | 31.3s
                dct rate lambda 20 mode es+trunc proxy total 93555 /blk 22.84 = 0.3569 bpp | h0 ac (run+mag+sign+eob) /blk 23.14 ratio proxy/h0 0.987 | ... | trunc last-snap 922 of 17165 (5.4%) | hits 86.1% (of 16384); reprobe 3.1%; codes k4 1087 k5 1975 k6 1034 mean_k 4.99
m1 lam80:       done: ... (bit simulator, lambda 80) ... 26.05 dB at 2.898 bpp raw, 2.184 bpp entropy-coded, 2.161 bpp ... [level 0 alone: 0.299] [dct: raw 0.833 h0 0.322 ctx 0.299 code raw/ctx 0.062/0.016 bpp; nz 2.14/blk eob0 54.0%] | 31.3s
                dct rate lambda 80 mode es+trunc proxy total 58646 /blk 14.32 = 0.2237 bpp | h0 ac ... /blk 14.56 ratio proxy/h0 0.983 | ... | trunc last-snap 3183 of 11961 (26.6%) | hits 54.5% (of 16384); reprobe 11.6%; codes k4 46 k5 2191 k6 1429 k7 430 mean_k 5.55
m1 lam80 es:    done: ... (bit simulator, lambda 80 es-only) ... 26.09 dB at 2.975 bpp raw, 2.218 bpp entropy-coded, 2.193 bpp ... [level 0 alone: 0.330] [dct: raw 0.909 h0 0.355 ctx 0.330 ...; nz 2.45/blk eob0 48.9%] | 29.5s
                dct rate lambda 80 mode es-only proxy total 65537 /blk 16.00 = 0.2500 bpp | h0 ac ... /blk 16.35 ratio proxy/h0 0.979 | ... | trunc last-snap 0 of 0 (0.0%) | hits 57.8% (of 16384); reprobe 48.1%; codes k3 1 k4 341 k5 2071 k6 1167 k7 516 mean_k 5.45
m1 lam80 trunc: done: ... (bit simulator, lambda 80 trunc-only) ... 26.27 dB at 3.770 bpp raw, 2.412 bpp entropy-coded, 2.367 bpp ... [level 0 alone: 0.530] [dct: raw 1.704 h0 0.575 ctx 0.530 ...; nz 5.63/blk eob0 11.5%] | 27.8s
                dct rate lambda 80 mode trunc-only proxy total 124294 /blk 30.35 = 0.4741 bpp | h0 ac ... /blk 29.68 ratio proxy/h0 1.022 | ... | trunc last-snap 3182 of 26236 (12.1%) | hits 0.0% (of 0); reprobe 4.6%; codes k1 45 k2 1259 k3 1313 k4 501 k5 946 k6 32 mean_k 3.28
m1 lam320:      done: ... (bit simulator, lambda 320) ... 25.33 dB at 2.459 bpp raw, 2.015 bpp entropy-coded, 2.000 bpp ... [level 0 alone: 0.146] [dct: raw 0.394 h0 0.161 ctx 0.146 ...; nz 0.39/blk eob0 79.8%] | 31.0s
                dct rate lambda 320 ... proxy total 20668 /blk 5.05 = 0.0788 bpp | h0 ac ... /blk 3.89 ratio proxy/h0 1.297 | ... | trunc last-snap 19476 of 21060 (92.5%) | hits 21.1% (of 16384); reprobe 52.0%; mean_k 5.53
m1x2 lam20:     done: ... (bit simulator, lambda 20) ... 26.69 dB at 4.040 bpp raw, 2.567 bpp entropy-coded, 2.521 bpp ... [level 0 alone: 0.676] [dct: raw 1.972 h0 0.723 ctx 0.676 code raw/ctx 0.125/0.036 bpp; nz 2.76/blk eob0 40.0%] | 31.9s; dct rate ... proxy total 137644 /blk 33.60 ... ratio proxy/h0 0.991 | trunc last-snap 1818 of 24398 (7.5%) | hits 76.0% (of 32768)
m1x2 lam80:     done: ... (bit simulator, lambda 80) ... 26.22 dB at 3.344 bpp raw, 2.358 bpp entropy-coded, 2.314 bpp ... [level 0 alone: 0.467] [dct: raw 1.276 h0 0.511 ctx 0.467 ...; nz 1.36/blk eob0 61.0%] | 32.0s; dct rate ... proxy total 81926 /blk 20.00 ... ratio proxy/h0 1.016 | trunc last-snap 7212 of 18392 (39.2%) | hits 41.5% (of 32768)
m1x2 lam320:    done: ... (bit simulator, lambda 320) ... 25.29 dB at 2.831 bpp raw, 2.146 bpp entropy-coded, 2.114 bpp ... [level 0 alone: 0.282] [dct: raw 0.764 h0 0.314 ctx 0.282 ...; nz 0.34/blk eob0 86.2%] | 31.6s; dct rate ... proxy total 36358 /blk 8.88 ... ratio proxy/h0 1.383 | trunc last-snap 34956 of 37740 (92.6%) | hits 11.8% (of 32768)
model11 lam0:   done: ... (bit simulator) ... 35.84 dB at 1.357 bpp raw, 0.972 bpp entropy-coded, 0.955 bpp ... [level 0 alone: 0.168] [dct: raw 0.459 h0 0.184 ctx 0.168 code raw/ctx 0.062/0.022 bpp; nz 0.65/blk eob0 72.8%] | 22.7s; codes k0 5761 k1 2961 k2 2621 k3 1971 k4 622 k5 236 k6 108 k7 39 mean_k 1.31 | dc-only-by-table 79.2%; reprobe 9.2%
model11 lam80:  done: ... (bit simulator, lambda 80) ... 35.97 dB at 1.297 bpp raw, 0.963 bpp entropy-coded, 0.942 bpp ... [level 0 alone: 0.143] [dct: raw 0.399 h0 0.164 ctx 0.143 code raw/ctx 0.062/0.018 bpp; nz 0.41/blk eob0 79.6%] | 24.9s
                dct rate lambda 80 ... proxy total 61439 /blk 4.29 = 0.0670 bpp | h0 ac ... /blk 3.38 ratio proxy/h0 1.269 | ... | trunc last-snap 1115 of 6976 (16.0%) | hits 21.6% (of 57276); codes k1 3 k2 35 k3 497 k4 3459 k5 4179 k6 3366 k7 2172 k8 608 mean_k 5.35 | dc-only-by-table 0.3%; reprobe 13.8%
model11 lam320: done: ... (bit simulator, lambda 320) ... 35.54 dB at 1.242 bpp raw, 0.946 bpp entropy-coded, 0.924 bpp ... [level 0 alone: 0.125] [dct: raw 0.344 h0 0.148 ctx 0.125 ...; nz 0.19/blk eob0 89.0%] | 24.4s; dct rate ... proxy total 46764 /blk 3.27 ... ratio proxy/h0 1.691 | trunc last-snap 8989 of 11709 (76.8%) | hits 5.6% (of 57276); mean_k 5.10 dc-only-by-table 3.1%
model11 lam1280: done: ... (bit simulator, lambda 1280) ... 35.00 dB at 1.208 bpp raw, 0.927 bpp entropy-coded, 0.905 bpp ... [level 0 alone: 0.110] [dct: raw 0.310 h0 0.132 ctx 0.110 ...; nz 0.05/blk eob0 96.2%] | 24.5s; dct rate ... proxy total 34590 /blk 2.42 ... ratio proxy/h0 3.644 | trunc last-snap 39068 of 39818 (98.1%) | hits 0.9% (of 57276); mean_k 4.65 dc-only-by-table 15.7%
```
Every switch line is the control's (`psnr 29.46 -> 23.84 dB`, histogram `0 0 0 0 0 2 2669 1345 80` on m1; `35.12 -> 24.61` on the 2-plane
runs; `43.37 -> 34.33`, `0 0 2 237 955 4235 5629 3216 45` on model11) up to the truncation of the first snap (`nz 11.20 -> 11.18 / 11.11 /
9.84` at lambda 0 / 20 / 80 / 320 on m1), since lambda acts from the switch on.

**Reading.**

1. *Part A has a signal.* `hits` is 54-86% at lambda 20-80 (m1), 42-76% (2 planes), 22% (model11 at lambda 80). That is the plan's
   prediction, not more: `hits` is per (block, channel, pair) over 63 coefficients, so with a per-coefficient crossing probability p the
   fraction of evaluations with at least one crossing is 1 - (1 - p)^63 = 54.5% at p = 1.24% and 86.1% at p = 3.1%, i.e. the per-
   coefficient crossing rate behind the m1 figures is 1-3%, the "few percent" the plan expected (model11 lambda 80: 21.6% -> p = 0.39%;
   lambda 1280: 0.9% -> p = 0.014%). `hits` also gauges how close the population sits to the bin edges: model11 lambda 1280 at 0.9% means
   almost no coefficient is within a perturbation of an edge, A's signal is exhausted there. The A-only / B-only pair at lambda 80 on m1
   attributes the effect: A alone (`es`) goes from the control's 26.27 dB at 2.425 bpp order-0 (level 0 h0 0.581, nz 5.76) to 26.09 at
   2.218 (0.355, nz 2.45), B alone (`trunc`) only to 26.27 at 2.412 (0.575, nz 5.63) although it zeroes 3182 ACs per snap, and both
   together to 26.05 at 2.184 (0.322, nz 2.14). So the rate reduction is almost entirely part A's. Why B alone does so little: truncation
   zeroes the *symbol*, the shadow keeps the coefficient, so the same population is truncated again at every snap (the trunc-only run's
   `trunc` at the switch and the eleven refit lines: 355 -> 1057 -> 2123 -> 2521 -> 2536 -> 2534 -> 2688 -> 2936 -> 2998 -> 3001 -> 3030 ->
   3153, the last snap 3182; the progress prints 354 (at the switch), 2539, 2761, 2555, 2981, 3005, 2684, 2880, 2865, 2913, 3031, 2918,
   3025, 3080, 3060, 3124, 3160, 3132, 3153, 3027, 3122, 3146, 3119, 3159, 3182: a rising trend with dips of up to 321). The decisive number is the
   trunc-only run's `trunc last-snap 3182 of 26236`: 26236 / 4096 = 6.41 pre-truncation nonzeros per block against the control's 5.76,
   so under B the mse ES pushes *more* coefficients past the (moved) edge, B only relocates the edge (risk 2 of the plan: the shadow
   grows back what the snap removes). The decomposition of the both-mode run: `trunc 3183 of 11961` -> 2.92 pre-truncation nonzeros per
   block against 5.76 (A: -2.84/blk), then B's truncation 2.92 -> 2.14 (-0.78/blk); es-only ends at 2.45. The lambda-320 trunc-only run
   of the review (below) says the same louder: `nz 4.29 (pre-trunc 8.24)`, `trunc 16175/33755` (47.9%), 8.24 pre-truncation nonzeros
   per block against 5.76. The A2 fallback is not needed.
2. *Where the points lie.* The m1 1-plane points all lie below the plan's lowest reference (q 30), so the first reading compared them
   with the lambda-0 control: lambda 20 gains +0.07 dB *and* 0.117 bpp order-0 (0.448 bpp of container: 124715 -> 110023 bytes) on
   the control, a strict improvement; lambda 80 costs 0.22 dB for 0.241 bpp (0.9 dB/bpp); lambda 320 costs 0.94 dB for 0.410 bpp (2.3
   dB/bpp). Re-judged by interpolation between the start-0.25 references q 15 (25.60 @ 2.130), q 20 (25.89 @ 2.228) and q 30 (the
   control, 26.27 @ 2.425): lambda 20 sits +0.30 dB above the q 20 -> 30 segment (26.34 vs 26.04 at 2.308), lambda 80 +0.29 above the
   q 15 -> 20 segment (26.05 vs 25.76 at 2.184), es-only +0.23 (26.09 vs 25.86 at 2.218), trunc-only +0.03 (26.27 vs 26.245 at 2.412),
   lambda 320 +0.07 above the q 15 -> 20 slope extrapolated to 2.015 (25.33 vs 25.26; below the lowest reference, so an extrapolation),
   and the lambda-320 trunc-only run +0.07 (26.20 vs 26.13 at 2.351). So on m1 lambda 20-80 buys 0.3 dB over changing q at the same
   order-0 rate, and lambda 320 is at best on the curve. With 2 planes lambda 20 sits +0.58 dB above the q 20,20 reference at 0.017 bpp
   more (26.69 vs 26.11) and 0.13 dB below the 30,30 control at 0.273 bpp less; the 2-plane lambda 80 / 320 rows are still
   extrapolations below q 20,20 (no lower 2-plane reference was run). On model11 lambda 80 gains +0.13 dB and 0.009 bpp order-0
   (container 1.358 -> 1.298 bpp) on the control, +0.23 dB above the curve; 320 and 1280 fall below (-0.02, -0.36). On the (container
   bpp, PSNR) plane the same ordering holds (the container tracks the raw simulator, which falls faster than order-0 as the ACs
   vanish). The best working points are lambda*/4 (m1) and lambda*/4 (model11: 80 of 333).
3. *Flat blocks go DC-only.* model11 `eob0` 72.8% -> 79.6% -> 89.0% -> 96.2% at lambda 0 / 80 / 320 / 1280 (`nz` 0.65 -> 0.41 -> 0.19
   -> 0.05). By eye in `dct_map_dconly.png` (white = DC-only): at lambda 0 the large uniform regions (the top left and bottom left of the map) carry
   scattered black blocks; at 80 those scattered blocks are gone and the black concentrates on the edges and the horizontal structure at the
   right;
   at 320 only edges remain; at 1280 the edge structure itself is mostly white. The `dc-only-by-table` figure does *not* track this:
   it drops from 79.2% (lambda 0, mean_k 1.31) to 0.3% (lambda 80, mean_k 5.35): under lambda the refits move the codes *up* (finer
   steps), so the blocks go DC-only through A's barrier and B's truncation, not through coarse tables. Why the probe assigns finer
   tables is a hypothesis, not a measurement: "a decoder that reads a DC-only plane becomes more sensitive to the selector" would be
   read off the mean gain, but the switch / refit lines print the mean-gain clause only for C > 1, and these are 1-plane runs. The end-of-run reprobe is 3-20% on the
   lambda <= 80 runs (4.0 / 3.1 / 11.6 / 4.6% on m1, 5.6 / 20.0% with 2 planes, 9.2 / 13.8% on model11) and 48-52% on the A-only m1 run
   and m1 lambda 320 (the decoder keeps changing its sensitivity there).
4. *The proxy tracks h0.* `proxy/h0` (AC tokens only) is 0.98-1.02 on m1 at lambda 0-80 and in the 500-iteration runs (1.022), the
   calibration of plan 2.2; it rises to 1.27-1.69 on model11 and 1.30-1.38 at lambda 320, and 3.6 at lambda 1280, where the stream
   is almost all EOBs: the proxy's fixed 2-bit EOB overcharges an adaptive coder's 0.1-0.7-bit EOB. A cheaper EOB in the model, or an
   adaptive estimate, would matter only in that regime. (The `dct rate` line's EOB-excluded ratio, added by the review, confirms the
   attribution: proxy tokens / h0 run+mag+sign is 0.954 on the m1 lambda-80 model, 0.793 on model11 lambda 80 and 0.705 at lambda 1280,
   so the whole excess above 1 is the EOB.) What the h0 lines of the model11 runs say about where the bits are: the code bits *rise*
   with lambda (h0 code 31666 -> 32976 -> 38084 -> 40813 at lambda 0 / 80 / 320 / 1280, `dc-only-by-table` 79.2% -> 0.3% -> 3.1% ->
   15.7%), the DC stays at ~70 kbit (75822 / 69192 / 69852 / 70418) and only the AC tokens fall (run + mag + sign + eob 61534 -> 48429
   -> 27653 -> 9493 bits, 61.5 -> 9.5 kbit). At lambda 1280 level 0 (0.132 bpp over 916416 texels) is DC 0.077 + codes 0.045 + AC 0.010
   bpp, and the q 5 reference (`out_model11_b6_dct5_c4bilin_qes8_mlp27_refit500_cuda8k_fd50`: h0 dc 75703 run 3889 mag 153 sign 583 eob
   824 code 29997 total 111245) already has AC = 5% of its level-0 bits. The proxy prices neither the DC nor the codes, so below q ~10
   lambda is not the lever: a rate-aware refit (the codes) or DC / code coding is.
5. *Cost.* The m1 runs take 31.3 s against the control's 27.7 s (+13%; the es-only run 29.5 s, trunc-only 27.8 s), model11 24.9 s
   against 22.7 s (+10%): more than the plan's 1.6-3.5% MAC estimate, because the two rate kernels launch 4096-16384 threads per pair
   (one per (block, channel), low occupancy on 170 SMs) eight times per latent step. Host runs: the 500-iteration CPU run took 44.6 s
   against the addendum's 42.3 s for the same layout without lambda.
6. *Where to start at low q.* lambda* = 30 S(q)^2 with the libjpeg scale S(q) = 50 / q below q 50, and the sweeps put the working
   point at lambda*/4 ~ 7.5 S(q)^2: q 30 start 20 (bracket 10 / 40), q 15 start 80 (40 / 160), q 10 start 200 (100 / 400), q 5 nominally
   750 (375 / 1500), but at q 5 the AC lever is ~0.03 bpp of level 0 (item 4: the q 5 reference's AC tokens are 5449 of 111245 h0 bits)
   and the DC and the codes, which lambda does not price, are the rest.

### Deviations from the plan, with reasons

1. `LatentTrainer::step` is a dispatcher on `step_impl<RATE>` with the pass in `dct_rate_pass` (`NTC_NOINLINE`), not the plan's
   in-function block: the `/fp:fast /arch:AVX2` build changes the numerics of a function whose code shape changes (the regression pin
   hashed `5fe8b3f8...` with the block inline and `cf5357ba...` with a guarded call inside the loop; `cb884707...` with the template).
   The `RATE = false` instantiation is the original text.
2. The check-4 replica's rate pass is `cuda_check_rate_replica`, called once after the mse loop (the hash noise is regenerable), for
   the same reason. The `dct rate` line compares the device's snapped-plane bits and truncation counts against a *host re-snap of the
   downloaded shadow* (check 7's convention), not against the host state: for a loaded v13 the host is not re-snapped at load while the
   device's lr-0 step re-snaps the reconstructed shadow, which is not idempotent for a few blocks (3 of 4096 bits, 1 truncation on the
   m1 lambda-80 model), so the plan's "snapped-plane proxy bits 0 of N differ, truncated H = D" would have failed on every loaded model.
   The line's clause order is `bit differences, hits; host re-snap ... proxy bits, truncated`.
3. `dct_desc_of(Q, W, H)` builds the `DctDesc` (the three `set_dct` sites needed the new fields; the plan named only `dct_sync_device`).
4. `dct_quantize_block` returns the bits only when `with_bits` (the ES evaluations); the snap lets `dct_recon_block` fill `Q.bits`, so
   the snap path computes the bits once, not twice (the plan: "returns dct_block_bits(sym) when lam16 > 0 || es on").
5. The device had no zigzag table; `ModelDesc::dct_zigzag` uploads the host's `DCT_ZIGZAG` into `c_dct_zigzag` (copied, not recomputed).
6. `Options::dct_rate_given` for the "no effect without --dct-lambda" note.
7. The `dct rate` line's ratio is against the h0 line's AC tokens (run + mag + sign + eob), the tokens the proxy prices, not the whole h0
   (which includes the DC, the codes and the headers); the line prints that figure as `h0 ac (run+mag+sign+eob) /blk`.
8. The `dct[c]` clause is ` proxy P (x/blk) trunc N` (the plan: ` proxy P trunc N`).
9. The progress line's `chg sym / zq` printf was split so ` proxy P` sits after `bits/blk` and ` trunc N hits X%` closes the group.
10. `--cuda-check` on a loaded trained model was added as a verification (the iteration-0 lines have `hits 0`); it is outside the tool's
    tolerance envelope for the model11 decoder (check 1 at 1.03e-5 with the lambda model, check 4 at 1.77e-3 with the HEAD binary on the
    lambda-0 model), so only the m1 loaded-model line counts as a pass.

Not run: `--lat-pairs 8` (not in this plan); a matched-rate q sweep with lambda on every point (the sweep here brackets lambda* once).

### Review fixes (September 7, 2026), applied

Code (every changed line tagged `[DCT]`; `main.cpp` keeps its one `\r\r\n` line, count 1 before and after; 0 added source lines without the tag):

1. `--dct-lambda` refuses a non-finite value: `if (!(o.dct_lambda >= 0.0f) || !std::isfinite(o.dct_lambda))` ->
   `--dct-lambda needs a finite L >= 0 (0 = off)`, rc 1 (`--dct-lambda nan`, `-1` and `1e40` (atof: inf) all print it).
2. The check-4b re-snap in `cuda_check` (`D.lat.z = zs; D.qes_refresh();`) carries a three-line comment: it mutates the host state
   (`D.lat.z`, `D.zq`, `D.dct.sym` / `bits` / truncation counts) before checks 5-7; benign today (5 is excluded under `--dct-q`, 6
   downloads the device shadow afresh, 7 host-snaps first), and a check inserted between 4 and 7 that reads host symbols or `zq` must
   not inherit it.
3. Progress line: `nz X (pre-trunc Y)` (Y = last-snap pre-truncation nonzero ACs per (block, channel)), `proxy P (Rx h0ac)` (R = proxy
   over the h0 run + mag + sign + eob bits), `trunc N/M` (N zeroed of M pre-truncation nonzeros); `trunc off` / `hits off` for the part
   `--dct-rate` leaves out. From the lambda-320 trunc-only run below:
   `iter   8000  mse 156.028  psnr  26.20 dB  best  29.29 | dct30,q8 psnr  26.20 @ 3.436 bpp (ent 2.351, ctx 2.306) | dct nz 4.29 (pre-trunc 8.24) lnz 15.6 eob0 31.2% clamp 2.03% bits/blk 87.7/31.3/28.4 proxy 24.3 (1.00x h0ac) chg sym 9126 zq 170741 trunc 16175/33755 hits off | ...`
4. Switch / refit lines: `; trunc N of M pre-truncation nonzero ACs` (`; trunc off` in `es` mode). The lambda-80 check line now prints
   `...; histogram k0..k15 0 0 294 3802 0 0 0 0 0 0 0 0 0 0 0 0; trunc 165 of 165 pre-truncation nonzero ACs`, the `--dct-rate es`
   line `...; trunc off`, the `--dct-rate trunc` line `...; trunc 821 of 843 pre-truncation nonzero ACs`.
5. `dct rate` line: `ratio proxy/h0 R (eob excluded R2)` with R2 = (proxy - 2 bits x EOB tokens) / (h0 run + mag + sign); `lambda_t16 off`
   / `trunc off` / `hits off` for a part that is off. `--load --iters 0` on the trained models (CPU binary):
   m1 lambda 80: `dct rate lambda 80 mode es+trunc proxy total 58646 /blk 14.32 = 0.2237 bpp | h0 ac (run+mag+sign+eob) /blk 14.56 ratio proxy/h0 0.983 (eob excluded 0.954) | lambda_t16 by k 104.5 ... 0.0003986 | trunc last-snap 0 of 0 (0.0%) | hits 0.0% (of 0)`
   model11 lambda 80: `... /blk 3.38 ratio proxy/h0 1.269 (eob excluded 0.793) ...`; model11 lambda 1280: `... /blk 0.66 ratio proxy/h0 3.644 (eob excluded 0.705) ...`
   (the load `done:` lines are the runs' own, e.g. `26.05 dB at 2.898 bpp raw, 2.184 bpp entropy-coded` and `35.00 dB at 1.208 bpp raw,
   0.927 bpp entropy-coded`); the trunc-only run's final line: `dct rate lambda 320 mode trunc-only proxy total 99662 /blk 24.33 = 0.3802 bpp | h0 ac (run+mag+sign+eob) /blk 24.24 ratio proxy/h0 1.004 (eob excluded 1.023) | lambda_t16 by k 418 181.9 ... 0.001594 | trunc last-snap 16175 of 33755 (47.9%) | hits off`.
6. `cuda/ntc_cuda.cu` `lat_step`: the synchronous per-step `cudaMemcpy` of `evals` is gone; `Impl::dct_evals` (host int, `dct_n * K`
   when part A ran, else 0) is returned by `download_dct_rate`, `d_dct_cnt` is 3 ints (ntrunc, nz_before, hits). No kernel changed: the
   seventeen `--cuda-check` lines below print the figures of Verification 1 / 2 exactly.

Documentation: README (the `--dct-lambda` paragraph: the proxy-tracks-h0 claim qualified to m1 at lambda <= 80 with the EOB-dominated
exception, the starting point `lambda*/4 ~ 7.5 S(q)^2 (20 at q 30, 80 at q 15, 200 at q 10), bracket by 2x`, the new progress-line
clauses; the flag row and `usage()`: "every nonzero AC, walked from zigzag 63 downward" instead of "the trailing ACs"; the check-4
sentence: the pair differences and hits are against the host replica, the proxy bits and truncation counts against a host re-snap of
the downloaded shadow) and this section (the m1 rows of the table re-judged against the new references, Reading items 1-4 rewritten
as above: the trunc-only mechanism and its decomposition, `hits` as 1 - (1 - p)^63, the mean-gain sentence marked a hypothesis, the
model11 h0 decomposition and the low-q recommendation as item 6).

Runs (GPU, the 1-plane line of the sweep: `m1.png --cuda --block 4 --latent 0 0 1 --latent2 0 0 4 --filter nearest,bilinear --pos lv1local
--dct-start 0.25 --dct-refit 500 --qes 0,8 --mlp 27,27 --mlp-pairs 256 --iters 8000 --lr-anneal 0.5 0.05 --mlp-fd 0.5 --leak 0.0009765625
--dct-map all --write-ntcb`, then `build\Release\ntc_decode.exe <out>/model.ntcb -o <out>/ntcb_decoded --compare m1.png`; final binaries):
```
--dct-q 20 -> out_m1_b4_dct20_c4bilin_qes8_mlp27_refit500_start25_cuda8k_fd50_ntcb
iter   2000  dct: scale codes fitted ...; psnr 29.46 -> 23.23 dB; nz 8.03/blk, eob-only 0.2%, clamped 1.10%; histogram k0..k15 0 0 0 0 0 2 2669 1345 80 0 0 0 0 0 0 0
done: final psnr 25.89 dB (best 29.14) at fp32 40.129 bpp | --dct-q 20 level 0 (bit simulator) + --qes 0,8 latent + fp16 mlp: psnr 25.89 dB at 3.117 bpp raw, 2.228 bpp entropy-coded, 2.202 bpp with an (up, left) context on level 0 [level 0 alone: 0.338] [dct: raw 1.051 h0 0.364 ctx 0.338 code raw/ctx 0.062/0.023 bpp; nz 3.02/blk eob0 26.9%] | 50.1s
file: .../model.ntcb 102249 bytes = 3.120 bpp (simulator raw 3.117 bpp); content 816704 bits = simulator 817056 - 352 side-info bits (stored in the header) [OK]; ...; self-check OK
compare t0: m1.png (512x512): PSNR 25.89 dB, max |diff| 123
--dct-q 15 -> out_m1_b4_dct15_c4bilin_qes8_mlp27_refit500_start25_cuda8k_fd50_ntcb
iter   2000  dct: scale codes fitted ...; psnr 29.46 -> 22.87 dB; nz 6.05/blk, eob-only 1.2%, clamped 0.97%; histogram (the same)
done: final psnr 25.60 dB (best 29.26) at fp32 40.129 bpp | --dct-q 15 level 0 (bit simulator) + --qes 0,8 latent + fp16 mlp: psnr 25.60 dB at 2.786 bpp raw, 2.130 bpp entropy-coded, 2.110 bpp with an (up, left) context on level 0 [level 0 alone: 0.241] [dct: raw 0.720 h0 0.260 ctx 0.241 code raw/ctx 0.062/0.022 bpp; nz 1.69/blk eob0 45.4%] | 31.3s
file: .../model.ntcb 91393 bytes = 2.789 bpp (simulator raw 2.786 bpp); content 729856 bits = simulator 730208 - 352 side-info bits (stored in the header) [OK]; ...; self-check OK
compare t0: m1.png (512x512): PSNR 25.60 dB, max |diff| 131
--dct-q 30 --dct-rate trunc --dct-lambda 320 -> out_m1_b4_dct30_c4bilin_qes8_mlp27_refit500_start25_lam320_trunc_cuda8k_fd50_ntcb
iter   2000  dct: scale codes fitted ...; psnr 29.46 -> 23.80 dB; nz 9.84/blk, eob-only 2.1%, clamped 1.23%; histogram (the same); trunc 5553 of 45871 pre-truncation nonzero ACs
refit lines: trunc 8481 of 25389, 14344 of 34832, 15329 of 36065, 13975 of 32511, 14425 of 35821, 15265 of 33667, 15711 of 34471, 15525 of 33812, 15746 of 33795, 15734 of 33819, 15870 of 33606 pre-truncation nonzero ACs
done: final psnr 26.20 dB (best 29.29) at fp32 40.129 bpp | --dct-q 30 level 0 (bit simulator, lambda 320 trunc-only) + --qes 0,8 latent + fp16 mlp: psnr 26.20 dB at 3.436 bpp raw, 2.351 bpp entropy-coded, 2.306 bpp with an (up, left) context on level 0 [level 0 alone: 0.444] [dct: raw 1.370 h0 0.488 ctx 0.444 code raw/ctx 0.062/0.021 bpp; nz 4.29/blk eob0 31.2%] | 51.7s
file: .../model.ntcb 112697 bytes = 3.439 bpp (simulator raw 3.436 bpp); content 900288 bits = simulator 900640 - 352 side-info bits (stored in the header) [OK]; ...; self-check OK
compare t0: m1.png (512x512): PSNR 26.20 dB, max |diff| 123
```
The switch histogram of all three is the control's (`0 0 0 0 0 2 2669 1345 80`: the probe does not see q or lambda). Re-judging the
m1 1-plane lambda points between q 15 (25.60 @ 2.130), q 20 (25.89 @ 2.228) and q 30 (26.27 @ 2.425) on the (order-0 bpp, PSNR)
plane: lambda 20 26.34 @ 2.308 vs 26.044 on the curve (+0.30 dB); lambda 80 26.05 @ 2.184 vs 25.760 (+0.29); 80 es-only 26.09 @ 2.218
vs 25.860 (+0.23); 80 trunc-only 26.27 @ 2.412 vs 26.245 (+0.03); 320 25.33 @ 2.015 vs 25.260 by the q 15 -> 20 slope (+0.07,
extrapolated: 2.015 is below q 15); 320 trunc-only 26.20 @ 2.351 vs 26.127 (+0.07). The table above carries these values. Run 10 at
lambda 320 trunc-only: 8.24 pre-truncation nonzero ACs per block (33755 / 4096) against the control's 5.76 and the lambda-80 trunc-only
run's 6.41, with `nz 4.29` after truncation: the harder B truncates, the more coefficients the mse ES pushes past the moved edge, which
is the "B only relocates the edge" explanation of Reading item 1 (plan risk 2), not "edge-hugging coefficients that the shadow re-grows".
Its PSNR / rate (26.20 @ 2.351, +0.07 dB on the curve) also says B alone is worth about what changing q is worth, and A is the gain.

Verification (final binaries, both builds 0 compiler warnings, the CUDA link's pre-existing `LNK4098` only):
```
build\Release\ntc.exe --crop 512 --mlp-pairs 32 --iters 200 --print-every 200 --save-every 200 --out out_reg
iter    200  mse 269.314  psnr  23.83 dB  best  23.83 | q8 psnr  23.83 @ 0.552 bpp (ent 0.500, ctx 0.084) | mlp batch 0.00442 dstd 9.65e-05 | lat mean -0.051 sd 0.551 max 2.08 | 10.6s (18.90 it/s)
done: final psnr 23.83 dB (best 23.83) at fp32 2.103 bpp | 8-bit latent + fp16 mlp: psnr 23.83 dB at 0.552 bpp raw, 0.500 bpp entropy-coded, 0.084 bpp with an (up, left) context on level 0 [level 0 alone: 0.032] | 10.8s
sha256 out_reg/model.bin cb88470777e1d041c8803bcf3ca531434b9a2b2b5bce2ef88e031fbc86d69600
```
`--dct-selftest` on both builds: `diff`-identical, 12 PASS lines, `dct-selftest: all passed`, rc 0 (the three rate items as in
Verification 1). The seventeen `--cuda-check` lines (the seven README general / DCT / 2-channel lines, the `--dct-q 90` / `20`
variants, the two README lambda lines, the 2-channel `--dct-q 50` and 4-channel `--dct-q 30,50,70,90` variants, `--dct-lambda 100000`,
the loaded m1 lambda-80 model, and `--dct-rate es` / `--dct-rate trunc` on the m1 1-plane layout): every one `cuda-check: all
passed`, rc 0. Figures: decode 1.788e-07 / 1.788e-07 / 1.192e-07 x 7 / 1.192e-07 (lambda 80) / 1.192e-07 (2-ch lambda 80) / 1.192e-07
/ 1.192e-07 / 1.192e-07 (1e5) / 4.172e-07 (loaded) / 1.788e-07 / 1.788e-07; latent ES grad 2.174e-05 / 7.003e-06 / 4.645e-05 /
3.884e-05 / 3.669e-05 / 3.305e-05 / 3.974e-05 / 3.304e-05 / 2.095e-05 / 3.304e-05 / 2.094e-05 / 2.095e-05 / 2.382e-05 (4 ch) /
3.304e-05 / 5.150e-05 / 1.023e-05 / 1.024e-05 (= Verification 1 and 2 where those lines exist); qat search `3 of 524288` / `3 of
786432` / `1 of 262144`; qes snap 0 / 0; dct snap 0 / 0 (max 0 ulp) / 0 before and after the step on every DCT line; dct refit moved
195 / 193 / 196 / 5 / 196 / 5 / 5 / 1931 (of 16384, 4 ch) / 196 / 479 / 82 / 84; the `dct rate` lines:
```
[lambda 80]      dct rate        : pair 3 bit differences 0 of 4096 differ, hits 0 = 0 (of 16384 = 16384); host re-snap of the downloaded shadow vs k_dct_snap: proxy bits 0 of 4096 differ, truncated 165 = 165 (of 165 = 165 nonzero)  PASS
[2 ch lambda 80] dct rate        : pair 3 bit differences 0 of 8192 differ, hits 0 = 0 (of 32768 = 32768); host re-snap of the downloaded shadow vs k_dct_snap: proxy bits 0 of 8192 differ, truncated 173 = 173 (of 173 = 173 nonzero)  PASS
[lambda 1e5]     dct rate        : pair 3 bit differences 0 of 4096 differ, hits 0 = 0 (of 16384 = 16384); ... proxy bits 0 of 4096 differ, truncated 165 = 165 (of 165 = 165 nonzero)  PASS
[loaded lam80]   dct rate        : pair 3 bit differences 0 of 4096 differ, hits 1132 = 1132 (of 16384 = 16384); ... proxy bits 0 of 4096 differ, truncated 1 = 1 (of 8778 = 8778 nonzero)  PASS
[--dct-rate es]  dct rate        : pair 3 bit differences 0 of 4096 differ, hits 5501 = 5501 (of 16384 = 16384); ... proxy bits 0 of 4096 differ, truncated 0 = 0 (of 0 = 0 nonzero)  PASS
[--dct-rate trunc] no dct rate line (part A off); dct snap / dct refit ...; proxy bits 0 of 8192 differ  PASS
```
(`evals` = 16384 / 32768 on every line is now the host figure.) The strongest firewall, the 8000-iteration 2-plane run without
lambda (the Verification 1 command, `--out out_rate_fw2`):
```
fc /b out_rate_fw2\model.bin out_m1_b4_dct30x30_c4bilin_qes8_mlp27_refit500_start25_cuda8k_fd50_ntcb\model.bin -> FC: no differences encountered
fc /b out_rate_fw2\model.ntcb ...\model.ntcb -> FC: no differences encountered
```

Firewall gaps named by the review, closed or recorded:
- A lambda-0 v13 `--load --iters 0` on the new binary (`out_m1_b4_dct30x30_c4bilin_qes8_mlp27_refit500_start25_cuda8k_fd50_ntcb/model.bin`
  with its options `--block 4 --latent 0 0 2 --latent2 0 0 4 --filter nearest,bilinear --pos lv1local --dct-q 30,30 --qes 0,8 --mlp 27,27
  --leak 0.0009765625`), CPU and `--cuda`, both:
  `done: final psnr 26.82 dB (best -1.00) at fp32 72.132 bpp | --dct-q 30,30 level 0 (bit simulator) + --qes 0,8 latent + fp16 mlp: psnr 26.82 dB at 4.974 bpp raw, 2.840 bpp entropy-coded, 2.765 bpp with an (up, left) context on level 0 [level 0 alone: 0.919] [dct: raw 2.906 h0 0.993 ctx 0.919 code raw/ctx 0.125/0.048 bpp; nz 4.63/blk eob0 12.5%] | 0.3s`
  = the log's `done:` line (26.82 dB at 4.974 / 2.840 / 2.765, level 0 alone 0.919, nz 4.63 eob0 12.5%); no `dct rate` line, no lambda clause.
- CPU lambda 0 with C = 4: `chief1.png --rng hash --iters 100 --print-every 20 --mlp-pairs 32 --block 8 --latent 0 0 4 --latent2 0 0 4
  --filter nearest,bilinear --pos lv1local --dct-q 30,50,70,90 --dct-start 0.5 --qes 0,8 --mlp 17,17 --leak 0.0009765625` on the new CPU
  binary and on the HEAD binary kept in the scratchpad (`ntc_run2/ntc.exe`: no `--dct-lambda` in its usage, reproduces the pin
  `cb884707...`): `model.bin` `cmp`-identical, the logs identical apart from the timing fields; both
  `iter     50  dct: scale codes fitted ... (mean gain ch 0.68 / 1.16 / 0.75 / 1.33); psnr 19.04 -> 18.35 dB; nz 8.98/blk, eob-only 7.1%, clamped 0.02%; histogram k0..k15 0 46 2963 5025 6362 1988 0 ...` and
  `done: final psnr 26.40 dB (best 26.40) at fp32 130.067 bpp | --dct-q 30,50,70,90 level 0 (bit simulator) + --qes 0,8 latent + fp16 mlp: psnr 26.40 dB at 12.790 bpp raw, 5.039 bpp entropy-coded, 4.444 bpp with an (up, left) context on level 0 [level 0 alone: 3.931] [dct: raw 12.254 h0 4.527 ctx 3.931 code raw/ctx 0.250/0.070 bpp; ...]`.
- Not closed, pre-existing: with T > 1 textures the DCT level, and so both parts of `--dct-lambda`, address texture 0 only (the rate
  kernels read `c_tex[0]`, the host pass `D.lat.lv[0]`); the flag is documented for the 1-texture layouts it was run on.

### `--dct-lambda-lo W` (September 7, 2026, evening): protecting the first-order pair

Richard saw blocking on low-variance blocks at lambda 60 on the 8x8-cell model11 run and asked for a way to keep
`--dct-lambda` from hitting the two lowest ACs so hard (XUBC7 strongly protects them at every quality). Chosen design:
the (0,1) and (1,0) tokens (zigzag 1 and 2) cost `W` x their bits in the proxy, in both the ES term and the truncation
(`dct_lo_weight` / `dev_dct_lo_weight`: `(token * round(16 W) + 8) >> 4`, integer, both backends; `DctLevel::lo_w16`,
`DctDesc::lo_w16`), default W = 0.25. W = 1 is the unweighted proxy: `--dct-lambda 25 --dct-lambda-lo 1` on the 8x8-cell
q 30 line reproduced `out_model11_b8_dct30_c4bilin_qes8_mlp17_refit500_lam25_cuda8k_fd50_ntcb/model.bin` and `.ntcb`
byte for byte (`cmp`). Regression pin `psnr 23.83 dB`, sha256 `cb884707...` unchanged; `dct-selftest: all passed`;
`--cuda-check` with `--dct-lambda 80` (default W: `dct rate ... 0 of 4096 differ, hits 76 = 76 ... PASS`), with
`--dct-q 50,20 --dct-lambda 80 --dct-lambda-lo 0`, and the lambda-0 DCT line: `all passed`.

model11 q 30, 17,17 decoder, container-decoded PSNR (`ntcb_decoded.png` in each dir; dirs `..._lam<L>_lo25_...`):

| cells | lambda | W = 1 (unweighted) | W = 0.25 |
|---|---|---|---|
| 8x8 | 25 | 36.95 dB at 0.745 bpp order-0 (1.228 file), nz 1.69 | 37.03 dB at 0.767 (1.311 file), nz 2.02 |
| 8x8 | 60 | 36.48 at 0.700 (1.090), nz 1.13 | 36.64 at 0.725 (1.184), nz 1.51 |
| 4x4 | 25 | 38.51 at 1.864 (2.544), nz 0.95 | 38.58 at 1.890 (2.643), nz 1.34 |
| 4x4 | 60 | 38.28 at 1.847 (2.463), nz 0.63 | 38.34 at 1.872 (2.538), nz 0.93 |

The protected pair costs 0.02-0.03 bpp order-0 and returns 0.06-0.16 dB; whether it removes the blocking is judged by eye.
Removal: the option, the validation line, the usage lines, `DctLevel::lo_w16`, `dct_lo_weight` and its two uses, the `lo16`
parameter of `dct_block_bits` / `dct_truncate_block` / `dct_quantize_block` and their calls, `dct_desc_of`'s fill, the banner
clause, the done-line `lo N/16` label; CUDA: `DctDesc::lo_w16`, `dev_dct_lo_weight` and its two uses; the README row.
- `--dct-deadzone` (section 9, September 7, 2026; every line tagged `[DCT]`, the replaced ones with `hook:`): `Options::dct_deadzone`,
  the parser line, the two usage lines, the validation line, `--dct-deadzone` in the ignored-options note; `DctLevel::deadzone` and its
  fill in the level setup; the `dz` parameter of `dct_quant_ac` / `dct_dequant_ac` / `dct_dequant_block` / `dct_truncate_block` /
  `dct_quantize_block` and every call (`dct_rate_block`, `dct_recon_block`, `dct_snap_block`, the self-test's `true` arguments); the
  `plain rounding (dz 0)` self-test item; `save_model` (magic `0x4E54433E`, the `dz` int, the v14 comment); the loader (the v14 magic,
  `v14`, `saved_dct_dz`, its read, the `dct_ok` clause, the `--dct-deadzone 0` mismatch clause, the `v9..v14` message, the layout
  comment); `NTCB_VERSION = 2`, `NtcbLevel::deadzone`, the `dct_flags` byte in `ntcb_write_header` / `ntcb_read_header` (the version
  test, the reserved-bits refusal), the writer's `N.deadzone` fill and `ntcb_restore`'s clause; `dct_desc_of`'s fill; the banner's
  `AC quantizer %s` clause; the done-line ` dz0`; the `--resave` v14 text. CUDA: `DctDesc::deadzone`, the `!c_dct.deadzone ||` in
  `dev_dct_quant_ac` / `dev_dct_dequant_ac`, `dd.deadzone = 1` in `init`. ntc_decode: `Model::dct_deadzone`, the v14 range and read,
  the `dz` parameter of `dct_dequant_ac` and its call, `NtcbLevel::deadzone`, the version test and the `dct_flags` read, `load_ntcb`'s
  fill, the level-0 line's `plain-rounding ACs (dz 0)` clause. README: the flag row, the quantizer sentence, the v14 / ntcb v2 notes,
  the check line and sentence. NTCB_PLAN.md: the version-2 amendment and the two table lines. Restoring v13 / ntcb v1 as the written
  formats = drop the int / the byte and the version bump; the readers' v13 / v1 branches are the pre-change code.


## 9. `--dct-deadzone` (September 7, 2026)

Richard suspects the dead zone's zero bin (`|d| < L -> 0`) and its 1.5-step reconstruction of `|q| = 1` cause the
mosquito artifacts he sees at q 50-80. `--dct-deadzone 0|1` (default 1 = the XUASTC dead-zone quantizer, alpha 0.5, with
the first-order exemption; byte-identical behaviour) selects the AC quantizer: with 0 every AC is quantized and
dequantized exactly as the first-order pair is today, `roundf(d / L)` on the host / `roundf(__fdiv_rn(d, L))` on the
device with the +-1024 clamp, dequantized as `q * L`. The decoder must know which quantizer produced the symbols, so the
flag is in both formats: model file **v14** (magic `0x4E54433E`; the v13 layout plus one int `dct_deadzone` right after
the `int di[3] = {N, dc_step, C}` header ints and before the per-channel q ints; written only when the plane is live,
as v13 was; both loaders accept v13 (dead zone 1 implied) and v14; the option-mismatch check compares the flag and its
message prints `--dct-deadzone 0` when the file has 0) and **ntcb version 2** (a mode-3 level record gains one byte
`dct_flags`, bit 0 = dead zone, bits 1..7 reserved and refused, right after `dc_step` and before the q bytes; both
readers accept version 1 (dead zone implied, no flags byte) and 2; the writer always writes 2; NTCB_PLAN.md 1.1 has the
dated amendment). Plumbing: `Options::dct_deadzone` (parser, two usage lines, validation 0/1, the ignored-options note),
`DctLevel::deadzone`, `DctDesc::deadzone` (uploaded with `set_dct`; `dev_dct_quant_ac` / `dev_dct_dequant_ac` read
`c_dct.deadzone`), ntc_decode `Model::dct_deadzone`; the host `dct_quant_ac` / `dct_dequant_ac` take a `bool dz`
parameter, threaded through `dct_dequant_block` / `dct_truncate_block` / `dct_quantize_block` from `Q.deadzone` at the
three call sites (`dct_rate_block`, `dct_recon_block`, `dct_snap_block`) and `true` in the self-test (the fast-sweep
path and `dct_probe` reach the quantizer only through `dct_snap_level0`, so they follow the level's flag). The banner's
`dct      :` line names the quantizer (`AC quantizer dead zone alpha 0.5 with the first-order exemption` /
`AC quantizer plain rounding on every AC (--dct-deadzone 0)`); the done line's label gains ` dz0` when off
(`--dct-q 80 dz0 level 0 (bit simulator, ...)`); ntc_decode's level-0 line gains `, plain-rounding ACs (dz 0)`; the
`--resave` line says v14. With plain rounding the rate proxy's cost model and the truncation are unchanged (they see
symbols); the truncation's `dD = c^2 - (c - v)^2` uses the plain dequantization automatically because `v` comes from
`dct_dequant_ac(..., dz)`. The self-test gains one item, `plain rounding (dz 0)`, and keeps every existing line unchanged.

### Builds and the self-test
`cmake --build build --config Release` and `cmake --build build_cuda --config Release`: 0 compiler warnings (the CUDA
link prints only the pre-existing `LNK4098` defaultlib warning). `--dct-selftest` on both builds: `diff`-identical, rc 0,
every pre-change line byte-identical (`diff` against the previous self-test output shows only the added line):
```
  dead zone                    PASS  (1,0): 0.4L->0 0.6L->1; (2,0): 0.6L->0 0.99L->0 1.0L->1 1.1L->1; (0,2): -1.5L->-1; dequant 1@(2,0) = 1.5L, 3@(1,0) = 3.0L, -2@(2,3) = -2.5L
  plain rounding (dz 0)        PASS  (2,0): 0.4L->0 0.6L->1 1.49L->1 1.5L->2 -0.6L->-1; (0,2): -1.5L->-2; clamp 2000L->1024 -2000L->-1024; odd symmetry yes; dequant 1@(2,0) = 1.0L, 3@(1,0) = 3.0L, -2@(2,3) = -2.0L
dct-selftest: all passed
```
(13 PASS lines; the odd-symmetry clause sweeps every AC position at 61 inputs from -3L to 3L.)

### Firewall (default 1)
```
build\Release\ntc.exe --crop 512 --mlp-pairs 32 --iters 200 --print-every 200 --save-every 200 --out out_reg
iter    200  mse 269.314  psnr  23.83 dB  best  23.83 | q8 psnr  23.83 @ 0.552 bpp (ent 0.500, ctx 0.084) | mlp batch 0.00442 dstd 9.65e-05 | lat mean -0.051 sd 0.551 max 2.08 | 7.6s (26.16 it/s)
sha256 out_reg/model.bin: cb88470777e1d041c8803bcf3ca531434b9a2b2b5bce2ef88e031fbc86d69600 (magic 3c43 544e, v12)
```
The README's nine `ntc ... --cuda-check` lines (run with `build_cuda\Release\ntc.exe`) plus the `--dct-q 90` and `--dct-q 20`
variants of the DCT line: every one `cuda-check: all passed`, rc 0 (6 / 5 / 6 / 6 / 5 / 7 / 7 / 8 / 8 / 7 / 7 PASS lines). The
DCT line at q 50 / 90 / 20: `dct snap : symbols 0 of 262144 differ, plane 0 of 262144 values differ (max 0 ulp), codes 0 of
4096 differ ...; after a latent step with lr 0.02: 0 / 0 (max 0 ulp) / 0 ... PASS`, latent ES grad 3.305e-05 / 3.974e-05 /
3.304e-05, decode 1.192e-07, the switch lines `nz 0.04/blk, eob-only 96.0%` / `nz 3.95/blk, eob-only 1.4%` / `nz 0.00/blk,
eob-only 100.0%` (= section 3).

The v13 file `out_model16_b8_dct80_c4_mlp27_lam2_8k/model.bin`, `--load --iters 0` on CPU and `--cuda` (`model16.png --block 8
--latent 0 0 1 --latent2 0 0 4 --filter nearest,bilinear --pos lv1local --dct-q 80 --dct-refit 500 --qes 0,8 --mlp 27,27 --leak
0.0009765625 --dct-lambda 2`), both:
```
dct      : level 0 = 8x8 DCT-coded selector plane, q 80 (JPEG luma table K.1, libjpeg scaling), DC step 4 (uniform, independent of q), AC quantizer dead zone alpha 0.5 with the first-order exemption, per-block 4-bit scale code from the decoder sensitivity probe; ... (codes restored from the file, frozen); ...
done: final psnr 37.63 dB (best -1.00) at fp32 34.083 bpp | --dct-q 80 level 0 (bit simulator, lambda 2 lo 4/16) + --qes 0,8 latent + fp16 mlp: psnr 37.63 dB at 3.373 bpp raw, 1.651 bpp entropy-coded, 1.557 bpp with an (up, left) context on level 0 [level 0 alone: 1.059] [dct: raw 2.831 h0 1.153 ctx 1.059 code raw/ctx 0.062/0.018 bpp; nz 10.14/blk eob0 2.8%] | 0.4s
```
= the log's done line (37.63 dB at 3.373 / 1.651 / 1.557). `ntc_decode` on that `model.bin` (`model : ... (v13)`) and on its
`model.ntcb` (`ntcb v1`; `file : ... 172798 bytes = 3.375 bpp ... content 1381099 bits matches the simulator exactly`): both
PNGs `cmp`-identical to `ntcb_decoded.png` in that directory.

A fresh 200-iteration GPU run with the default (`model16.png --cuda --block 8 --latent 0 0 1 --latent2 0 0 4 --filter
nearest,bilinear --pos lv1local --dct-q 80 --dct-refit 500 --dct-start 0.5 --qes 0,8 --mlp 27,27 --leak 0.0009765625 --dct-lambda 2
--iters 200 --print-every 200 --save-every 200 --write-ntcb`, scratchpad `fresh_dz1`) writes v14 (`3e43 544e`) and ntcb version 2
(`0200` at offset 4):
```
file: .../fresh_dz1/model.ntcb 184928 bytes = 3.612 bpp (simulator raw 3.610 bpp); content 1478127 bits = simulator 1478479 - 352 side-info bits (stored in the header) [OK]; container header 1048 + section headers 192 + sync 42 + padding 15 bits; ...; self-check OK
done: final psnr 26.31 dB (best 26.31) at fp32 34.083 bpp | --dct-q 80 level 0 (bit simulator, lambda 2 lo 4/16) + --qes 0,8 latent + fp16 mlp: psnr 26.31 dB at 3.610 bpp raw, 1.673 bpp entropy-coded, 1.535 bpp with an (up, left) context on level 0 [level 0 alone: 1.018] [dct: raw 3.067 h0 1.156 ctx 1.018 code raw/ctx 0.062/0.019 bpp; nz 11.08/blk eob0 0.0%] | 1.1s
```
(the header grew by the flags byte: 1048 bits against 1040 for the v1 files.) `--load --iters 0` of `model.bin` and of
`model.ntcb`, CPU and `--cuda`: all four print the same done line (26.31 dB at 3.610 / 1.673 / 1.535, level 0 alone 1.018, nz
11.08). `ntc_decode` on both (`(v14)` / `(ntcb v2)`, `content 1478127 bits matches the simulator exactly`): the two PNGs
`cmp`-identical. The same run with an explicit `--dct-deadzone 1`: `model.bin` and `model.ntcb` byte-identical to the default
run's (`cmp`). Loading the v14 dead-zone file with `--dct-deadzone 0`: `... was saved with --latent 800 512 1 --latent2 100 64 4
--filter nearest,bilinear --leak 0.000977 --qes 0,8 --dct-q 80 --dct-dc-step 4 --mlp 27,27 --pos lv1local and 1 texture(s); pass
the same options`, rc 1. `--dct-deadzone 2`: `--dct-deadzone: 0 (plain rounding on every AC) or 1 (the dead-zone quantizer)`,
rc 1; `--dct-deadzone 0` without `--dct-q`: the ignored-options note now lists `--dct-deadzone`.

### With `--dct-deadzone 0`
The DCT `--cuda-check` line with `--dct-deadzone 0` (rc 0, 7 PASS):
```
iter      0  dct: scale codes fitted ... psnr 11.28 -> 11.28 dB; nz 0.12/blk, eob-only 88.9%, clamped 0.00%; histogram k0..k15 0 0 294 3802 0 0 0 0 0 0 0 0 0 0 0 0
  decode          : max |gpu - cpu| = 1.192e-07  PASS
  mlp ES dl       : max |gpu - cpu| / rms(dl) = 4.206e-07 over 8 pairs  PASS
  mlp FD dl       : max |gpu - cpu| / rms(dl) = 8.607e-05 over 24 weights  PASS
  latent ES grad  : max |gpu - cpu| / rms(grad) = 4.519e-05 over 278528 values  PASS
  qes snap        : device snapped copy vs host snap of the downloaded shadow: 0 of 278528 values differ, 0 after a latent step with lr 0.02  PASS
  dct snap        : symbols 0 of 262144 differ, plane 0 of 262144 values differ (max 0 ulp), codes 0 of 4096 differ (device plane from k_dct_recon on the host's symbols: IDCT + clamp parity); after a latent step with lr 0.02: 0 / 0 (max 0 ulp) / 0 (k_dct_snap: forward DCT + quantizer + IDCT parity)  PASS
  dct refit       : host re-probe moved 191 of 4096 codes (>= 2: 0), uploaded through set_dct: symbols 0 differ, plane 0 (max 0 ulp), codes 0 (k_dct_recon with the refitted codes); after a latent step: 0 / 0 (max 0 ulp) / 0 (k_dct_snap quantizes with the refitted codes)  PASS
cuda-check: all passed
```
(nz 0.12/blk at the fit against 0.04 with the dead zone on the same `N(0, 0.1)` plane: the zero bin is gone.) The 2-channel
`--dct-q 50,20 --dct-lambda 80` line with `--dct-deadzone 0` (rc 0, 8 PASS):
```
iter      0  dct: ... (mean gain ch 0.77 / 0.85); psnr 11.50 -> 11.48 dB; nz 0.00/blk, eob-only 99.9%, clamped 0.00%; histogram k0..k15 0 0 0 8187 5 0 ...; trunc 481 of 489 pre-truncation nonzero ACs
  dct rate        : pair 3 bit differences 0 of 8192 differ, hits 107 = 107 (of 32768 = 32768); host re-snap of the downloaded shadow vs k_dct_snap: proxy bits 0 of 8192 differ, truncated 481 = 481 (of 489 = 489 nonzero)  PASS
  dct snap        : symbols 0 of 524288 differ, plane 0 of 524288 values (2 channels) differ (max 0 ulp), codes 0 of 8192 differ (...); after a latent step with lr 0.02: 0 / 0 (max 0 ulp) / 0 (...); proxy bits 0 of 16384 differ (--dct-lambda, both compares)  PASS
  dct refit       : host re-probe moved 5 of 8192 codes (>= 2: 0), ...: symbols 0 differ, plane 0 (max 0 ulp), codes 0 (...); after a latent step: 0 / 0 (max 0 ulp) / 0 (...); proxy bits 0 of 16384 differ  PASS
cuda-check: all passed
```
(decode 1.192e-07, mlp ES 2.958e-07, FD 4.309e-05, latent ES grad 2.094e-05 over 589824 values.)

A fresh 200-iteration dz0 GPU run (the `fresh_dz1` command with `--dct-deadzone 0`, scratchpad `fresh_dz0`; v14 `3e43 544e`, ntcb `0200`):
```
dct      : level 0 = 8x8 DCT-coded selector plane, q 80 (...), DC step 4 (...), AC quantizer plain rounding on every AC (--dct-deadzone 0), per-block 4-bit scale code ...
file: .../fresh_dz0/model.ntcb 247123 bytes = 4.827 bpp (simulator raw 4.824 bpp); content 1975692 bits = simulator 1976044 - 352 side-info bits (stored in the header) [OK]; container header 1048 + section headers 192 + sync 42 + padding 10 bits; ...; self-check OK
done: final psnr 26.22 dB (best 26.22) at fp32 34.083 bpp | --dct-q 80 dz0 level 0 (bit simulator, lambda 2 lo 4/16) + --qes 0,8 latent + fp16 mlp: psnr 26.22 dB at 4.824 bpp raw, 2.013 bpp entropy-coded, 1.828 bpp with an (up, left) context on level 0 [level 0 alone: 1.311] [dct: raw 4.282 h0 1.496 ctx 1.311 code raw/ctx 0.062/0.019 bpp; nz 15.95/blk eob0 0.0%] | 1.1s
```
`--load --iters 0` with `--dct-deadzone 0` of `model.bin` (CPU and `--cuda`) and of `model.ntcb` (CPU): the same done line
(26.22 dB at 4.824 / 2.013 / 1.828, `dz0` label). Without the flag: `.../fresh_dz0/model.bin was saved with --latent 800 512 1
--latent2 100 64 4 --filter nearest,bilinear --leak 0.000977 --qes 0,8 --dct-q 80 --dct-dc-step 4 --dct-deadzone 0 --mlp 27,27 --pos
lv1local and 1 texture(s); pass the same options`, rc 1.
```
build\Release\ntc_decode.exe .../fresh_dz0/model.bin --compare .../fresh_dz0/recon_q_final.png --verify
model    : .../fresh_dz0/model.bin (v14)
level 0  : 800x512x1 nearest, dct 8x8, q 80, dc step 4, plain-rounding ACs (dz 0), 4-bit scale codes, 15.9 nonzero ACs/block, 0.0% EOB-only, fp32 planar
compare t0: .../fresh_dz0/recon_q_final.png (800x510): PSNR 90.20 dB, max |diff| 1, |diff| histogram: 0: 1223924 (99.9938%), 1: 76 (0.0062%), 2: 0 (0.0000%), >2: 0 (0.0000%)
verify   : pre-nonlinearity: max 0 ulp (0 of 1228800 values differ); output fp32: max 42 ulp, max |diff| 7.75e-07; RGB8 mismatches: 76 of 1228800 (0.00618%)
build\Release\ntc_decode.exe .../fresh_dz0/model.ntcb --compare ...: model (ntcb v2); the same level 0 line; file : ... 247123 bytes = 4.827 bpp (...; content 1975692 bits matches the simulator exactly; ...); compare PSNR 90.20 dB, max |diff| 1; the two PNGs cmp-identical
```

Corruption script (`ntcb_corrupt.py` in the scratchpad; its header parser now reads the version and skips the `dct_flags` byte
of a version-2 mode-3 record, its old `version 2` case became `version 3`, and a new case sets a reserved bit of `dct_flags`):
on the dz1 v2 container (`fresh_dz1/model.ntcb`, dct level 0) and on the dz0 one: 30 cases each, every one `ntc_decode rc=1`
and `ntc rc=1` with a named message, `ALL CASES EXIT 1: True` twice. The new lines:
```
level 0 dct_flags reserved bit 1 set (re-hashed)   ntc_decode rc=1: ...: level 0: dct_flags 3 has reserved bits set (only bit 0, the dead zone, is defined)   ntc rc=1: corrupt or truncated ntcb file: level 0: dct_flags 3 has reserved bits set (...)
version 3                                          ntc_decode rc=1: ...: ntcb version 3 (this build reads versions 1 and 2)                                        ntc rc=1: corrupt or truncated ntcb file: ntcb version 3 (this build reads versions 1 and 2)
```

### Experiment: plain rounding against the dead zone, model16, 8000 iterations (GPU)
`build_cuda\Release\ntc.exe model16.png --cuda --block 8 --latent 0 0 1 --latent2 0 0 4 --filter nearest,bilinear --pos lv1local
--dct-q Q --dct-refit 500 --qes 0,8 --mlp 27,27 --mlp-pairs 256 --iters 8000 --lr-anneal 0.5 0.05 --mlp-fd 0.5 --leak 0.0009765625
--dct-map all --dct-lambda L --dct-deadzone 0 --write-ntcb --out out_model16_b8_dctQ_c4_mlp27_lamL_dz0_8k` for (Q, L) = (50, 10) and
(80, 2), then `build\Release\ntc_decode.exe <out>/model.ntcb -o <out>/ntcb_decoded --compare model16.png` (the container rule);
the dead-zone runs are the existing `out_model16_b8_dct50_c4_mlp27_lam10_8k` and `out_model16_b8_dct80_c4_mlp27_lam2_8k` (same
command without the flag). Every figure from the logs (`decoder PSNR` = ntc_decode's `compare` line on the container).

| run (C:\dev\neural\...) | quantizer | decoder PSNR | container | order-0 bpp | level 0 order-0 | nz/blk | eob0 |
|---|---|---|---|---|---|---|---|
| `out_model16_b8_dct50_c4_mlp27_lam10_8k` | dead zone | 35.95 dB | 99681 bytes, 1.947 bpp | 1.053 | 0.556 | 4.42 | 12.3% |
| `out_model16_b8_dct50_c4_mlp27_lam10_dz0_8k` | plain rounding | 36.85 dB | 113449 bytes, 2.216 bpp | 1.132 | 0.634 | 5.50 | 12.0% |
| `out_model16_b8_dct80_c4_mlp27_lam2_8k` | dead zone | 37.63 dB | 172798 bytes, 3.375 bpp | 1.651 | 1.153 | 10.14 | 2.8% |
| `out_model16_b8_dct80_c4_mlp27_lam2_dz0_8k` | plain rounding | 38.42 dB | 207987 bytes, 4.062 bpp | 1.838 | 1.337 | 12.89 | 2.7% |

The dz0 done lines and switch lines:
```
q 50 dz0: iter   4000  dct: ... psnr 38.68 -> 36.37 dB; nz 8.42/blk, eob-only 2.9%, clamped 4.97%; histogram k0..k15 219 27 44 68 92 176 737 3034 ...   (dead zone: 38.68 -> 35.56 dB; nz 5.94/blk, eob-only 3.4%, clamped 4.40%)
          done: final psnr 36.85 dB (best 38.52) at fp32 34.083 bpp | --dct-q 50 dz0 level 0 (bit simulator, lambda 10 lo 4/16) + --qes 0,8 latent + fp16 mlp: psnr 36.85 dB at 2.214 bpp raw, 1.132 bpp entropy-coded, 1.083 bpp with an (up, left) context on level 0 [level 0 alone: 0.586] [dct: raw 1.671 h0 0.634 ctx 0.586 code raw/ctx 0.062/0.014 bpp; nz 5.50/blk eob0 12.0%] | 34.8s
          file: out_model16_b8_dct50_c4_mlp27_lam10_dz0_8k/model.ntcb 113449 bytes = 2.216 bpp (simulator raw 2.214 bpp); content 906299 bits = simulator 906651 - 352 side-info bits (stored in the header) [OK]; ...; self-check OK
          ntcb_decode.log: compare t0: model16.png (800x510): PSNR 36.85 dB, max |diff| 83; level 0 ... plain-rounding ACs (dz 0) ... 5.5 nonzero ACs/block, 12.0% EOB-only; idct 1.93 ms (8.8 MAC/texel, nnz 35189)
q 80 dz0: iter   4000  dct: ... psnr 38.68 -> 37.58 dB; nz 14.74/blk, eob-only 1.9%, clamped 5.68%   (dead zone: 38.68 -> 37.01 dB; nz 10.39/blk, eob-only 1.9%, clamped 5.00%)
          done: final psnr 38.42 dB (best 38.65) at fp32 34.083 bpp | --dct-q 80 dz0 level 0 (bit simulator, lambda 2 lo 4/16) + --qes 0,8 latent + fp16 mlp: psnr 38.42 dB at 4.060 bpp raw, 1.838 bpp entropy-coded, 1.727 bpp with an (up, left) context on level 0 [level 0 alone: 1.227] [dct: raw 3.518 h0 1.337 ctx 1.227 code raw/ctx 0.062/0.015 bpp; nz 12.89/blk eob0 2.7%] | 35.7s
          file: out_model16_b8_dct80_c4_mlp27_lam2_dz0_8k/model.ntcb 207987 bytes = 4.062 bpp (simulator raw 4.060 bpp); content 1662600 bits = simulator 1662952 - 352 side-info bits (stored in the header) [OK]; ...; self-check OK
          ntcb_decode.log: compare t0: model16.png (800x510): PSNR 38.42 dB, max |diff| 71; level 0 ... 12.9 nonzero ACs/block, 2.7% EOB-only; idct 2.49 ms (9.7 MAC/texel, nnz 82481)
```
Reading: at the same (Q, lambda) plain rounding keeps more ACs (no zero bin: nz 4.42 -> 5.50 at q 50, 10.14 -> 12.89 at q 80)
and gains 0.8-0.9 dB for 0.08-0.19 bpp order-0 (+7.5% / +11%), i.e. it lands at a different point of the curve rather than
strictly above or below it; the switch-time drop is smaller (2.31 dB against 3.12 at q 50; 1.10 against 1.67 at q 80) because
the plain quantizer's worst-case AC error is L/2 instead of L. Whether the mosquito noise is gone is for Richard's eye on
`ntcb_decoded.png` / `side_by_side.png` of the two dz0 directories; an equal-rate comparison needs a slightly larger lambda
(or lower Q) on the dz0 side. The plain quantizer's symbols are larger on average (`mag` h0 101106 against 83276 bits at q 50),
which the Exp-Golomb proxy and the order-0 estimate both charge.

### Files
main.cpp 4267 lines (was 4236), ntc_decode.cpp 1515 (1508), cuda/ntc_cuda.cu 1170 (1169), cuda/ntc_cuda.h 133 (132), README.md
1433 (1424), NTCB_PLAN.md 211 (209), DCT_NOTES.md this section and the section-5 bullet. Every added or changed source line
carries `[DCT]` (`git diff -U0 --ignore-cr-at-eol` over the four sources: 0 added lines without the tag); line endings kept
(main.cpp 4267 CRLF, 0 LF, the one `\r\r\n` line intact; the CUDA files LF). Deviations: none from the brief; the header's
`dct_flags` byte costs 8 bits of container overhead per DCT level (1048 against 1040 header bits in the runs above), and the
ntc_decode level-0 line and the `--resave` line were given a `dz 0` / v14 clause that the brief did not ask for (informational).
Nothing is committed.
