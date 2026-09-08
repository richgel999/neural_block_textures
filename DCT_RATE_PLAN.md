# DCT_RATE_PLAN.md: a rate proxy for the DCT-coded level 0 in the training loss

Plan written September 7, 2026 by a planning agent, read-only, against commit 2b5985e (tags `v0.10-ntcb-container`, `v0.11-dct-multichannel`; `main.cpp` 3921 lines, `cuda/ntc_cuda.cu` 1042, `cuda/ntc_cuda.h` 123, `ntc_decode.cpp` 1508). Line references are to that tree. It builds on DCT_MVP_PLAN.md (section 11 items 2 and 3 are the two parts below) and DCT_NOTES.md (sections 1-3 and 7, the refit addendum). Every new line is `[DCT]`-tagged, off by default, firewalled, removable; `ntc_decode.cpp`, the v13 file layout and the NTCB container are untouched (the decoder and the writer only see symbols, and a truncated block is a legal block).

Richard's decision (September 7, 2026): no contrast-masking / flat-block weight in this version (it stays a follow-up; see the deployment memory). Both parts stay behind one `--dct-lambda`.

## 0. Summary and decisions

Requested by Richard: "a simple rate proxy based off the number of zeros and the magnitude of the existing coefficients that propagates through into loss". Motivation (from the logs): at low q the trainer spends AC coefficients in flat areas. model11 at q 15 (`out_model11_b6_dct15_c4bilin_qes8_mlp17_refit500_cuda8k_fd50.log`) ends with `dct nz mean 0.65 median 0 ... zero% 99.0`, 72.8% EOB-only blocks, yet `dc-only-by-table 79.2%` of the blocks sit at codes k0-k3 (finer steps because the probe finds the decoder sensitive there) and the 9303 nonzero tokens cost 33463 + 8427 + 9303 = 51 kbit of the 169 kbit order-0 total; nothing in the objective prefers a zero plane, and the ES noise (`--lat-sigma 0.05` = 1.6 plane units per coefficient, section 2.3) is what walks a shadow coefficient across the dead zone.

Two parts, one option `--dct-lambda L` (loss = mse + L * bits / npix, L in LSB^2 per bit per pixel, i.e. LSB^2 per bpp), both on when L > 0:

- **A. Soft rate penalty in the latent ES objective.** For each antithetic pair the perturbed *shadow* (not the snapped copy: section 2.3 shows why the copy gives an identically zero gradient) is clamped, forward-DCT'd and quantized with the block's current scale code; the integer token cost of the symbols (section 2.2) enters the pair's loss difference, attributed to the 64 texels of the (block, channel) that produced it, exactly as the mse difference is attributed to a texel's pixel footprint. 2K forward-only evaluations per latent step (8 at `--lat-pairs 4`), about 1.6% of the ES decodes with a 27,27 decoder.
- **B. Deterministic truncation at snap time (RDO).** After quantizing a block, walk the nonzero ACs from the highest zigzag position downward and zero every one whose plane-domain distortion increase is below lambda_t(k) times its exact token-cost saving (run merge and EOB movement accounted for, no approximation). Same code in `dct_snap_block` and `k_dct_snap`; the symbols the decoder trains on change on both backends identically; check 7 already compares them.

**One lambda serves both.** A block's scale code k encodes its sensitivity gain g_k (LSB per plane unit, `dct_gain_of_code`); by Parseval a coefficient-domain distortion dD (plane units^2) is dD * g_k^2 LSB^2 summed over the block's pixels, so `g_k^2 dD / npix < L * dbits / npix` gives `lambda_t(k) = L / g_k^2`: a 16-entry float table per run, computed once on the host, uploaded with the step tables. Units: L is the slope of an R-D curve in LSB^2 per bpp (the JPEG / H.264 RDO convention, where lambda scales with the quantizer squared; here lambda* ~ 30 S(q)^2, section 7). No normalisation by the current mse (section 9, risk 4).

**Decisions to confirm:** (1) the rate term enters only the latent ES, not the MLP ES / FD (section 2.6: the symbols do not depend on the weights, so the term is a constant in every MLP loss difference); (2) the penalty applies only once `dct.live` (section 2.7); (3) `--dct-rate es | trunc | both` (default both) exists so the experiment can attribute the gain to A or B; (4) lambda is not stored in the model file (a training hyperparameter like `--lat-lr`); (5) the exact distortion form `c^2 - (c - v)^2` is used for B instead of the `v^2` approximation named in the request, because both cost the same and for |q| = 1 the exact value ranges over 0.75 L^2 .. 3.75 L^2 (section 2.4).

## 1. Options, validation, banner (main.cpp)

`Options` (after `dct_opts_given`, L171):
```
float dct_lambda = 0.0f;      // [DCT] --dct-lambda L: rate term in the loss, L in LSB^2 per bpp (0 = off: no rate term, no truncation, no behaviour change)
std::string dct_rate = "both"; // [DCT] --dct-rate es|trunc|both: which of the two parts --dct-lambda enables (A: ES rate term, B: snap-time truncation)
```
Parser (next to `--dct-map`, L2970): `--dct-lambda F`, `--dct-rate S`; both set `o.dct_opts_given` (so the existing "ignored without --dct-q" note covers them). `usage()` (after the `--dct-map` line, L246): two lines.

Validation (next to the `--dct-refit` checks, L3041-3044): `--dct-lambda < 0` -> "`--dct-lambda` needs L >= 0 (0 = off)"; `--dct-rate` not in {es, trunc, both} -> "`--dct-rate`: es | trunc | both"; `--dct-rate` given with `--dct-lambda 0` -> note only.

`DctLevel` (L668): `float lambda = 0; bool rate_es = false, rate_trunc = false; float lam_t16[16] = {0};` (lambda_t per code in 1/16-bit units, section 2.5), `std::vector<int32_t> bits; /* [block][channel] proxy bits of the snapped block, 1/16 bit */`, `size_t last_truncated = 0, last_truncated_c[MAX_DCT_CH] = {0,0,0,0}; size_t last_nz_before = 0;` (nonzero ACs before truncation in the last snap), `size_t rate_hits = 0, rate_evals = 0;` (last latent step: (block, channel, pair) evaluations whose bit difference was nonzero, section 6). Filled in the level-setup block (L3156-3173): `D.dct.lambda = o.dct_lambda; rate_es = lambda > 0 && dct_rate != "trunc"; rate_trunc = lambda > 0 && dct_rate != "es"; lam_t16[k] = rate_trunc ? (float)((double)lambda / (dct_gain_of_code(k) * dct_gain_of_code(k) * 16.0)) : 0.0f` (double, rounded once, the `DCT_AK` pattern); `bits.assign(nblk() * C, 0)`.

Banner (`dct      :` line, L3599): append `; rate: --dct-lambda L (es+trunc): loss = mse + L * proxy bits / npix, proxy = Exp-Golomb token costs (section 2.2), truncation lambda_t(k) = L / g_k^2` only when `lambda > 0`.

## 2. Definitions

### 2.1 Units
Image loss internal to the ES: `sum_pix sum_c cw (out - tgt)^2 / (3 wsum W H)` (`decode_err` L1498, `inv_px` L1726) = mse_LSB / 255^2 for one texture. Rate term: `L * bits / npix` in LSB^2, i.e. `L * bits / (npix * 65025)` internally; bits are integers in 1/16-bit units, so the internal weight is `rate_scale = L / (16 * 65025 * npix)` (float; `npix = W * H`, the padded decode size every bpp uses). One float, computed on the host, stored in `DctDesc`.

### 2.2 The token cost model (integer, both backends verbatim)
Per (block, channel), over the 63 ACs in `DCT_ZIGZAG` order, the same tokens `dct_analyze` counts (run, |q| - 1, sign; EOB when the last nonzero is below zigzag 63). Costs in 1/16 bit:
```
floor_log2(v)  : k = 0; while (v >>= 1) k++;            (v >= 1; a 6-iteration loop, no intrinsics)
c_run(r)       = 16 * (2 * floor_log2(r + 1) + 1)        r = 0..62 : 1, 3, 3, 5, 5, 5, 5, 7, ... , 11 bits
c_mag(m)       = 16 * (2 * floor_log2(m) + 1)            m = |q| >= 1 : 1, 3, 3, 5, 5, 5, 5, 7, ... bits (m capped at 256 as in dct_analyze)
c_sign         = 16
c_eob          = 32                                       (absent when the last nonzero sits at zigzag 63)
block bits     = sum over nonzero ACs [c_run(run) + c_mag(|q|) + c_sign] + (lnz < 63 ? c_eob : 0)
```
Exp-Golomb code lengths are the "a + b log2" form in integers (a = 1 bit, b = 2 bits per octave), a universal code that tracks an adaptive order-0 coder on geometric-like streams. Calibration against the simulator's h0 lines: m1 q 30 pays run 2.47 + mag 1.19 + sign 1 = 4.7 bits per nonzero token and 2.8 per EOB (`dct bits h0 ... run 61077 mag 29377 sign 24702 eob 11525` over 24702 tokens / 4096 EOBs); model11 q 15: 3.6 + 0.9 + 1 = 5.5 and 0.72 per EOB (72.8% of the blocks are EOB-only, so the EOB is cheap there). The model gives 3 bits for a run-0 |q| = 1 token, 5-7 with a short run, 13 for a lone coefficient at zigzag 63, and 2 per EOB. The DC is excluded (it is never zeroed by the trainer and its residual cost is roughly constant per block); zeros cost nothing except through the run of the next token. The `dct rate` line prints proxy / h0 so the tracking is visible (section 6). A `--dct-selftest` item pins the table values and a hand-built block (section 8).

### 2.3 Part A: the ES rate term
**Which buffer.** The ES perturbs the snapped copy for the mse (`zb = D.zdec()` L1717, `k_lat_pair` reads `d_zdec` L995): the straight-through rule. The rate must be evaluated on the *shadow* plus the perturbation, `clamp(z[i] +/- sigma eps[i], -1, 1)` with `z = D.lat.z` / `d_z`, for two reasons:
1. On the snapped copy the gradient is identically zero. A snapped block's coefficients sit at bin centres: dead-zone ACs at `tau + |q| L` (re-quantized `floor(|q| + 0.5) = |q|`), the first-order pair at `q L`, zeros at 0. The perturbation of a coefficient is N(0, (32 sigma)^2) in plane units (an orthonormal transform of iid Gaussians is iid with the same sigma), 1.6 units at `--lat-sigma 0.05`, while the distance to the nearest bin edge is 0.5 L >= 5 for every block with step >= 10 (k <= 5 at q 30, k <= 7 at q 15: the whole population of the logs). `bits(zq + sigma eps) - bits(zq - sigma eps) = 0` for every pair, apart from clamped texels and blocks with L <= 3.
2. The rate is a function of the shadow through the snap, `R(theta) = bits(snap(theta))`; `R(theta +/- sigma eps)` is the honest ES evaluation of it. The straight-through substitution exists only because the mse through the quantizer has zero gradient inside a bin; the rate term does not need it.

The clamp matches `dct_snap_block`'s in-place clamp (L800) so the evaluated plane is exactly `snap(theta +/- sigma eps)`'s input.

**What the term does.** The ES estimates the gradient of the Gaussian-smoothed staircase `E_eps[bits(theta + sigma eps)]`; it is nonzero only for coefficients within about 2 * 32 sigma = 3 plane units of a bin edge and pushes them toward the cheaper side. It is a reflecting barrier at |c| = L - 3 for zero-bin coefficients (the flat-area drift the motivation names: a coefficient diffusing under Adam-normalised noise gradients is pushed back before it crosses L) and at 2L - 3, 3L - 3, ... for magnitude increments. It does not push a coefficient sitting deep inside bin 1 back to zero: that is part B's job. Per hit the signal is large relative to the mse signal of a flat pixel (L = 80, 3 bits: `80 * 3 / (65025 * 262144) = 1.4e-8` per pair versus about 1e-9 per pixel for a 13-LSB error at g = 1.7), so the barrier acts at small lambda too; the sweep starts below lambda* for that reason (section 7).

**Formula.** For pair k with noise `eps`, per (block b, channel c): `dbits[b][c] = bits(snap_in(z + sigma eps)) - bits(snap_in(z - sigma eps))` (int32, 1/16 bit; the quantizer of section 2.4 including truncation when B is on, so the ES sees the bits of what would be coded). Gradient contribution for every texel i of block b, channel c: `grad[i] += rate_scale * dbits[b][c] * scale * eps[i]`, `scale = 1 / (2 K sigma)` (L1755). Added in a separate pass after the mse scatter on both backends (identical structure; check 4 is a tolerance check, section 8).

**Cost.** Per pair 2 forward DCTs + 2 quantize / truncate / tokenize passes per (block, channel): 2 * 16 MAC/texel + about 2 * 200 integer ops per block. At K = 4: 128 MAC/texel and 8 forward-only evaluations (no IDCT) per latent step, on top of the existing full snap (32 MAC/texel) in `qes_refresh` / `snap_latent_impl`. Against the 8 full decodes of the step: 7992 MAC/pixel with the 7,27,27,3 decoder (1.6%), 3672 with 7,17,17,3 (3.5%). Host: OpenMP over blocks, about 1 ms at 512^2.

### 2.4 Part B: truncation at snap time
In `dct_snap_block` between the quantize loop (L807) and `dct_recon_block` (L808), and in `k_dct_snap` between L594 and L595, when `lam_t16[code] > 0`. State: the 64 symbols (natural order), the coefficients `Cf` from the strict forward DCT, the step table `st`. Walk `p = 63 .. 1` over `DCT_ZIGZAG`; keep `next` = the nearest surviving nonzero above p (64 = none; p is then the last nonzero), and read `prev` = the nearest nonzero below p in the current symbols (0 = none). For a nonzero at p with `q`, `k = DCT_ZIGZAG[p]`, `v = dct_dequant_ac(q, st[k], u, v)`, `c = Cf[k]`:
```
r1 = p - prev - 1
token = c_run(r1) + c_mag(|q|) + c_sign
saved = (next == 64) ? token - (p == 63 ? c_eob : 0)                 // last nonzero: the EOB appears when p was 63, else it stays
                     : token + c_run(next - p - 1) - c_run(next - p - 1 + r1 + 1)   // interior: the run of the next token grows
dD    = c*c - (c - v)*(c - v)                                          // exact distortion increase (plane units^2); >= 0 for dead-zone ACs, may be ~0 for the first-order pair near c = 0.5 L
if (dD < lam_t16[code] * (float)saved) { sym[k] = 0; ntrunc++; }       // else next = p
```
Greedy downward: each decision is exact for the state at that moment (a coefficient below p truncated later lengthens p's run; the walk does not revisit). `dct_recon_block` then dequantizes the modified symbols, so the plane, the file, the container and `ntc_decode` all see the truncated block with no further change. The first-order pair has no dead zone; B gives it one that widens with lambda (kept iff `2cL - L^2 > lambda_t * 3 bits`, i.e. `c > L/2 + 1.5 lambda_t / L`). The `v^2` form of the request would charge 2.25 L^2 for every |q| = 1 coefficient; the exact form is 0.75 L^2 at c = L and 3.75 L^2 at c = 2L, so it truncates the edge-hugging coefficients first, which is the intended effect.

**Parity.** `c` and `v` are the strict-FP values both sides already agree on (check 7); the three new float operations are written `__fsub_rn(__fmul_rn(c, c), __fmul_rn(__fsub_rn(c, v), __fsub_rn(c, v)))` and `__fmul_rn(lam, (float)saved)` on the device, inside `STRICT_FP_BEGIN` on the host (contraction off, so `c*c - (c-v)*(c-v)` cannot fuse). `lam_t16[k]` are float bit patterns uploaded, never recomputed.

### 2.5 One lambda for A and B
`lambda_t(k) = L / g_k^2`, `g_k = (14/64) 2^(0.6 k)` (`dct_gain_of_code`), stored as `lam_t16[k] = L / (16 g_k^2)` so it multiplies the 1/16-bit `saved` directly; npix cancels (both sides of the criterion are per image pixel). Because `step = K1 * S(q) / g_k` (up to the integer rounding), `step * g_k = K1 * S(q)` is code-independent: the LSB-domain value of a |q| = 1 coefficient at zigzag position p is `K1[p] * S(q)` for every block, so B truncates by table entry (low K1 first) uniformly across codes, and its threshold scales with S(q)^2 (section 7).

### 2.6 The MLP objective
No. The symbols depend on the weights only through the refit (the probe reads the decoder at fixed selector points, then the codes change the step tables), a discrete coupling with no gradient; within an MLP ES or FD step the symbols are fixed, so `L * bits` is the same constant in every perturbed loss and cancels in every difference `k_mlp_es_finish` / `k_fd_grad` compute. Including it would change only the reported `mlp batch` loss. The MLP-side rate lever is a rate-aware refit (choose k per block by `D + lambda R` over the 16 tables, DCT_MVP_PLAN.md 11.4, initialised by the probe): a follow-up, not this plan.

### 2.7 Before the switch, refits, loads
- Before `dct.live` the plane is continuous, there are no symbols, codes or step tables, and the progress line charges level 0 at `--qbits`: the rate term is undefined and not applied. It starts at the switch (`dct_begin`) and, for a loaded v13, at the first latent step.
- A refit changes the codes, hence the step tables, hence `lam_t16[code]` and the symbols of the same shadow: the proxy of a fixed shadow jumps at a refit exactly as the symbols do (the refit line prints `psnr before -> after` and now `trunc N`). Nothing else to do.
- `--load` of a v13: the file's symbols are the truth, no re-snap at load (DCT_NOTES.md section 1), so no re-truncation at load; a different `--dct-lambda` on a resumed run is allowed and takes effect at the first snap. The mismatch message does not mention lambda.

## 3. main.cpp: functions and hook sites

Inside the DCT region, after `dct_dequant_block` (L771) and still inside `STRICT_FP_BEGIN`:
- `static inline int dct_floor_log2(unsigned v)`, `dct_c_run(int r)`, `dct_c_mag(int m)`; constants `DCT_C_SIGN = 16`, `DCT_C_EOB = 32`.
- `static int dct_block_bits(const int16_t* sym)`: the 1/16-bit proxy of one (block, channel) (integer only; the zigzag walk of `dct_analyze` pass 2 without the entropy terms; the comment says "keep in step with dct_analyze pass 2 and ntcb mode 3" as L2389 does).
- `static int dct_truncate_block(const float* Cf, const int* st, float lam16, int16_t* sym)`: section 2.4, returns the count zeroed.
- `static int dct_quantize_block(const float* Cf, const int* st, int dc_step, float lam16, int16_t* sym, int* ntrunc)`: the quantize loop of L807 + truncation; returns `dct_block_bits(sym)` when `lam16 > 0 || <es on>`, else 0. `dct_snap_block` calls it (hook: `was the one-line quantize loop`) and stores `Q.bits[b*C+c]`, accumulates `ntrunc` and the pre-truncation nonzero count through the existing `reduction(+:...)` / `omp atomic` idiom of `dct_snap_level0` (L815-823) into `last_truncated`, `last_truncated_c`, `last_nz_before`.
- `static int dct_rate_block(const DctLevel& Q, const Latent& L0, size_t b, int c, const float* z, const float* eps, float sg, float sign)`: gather `clamp(z + sign * sg * eps)`, `w = (s + 1) * 32`, `dct_fwd8`, `dct_quantize_block` on a local `int16_t sym[64]`, return the bits. Two calls per (block, channel, pair).
- `dct_recon_block` (L776): also fill `Q.bits` when `Q.lambda > 0` (so `--load --iters 0` prints the `dct rate` line for the file's symbols).

Outside strict FP: `static void dct_proxy_totals(const DctLevel& Q, double& bits, double* per_c)` (sum of `bits` / 16 per channel) and the `dct rate` printer (section 6).

Hooks:
- `LatentTrainer::step` (L1730-1778): inside the pair loop, after the scatter loop over levels, `if (D.dct.live && D.dct.rate_es) { /* dbits per (block, channel): OpenMP over blocks, dct_rate_block(+1) - dct_rate_block(-1) into a std::vector<int32_t> dbits(nbc); hits += (dbits != 0); then for every level-0 texel and channel grad[i] += rate_scale * dbits[b*C+c] * scale * eps[i]; }`. Block of texel `(tx, ty)`: `b = (ty / 8) * BW + tx / 8`. After the loop: `D.dct.rate_hits = hits; D.dct.rate_evals = nbc * K`. `rate_scale` computed once per step from `D.dct.lambda` and `D.W * D.H`.
- `cuda_check` check 4 (L2721-2768): the host replica gets the same rate pass (it must, or the gradient comparison fails with lambda > 0); it keeps the host `dbits` of pair K-1 for the new `dct rate` line (section 8).
- `qes_refresh` (L1414-1425): unchanged (the snap already runs through `dct_snap_level0`).
- `dct_begin` (L3425-3441) and `dct_refit_now` (L3450-3471): append `; trunc N of M` to the switch / refit line when `lambda > 0`.
- Progress line token group (L3788-3798), the final block (L3848-3850), the `done:` line (L3901), the banner (L3599): section 6.
- `dct_sync_device` (L3496-3501): fill the new `DctDesc` fields (section 4).

## 4. CUDA (`cuda/ntc_cuda.h`, `cuda/ntc_cuda.cu`)

Header: `DctDesc` (L68-70) gains `int rate_es, rate_trunc; float rate_scale; float lam_t16[16];` (zero at `init`, filled by `set_dct`; the struct is uploaded whole, so nothing else changes). `Trainer` gains `void download_dct_rate(int32_t* bits, int32_t* dbits_last_pair, int* ntrunc, int* nz_before, int* hits, int* evals);   // [DCT]`.

`.cu`, inside the DCT region:
- `dev_dct_floor_log2 / dev_dct_c_run / dev_dct_c_mag`, `dev_dct_block_bits(const int16_t*)`, `dev_dct_truncate_block(const float* Cf, const int* st, float lam16, int16_t* sym)`, `dev_dct_quantize_block(...)`: mirrors of section 3 with the `_rn` intrinsics of section 2.4.
- `k_dct_snap` (L578-596): the quantize loop (L594) becomes `dev_dct_quantize_block(Cf, st, c_dct.dc_step, c_dct.lam_t16[code[i]], sy, &nt)`; writes `bits[i]`; `if (nt) atomicAdd(d_ntrunc, nt)` and `atomicAdd(d_nz_before, nz)` (integer adds, order-independent). `k_dct_recon` (L598): writes `bits[i]` too.
- `k_dct_rate_pair(uint64_t seed, int step, int pair, float sg, const float* z, const uint8_t* code, int32_t* dbits, int* hits)`: one thread per (block, channel), the `k_dct_snap` gather with `s = clamp(z[idx] + sg * ntc_gauss(seed, NS_LAT, step, pair, tex.z_off + L0.off + idx))` for +1 and -1 (`idx = (y * W + x) * C + c`, the same key `k_lat_pair` uses for a nearest level, L374-377), two `dev_dct_fwd8` + `dev_dct_quantize_block`, `dbits[i] = bp - bm`, `if (dbits[i]) atomicAdd(hits, 1)`. Local memory: `w[64], Cf[64], sym[64]`, like `k_dct_snap`.
- `k_dct_rate_gather(uint64_t seed, int step, int pair, float scale, const int32_t* dbits, float* grad)`: one thread per level-0 texel, `for c: grad[base + c] += c_dct.rate_scale * (float)dbits[b*C+c] * scale * ntc_gauss(seed, NS_LAT, step, pair, base + c)`. Kept separate from `k_lat_gather` (L409-442) so the existing kernel is untouched.
- `lat_step` (L992-1005): inside the pair loop, after `k_lat_pair` and the `k_lat_gather` loop: `if (I.dct_live && c_dct.rate_es) { k_dct_rate_pair<<<blocks_for(I.dct_n, 128), 128>>>(...); k_dct_rate_gather<<<blocks_for(ntx0, 256), 256>>>(...); }`; `cudaMemsetAsync` of the two counters at the top. `rate_es` is read from a host-side copy in `Impl` (`I.dct_rate_es`), not from the constant.
- `Impl` (L619-620): `int32_t* d_dct_bits, *d_dct_dbits; int* d_dct_cnt /* ntrunc, nz_before, hits */; bool dct_rate_es, dct_rate_trunc;`; allocation next to the symbol / code buffers (after L769), the two `cudaFree`s next to L640; `set_dct` (L826-838) copies the flags; `download_dct_rate` copies the three buffers.

Firewall on the device: with lambda 0 `lam_t16` is all zero, `rate_es` is 0, `k_dct_snap` runs the same float instructions plus a not-taken branch and an integer bit count; no new kernel launches; `k_lat_gather` unchanged.

## 5. ntc_decode.cpp, the file, the container
No change. The v13 payload is the symbols and codes; a truncated block is a valid block; `ntc_decode`'s `dct_recon_level0` and `dct` / `idct` lines report what they always did (and their `nz` figures will show the truncation). The NTCB writer's mode-3 token loop (L2389) mirrors `dct_analyze` pass 2, whose grammar is unchanged; the `file:` self-check against `bitrate_stats` stays exact. `--load --iters 0` prints the writing run's `done:` line as before.

## 6. Statistics and prints (all gated on `lambda > 0`; nothing changes otherwise)

- Progress line, inside the existing `| dct ...` group (L3796-3797): after `bits/blk R/H/C` insert ` proxy P` (proxy bits per block position summed over channels, /16) so it sits next to h0 and the two should track; at the end of the group ` trunc N hits X%` (N = coefficients zeroed by B in the last snap; X = `rate_hits / rate_evals` of the last latent step, the one diagnostic that says whether A has a signal). On `--cuda` the figures come from the host re-snap after `download_model` (bit-identical to the device by check 7) except `hits`, downloaded with `mlp_stats` at print time.
- Switch and refit lines: `; trunc N of M nonzero` appended.
- Final block, one new line before `dct plane`: `dct rate lambda L mode es+trunc proxy total B /blk P = bpp | h0 /blk H ratio proxy/h0 R | lambda_t16 by k v0 .. v15 | trunc last-snap N of M (x%) | hits x% (of E)`; with C > 1 the `dct[c]` lines gain ` proxy P trunc N`.
- `done:` line (L3901): `| --dct-q 30 level 0 (bit simulator, lambda 80)` (`, lambda 80 es-only` / `trunc-only` for the partial modes); every other field verbatim.
- Maps: nothing new; `--dct-map all` already writes `dconly` and `nz`, which is what "do flat blocks go DC-only" is read from.

## 7. Lambda values and the sweep

**Scale.** With `step * g_k = K1[p] S(q)` (2.5), the cheapest coefficient B can remove is a run-0 |q| = 1 dead-zone AC at zigzag 3 (K1 = 10..16) hugging its edge, dD = 0.75 L^2, saved = 3 bits: it goes when `L > 0.75 (11 S)^2 / 3 ~ 30 S(q)^2` LSB^2/bpp. So lambda* = 30 S(q)^2: 30 at q 50, 83 at q 30, 333 at q 15, 3000 at q 5; first-order-pair coefficients near c = L/2 go at any lambda, wholesale removal of |q| = 1 (dD up to 3.75 L^2) needs 5 lambda*. The measured slopes of the q curve (order-0 bpp, mse in LSB^2) are lower: m1 q 30 -> 50: 32 LSB^2/bpp, q 50 -> 70: 25; model11 q 5 -> 15: 45, q 15 -> 50: 11. That gap is expected: B measures deviation from the shadow (the codec's distortion), and in a flat block the shadow's coefficient is noise the target never asked for, so B cannot value it correctly; A is the mechanism for flat areas and it acts below lambda*. Hence the sweep brackets lambda* on both sides and A-only / B-only runs attribute the effect.

**Values.** lambda in {lambda*/4, lambda*, 4 lambda*}: m1 at q 30: **20, 80, 320**; model11 at q 15: **80, 320, 1280**. Plus the lambda 0 control (which doubles as the firewall run) and, at the middle value, `--dct-rate es` and `--dct-rate trunc`.

**Commands** (every run `--write-ntcb`, then `ntc_decode <out>/model.ntcb -o <out>/ntcb_decoded --compare <image>`; the container bpp and the decoder's PSNR are the figures of record next to the trainer's order-0 bpp):
```
build_cuda\Release\ntc.exe m1.png --cuda --block 4 --latent 0 0 1 --latent2 0 0 4 --filter nearest,bilinear --pos lv1local --dct-q 30 --dct-start 0.25 --dct-refit 500 --qes 0,8 --mlp 27,27 --mlp-pairs 256 --iters 8000 --lr-anneal 0.5 0.05 --mlp-fd 0.5 --leak 0.0009765625 --dct-map all --dct-lambda L --write-ntcb --out out_m1_b4_dct30_c4bilin_qes8_mlp27_refit500_start25_lamL_cuda8k_fd50_ntcb
  (the same with --latent 0 0 2 --dct-q 30,30 -> out_m1_b4_dct30x30_..._start25_lamL_...)
build_cuda\Release\ntc.exe model11.png --cuda --block 6 --latent 0 0 1 --latent2 0 0 4 --filter nearest,bilinear --pos lv1local --dct-q 15 --dct-refit 500 --qes 0,8 --mlp 17,17 --mlp-pairs 256 --iters 8000 --lr-anneal 0.5 0.05 --mlp-fd 0.5 --leak 0.0009765625 --dct-map all --dct-lambda L --write-ntcb --out out_model11_b6_dct15_c4bilin_qes8_mlp17_refit500_lamL_cuda8k_fd50_ntcb
```
Reference points (order-0 bpp / PSNR, container bpp): m1 1 plane q 30 26.08 dB at 2.443 (3.874 bpp file; note this log used `--dct-start 0.5`, so the lambda 0 run at 0.25 is the control), q 50 26.41 at 2.810, q 70 26.88 at 3.426; 2 planes 20,20 26.11 at 2.550, 30,30 (start 0.25) 26.82 at 2.840 (4.977 file), 30,50 27.08 at 3.224; model11 q 5 35.13 at 0.906, q 15 (17,17) 35.84 at 0.972, q 50 39.56 at 1.877. A win is a point above the piecewise-linear q curve on the (order-0 bpp, PSNR) plane and on the (container bpp, PSNR) plane; the `dct_map_dconly.png` / `dct_map.png` pair and `nz`, `eob0`, `dc-only-by-table` say whether the flat blocks went DC-only, and `hits` / `trunc` which part did it.

## 8. Verification

1. **Firewall (lambda 0).** The regression pin (`psnr 23.83 dB`, `out_reg/model.bin` sha256 `cb88470777e1d041...`); the nine README `--cuda-check` lines (`cuda-check: all passed`, unchanged figures); `--dct-selftest` (the nine existing lines unchanged); `--load out_m1_b4_dct30x30_..._start25_..._ntcb/model.bin --iters 0` printing the log's `done:` and `dct` lines; and the strongest one: the 2-plane start-0.25 command re-run with the new binary without `--dct-lambda` must give a `model.bin` byte-identical to the log's (`cmp`; the GPU path is deterministic, DCT_NOTES.md refit addendum).
2. **Self-test items** (three new lines): the cost table values (`c_run(0..7) = 1,3,3,5,5,5,5,7 bits`, `c_mag(1..8)` the same, monotone to 62 / 256); a hand-built block's bits against a hand count; truncation cases at lambda_t chosen so that (a) a lone |q| = 1 at zigzag 63 is dropped iff `dD < lambda_t (c_run(62) + c_mag(1) + 16 - c_eob)`, (b) an interior coefficient's saving includes the run merge, (c) lambda 0 truncates nothing, (d) the exact dD of a first-order coefficient at c = 0.5 L is 0 and it is dropped by any lambda > 0.
3. **`--cuda-check`** on the existing DCT line with `--dct-lambda 80` and on the 2-channel `--dct-q 50,20` line with `--dct-lambda 80` (two new README lines), plus `--dct-lambda 100000` (every block collapses to DC-only: the extreme truncation path): check 4 PASS with the rate term in the host replica; a new `dct rate` line: `pair K-1 bit differences 0 of N differ, snapped-plane proxy bits 0 of N differ, truncated H = D, hits H = D  PASS` (host `dbits` from the replica of check 4, whose noise step is 0 and whose lr is 0 so the shadow is unchanged; device values through `download_dct_rate`); check 7 with truncated symbols `0 / 0 (max 0 ulp) / 0` before and after the step and on the refit line. The `dct snap` PASS line gains a `bits 0 of N differ` clause only when lambda > 0 (the `chs` pattern, so the C == 1 lambda 0 line is byte-identical).
4. **CPU vs GPU, `--rng hash`, 500 iterations** (the DCT_NOTES.md section 3 command with `--dct-lambda 80`): identical through the switch (same `trunc`, same `hits`), display-precision agreement at 500.
5. **`--load --iters 0` equality** of a lambda run's `model.bin` on both backends (the run's `done:` line, `dct rate` line with `trunc 0` since a load does not re-snap), `ntc_decode --compare --verify` max |diff| 1 and 0 ulp, the container round trip `[OK]` / `self-check OK`.
6. The section-7 experiments, logs committed under the naming above; a DCT_NOTES.md section in the QES_NOTES.md form with every number from program output.

## 9. Line counts, risks, removal list

**Lines.** `main.cpp` ~300 (options / parser / usage / validation 25; cost model, tokenizer, truncation, `dct_quantize_block`, `dct_rate_block` 120; `DctLevel` fields and setup 15; `LatentTrainer::step` rate pass 45; check-4 replica and the `dct rate` check line 40; prints, banner, done line 45; self-test 20). `cuda/ntc_cuda.h` ~12. `cuda/ntc_cuda.cu` ~170 (device cost model + truncation + bits 70, `k_dct_rate_pair` 40, `k_dct_rate_gather` 20, `lat_step` hook 10, buffers / free / `set_dct` / `download_dct_rate` 30). README ~30 (flag rows, the `--dct-q` paragraph, two check lines). About 510 lines, all deletable by the list below.

**Risks.**
1. *ES attribution of a per-block scalar to 64 texels.* The mse term on a full-resolution nearest level is effectively a per-texel finite difference (one pixel per texel); the rate term is a 64-dimensional ES with K = 4 pairs sharing one integer per (block, channel): the projection onto the crossing coefficient's basis vector is signal, the other 63 directions are noise that Adam normalises like any other. `hits` and the PSNR at matched rate are the test. Fallback **A2** (~80 lines per side, same hook sites, deterministic, per texel): the smoothed staircase's gradient in closed form, `d/dc E[bits] = sum over the two nearest edges e of dbits_e * phi((c - e)/s) / s` with `dbits_e` from re-tokenizing the block at q +/- 1 (126 integer tokenizations per block) and the plane gradient = one IDCT of the coefficient gradient (16 MAC/texel), added to `grad` before `adam.step` (L1779) and before `adam_step` (L1006); no ES noise, no pairs, compared by check 4's tolerance. Build A first as requested; switch to A2 if `hits` stays below a few percent or the matched-rate PSNR does not move.
2. *Oscillation at the dead-zone edge.* B moves the effective edge from L to `c* = 0.75 L + lambda_t saved / (3 L)`; the straight-through limit cycle DCT_MVP_PLAN.md 13.8 describes (plane jumps by 1.5 L at a crossing, damped by the annealed lr) now happens at c*, and A's barrier sits just below it. `chg sym` on the progress line shows churn; the mitigation, if needed, is a hysteresis on the previous symbol (keep a coefficient that was nonzero last snap unless `dD < 0.5 lambda_t saved`; history-dependent, so it would need its own check-7 clause), not in this plan.
3. *Zero gradient if A is evaluated on the snapped copy.* Designed out (section 2.3); the `hits` figure would read 0.0% and catch a regression.
4. *Lambda units across images.* lambda is absolute (LSB^2 per bpp), the RDO convention; the natural scale is lambda* = 30 S(q)^2, which depends on q and not on the image, and the observed q-curve slopes (11-45 LSB^2/bpp) sit within a factor of a few of each other on m1 and model11. Normalising by the current mse is rejected: it makes the objective non-stationary as the mse falls through training, A's strength is already Adam-normalised per parameter (only its sign structure matters), and B is the part that needs absolute units. A target rate (a controller on lambda) is a follow-up; the sweep is the tool now.
5. *B cannot see flat-area waste.* Its distortion is against the shadow, not the target (section 7); if the flat-block coefficients survive B at lambda* the finding is that A (or A2) is the necessary part, which the A-only / B-only runs show.
6. *Cost.* 1.6-3.5% of the latent step; host runs unaffected in practice.
7. *Parity surface.* Three new float operations in the strict block (2.4), integer everywhere else; the extreme-lambda check line exercises the truncation walk fully.

**Removal list** (delete, rebuild, re-run section 8 item 1):
- `main.cpp`: `Options::dct_lambda`, `dct_rate`; the two parser lines, two usage lines, three validation lines; the `DctLevel` rate fields; the cost model / `dct_block_bits` / `dct_truncate_block` / `dct_quantize_block` / `dct_rate_block` / `dct_proxy_totals` functions and the `dct rate` printer; the `dct_snap_block` hook (restore the one-line quantize loop) and the `dct_recon_block` bits line; the `dct_snap_level0` reductions; the level-setup lines; the `LatentTrainer::step` rate pass; the check-4 replica lines and the `dct rate` check line and the `bits` clause of the `dct snap` line; the progress / switch / refit / final-block / done / banner additions; the `dct_sync_device` fills; the self-test items; the DEPENDENCY comment bullet.
- `cuda/ntc_cuda.h`: the `DctDesc` fields, `download_dct_rate`.
- `cuda/ntc_cuda.cu`: the device cost model and truncation, the `k_dct_snap` / `k_dct_recon` hooks (restore L594), `k_dct_rate_pair`, `k_dct_rate_gather`, the `lat_step` hook, the `Impl` fields, buffers, frees, `set_dct` copies, `download_dct_rate`.
- README: the two flag rows, the paragraph sentences, the two check lines; DCT_NOTES.md section; this plan stays as a record.

## 10. Implementation order (each step builds, both configs, 0 warnings)

1. Options, validation, usage, `DctLevel` fields, banner / done-line labels; cost model, `dct_block_bits`, self-test items (host only). Check: refusals; self-test; regression pin.
2. `dct_quantize_block` with truncation, the `dct_snap_block` / `dct_recon_block` hooks, `trunc` statistics and the `dct rate` line. Check: lambda 0 byte-identity of the 2-plane start-0.25 `model.bin` on the CPU path of a short run; `--dct-lambda 100000` on chief1 for 100 iterations makes every block DC-only (`lnz 0`, `eob0 100%`).
3. Part A on the host (`dct_rate_block`, the `LatentTrainer::step` pass, `hits`). Check: `hits` reads 0.0% when the pass is pointed at `zq` (a one-line temporary swap, then reverted) and a few percent on the shadow; the chief1 100-iteration run at lambda 80 reports fewer `nz` than lambda 0.
4. CUDA mirror: device cost model, `k_dct_snap` hook, the two rate kernels, `lat_step`, buffers, `download_dct_rate`; check-4 replica and the `dct rate` check line. Check: section 8 items 3 and 4.
5. `--load --iters 0`, container round trip, README, DCT_NOTES.md section. Check: section 8 item 5.
6. Experiments (section 7), logs committed.
