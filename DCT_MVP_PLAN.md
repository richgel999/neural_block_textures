# DCT_MVP_PLAN.md — MVP of the DCT-coded selector plane (level 0)

Plan written September 6, 2026, read-only. Line references are to the working tree at tag `v0.8.1-psnr-basisu` (commit 1a6013b; `main.cpp` 2205 lines, `cuda/ntc_cuda.cu` 889, `cuda/ntc_cuda.h` 106, `ntc_decode.cpp` 1038; the sources are unchanged since 3b37fcb, so DCT_SELECTOR_PLAN.md's line numbers still hold). basisu references are to `C:\dev\bu_8_31\basis_universal`. This plan does not repeat DCT_SELECTOR_PLAN.md's evaluation (its sections 1-4) or PERCEPTUAL_PLAN.md; it turns Richard's decisions 1-11 (below) into insertion points, functions, kernels, file fields and checks. Fallback: the tag.

Decisions (September 6, 2026, binding): (1) experimental, off by default, firewalled and removable; (2) level 0 only, replacing `--qat`, one selector channel first; (3) XUASTC's quality math and table; (4) the per-block adaptive scale is required in the MVP, as a trainer-side probe stored as a 4-bit code; (5) training through the `--qes` mechanism, spatial shadow, no soft penalty, no truncation RDO; (6) rate = a bit simulator over zigzag / run-length symbols, nothing byte-coded; (7) plain basisu PSNR; (8) model file v13; (9) CUDA parity with a seventh check; (10) `ntc_decode` reads v13 and reports the IDCT stage; (11) load-time decode stays cheap.

## Terminology amendment (Richard, September 6, 2026)

Two independent sizes, not to be conflated: the **latent cell size** (`--block N`: the block
latent's downsampling factor, 4 / 6 / 8, today's padding multiple) and the **DCT block size**
(`--dct-block`: the transform block on the selector plane; square; 8 in the MVP; 4 / 12 / 16 are
easy later because the zigzag and the table resampling are generic in the size). They are only
coupled through padding: the image is padded to a multiple of lcm(latent cell, DCT block), e.g.
6 and 8 -> 24, 4 and 8 -> 8. This replaces the "`--block` must be a multiple of 8" refusal in
section 1 item 4 and the `--block 6` exclusion; the per-block scale code is per DCT block, the
cell-position input stays per latent cell, nothing else requires the grids to nest.

## Two more decisions (Richard, September 6, 2026)

- **Up to four selector channels eventually.** The MVP lifts only the one-channel case, but
  every array, table and file field is indexed `[block][channel]` from the start (symbols,
  scale codes, per-channel q, `MAX_DCT_CH = 4`), so multi-channel level 0 is the removal of the
  `C0 == 1` refusal plus per-channel q, with no file-format change (sections 3, 4.6, 5).
- **Level 0 has no bit depth in this mode.** A `--qat` level is quantized to B bits per texel; a
  DCT-coded level is quantized only through the symbols, i.e. by the quality factor and the
  table (and the per-block scale). Nothing may describe it as a B-bit level: the banner, the
  `bitrate` line, the `q<spec>` label, the file header and `ntc_decode`'s summary all say
  "DCT-coded, q N" and charge it by the bit simulator, never by bits per texel.

## 0. Summary of what gets built

With `--dct-q Q` on a `--latent 0 0 1 --filter nearest,...` layout at `--block 8` (or a multiple of 8), level 0 trains continuous and clamped to [-1,1] until `ceil(--dct-start * iters)`; at that iteration the trainer probes the decoder's sensitivity to the selector in every 8×8 block, quantizes it to a 4-bit log code, freezes the codes, and from then on every decode reads a level-0 plane that is the shadow pushed through forward DCT → dead-zone quantizer (JPEG luminance table, libjpeg quality scaling, per-block step scale from the code) → dequantizer → inverse DCT → clamp. ES on level 0 is untouched (`LatentTrainer::step` L896-963 never changes: level 0 is ES-active because `o.qat == 0` at L916/L933). The rate is a bit simulator over the zigzag/run-length symbols; nothing is coded to a byte stream. The model file becomes v13 only when the feature is on; `ntc_decode` reconstructs the plane by the same strict-FP IDCT. The CUDA backend mirrors the snap with a kernel that reads host-built integer tables and the host-fitted codes.

## 1. Options, validation, banner, refusals

All in `main.cpp`; every line tagged `// [DCT]` so the firewall list (section 7) can be executed by grep.

**`Options` (after `qes_start`, L143):**
```
// [DCT] --dct-q: level 0 is an 8x8 DCT-coded selector plane (see "DCT" block). 0 = off.
int dct_q = 0;             // 1..100 (JPEG quality; 100 = every AC step 1)
float dct_start = 0.5f;    // fraction of --iters at which the scale codes are fitted, frozen, and snapping begins (0 = from the start)
int dct_dc_step = 4;       // uniform DC quantizer step in [0,64]-plane DC units (orthonormal DC = 8 * block mean)
int dct_block = 8;         // transform block size; 8 is the only value accepted in the MVP
bool dct_selftest = false; // --dct-selftest: run the transform / table / zigzag self-test and exit
```
**Parser** (the `--qes-start` line, L1572, is the neighbour): `--dct-q N`, `--dct-start F`, `--dct-dc-step N`, `--dct-block N`, `--dct-selftest`.

**Validation** (after the `--qes` checks, L1624-1628, and after level resolution L1720-1735 where `nlev`, `near0` and `o.LW/LH` are known). Refusals, each `printf` + `return 1`:
1. `--dct-q` outside 1..100; `--dct-dc-step` outside 1..64; `--dct-block != 8` → "`--dct-block`: only 8 in the MVP"; `--dct-start` outside [0,1]. Notes (not errors): any `--dct-*` without `--dct-q` → "note: --dct-start / --dct-dc-step / --dct-block are ignored without --dct-q".
2. `--dct-q` with `--qat` → "`--dct-q` replaces `--qat` on level 0; drop one". (The two quantizers are mutually exclusive exactly as `--qat`/`--qes` are at L1725.)
3. `--dct-q` with a `--qes` list whose entry 0 is nonzero → "`--qes` on level 0 conflicts with `--dct-q`: use `--qes 0,B`". A single-value `--qes B` applies to levels >= 1 only, the same rule L1723 applies under `--qat`: add `|| o.dct_q > 0` to that condition.
4. `--dct-q` needs `--filter nearest` on level 0 (`near0`, L1696-1711), level 0 at full resolution (`o.LW == target.w && o.LH == target.h` after L1680-1681), `--block > 0 && --block % 8 == 0` (so W, H are multiples of 8 after padding, L1675), and `o.LC == 1` → "multi-channel DCT level 0 is a follow-up (the file layout already carries C0 channels)".
5. `--dct-q` with `--cuda` is supported (section 5); `--eval-filter` reads `zdec()` (L1281, L1289) and needs nothing.

**Banner** (after the `qes      :` line, L2004-2010):
```
dct      : level 0 = 8x8 DCT-coded selector plane, q 50 (JPEG luma table K.1, libjpeg scaling), DC step 4 (uniform), per-block 4-bit scale code from the decoder sensitivity probe; ES on the fp32 shadow, decode from the snapped plane from iteration 4000 (codes fitted then, frozen); rate figures are a bit simulator (nothing is entropy coded)
```
with "codes restored from the file, frozen" when loaded from v13.

**`--dct-selftest`**: runs section 8.4's self-test right after the parser (before any image is loaded) and exits 0/1.

## 2. Definitions (the math, with its sources)

### 2.1 Units
Selector `s ∈ [-1,1]` (the shadow is clamped to it at every refresh once `--dct-q` is on). Plane `w = (s + 1) * 32 ∈ [0,64]`, the normalization XUASTC/XUBC7 use (`transcoder/basisu_xbc7_decoder.h` L675-676; XUASTC `encoder/basisu_astc_ldr_encode.cpp` L300-303; DCT_SELECTOR_PLAN.md 1.3 row 3). Reconstruction `s' = clamp(w' / 32 - 1, -1, 1)`.

### 2.2 Transform
Orthonormal 2D DCT-II / DCT-III, ported from `dct2f::init` / `forward` (`transcoder/basisu_transcoder.cpp` L26613-26663, L26680-26721): basis `B[u][x] = alpha(u) * cos(pi * (2x + 1) * u / 16)`, `alpha(0) = sqrt(1/8)`, `alpha(u > 0) = sqrt(2/8)` (L26633-26641, L26644-26651). The MVP stores the 64 products `alpha(u) * cos(...)` in one host table `DCT_BASIS[8][8]` (float) filled once at startup (as `qat_init_grid` fills `QAT_GRID`, L974-976), so every consumer, host snap, device kernel, `ntc_decode`, copies bit patterns and never calls `cosf` (the commit-4941c22 rule, main.cpp L45-47). Evaluation order, fixed and mirrored verbatim:
```
forward:  T[x][v] = sum_{y=0..7} w[x][y] * B[v][y]   (sequential float adds from 0.0f, y ascending)
          C[u][v] = sum_{x=0..7} T[x][v] * B[u][x]
inverse:  T[x][v] = sum_{u=0..7} B[u][x] * C[u][v]
          w[x][y] = sum_{v=0..7} B[v][y] * T[x][v]
```
Since `DCT_BASIS` already contains alpha, `dct2f`'s trailing `* m_a_row[v]` (L26703, L26718) is folded in. Zero-skipping in the inverse (`if (C[u][v] == 0) continue`) is bit-exact: adding `0.0f * B` to a float sum leaves it unchanged, so `ntc_decode` may skip zeros freely.

### 2.3 Quality scaling and table
Per XUASTC `compute_level_scale` (`basisu_transcoder.cpp` L27167-27206):
```
q = clamp(q, 1, 100)
S = q < 50 ? 5000 / q : 200 - 2q        (L27178-27181)
S /= 100                                 (L27183)
A_k = 64 / max(span, 14) with span/64 replaced by the block gain g_k  (L27186-27191; section 2.5)
level_scale_k = S * A_k                  (L27201)
```
`g_scale_quant_steps` (L27164, L27194-27195) is dropped (factor 1): the plane is not requantized to a lattice after the IDCT (DCT_SELECTOR_PLAN.md 1.3).

Table: `g_baseline_jpeg_y[8][8]` (L26932-26943; Annex K Table K.1 with the DC entry 4) sampled bilinearly at `(x * 8/N, y * 8/N)` (`sample_quant_table` L27208-27252; `sample_quant_table_state::init` internal.h L1846-1857 uses the block size). At N = 8 the sampling is the identity and the DC entry is never read (DC has its own step, 2.4). Entry: `step_k[u][v] = max(1, (int)(base * level_scale_k + 0.5f))` (L27247-27249), no 255 clamp (XUASTC has none). `q >= 100` → every AC step 1 (`compute_quant_table` L27262-27269). Steps are built per code `k = 0..15` once, as 16 × 64 ints per channel, in one strict-FP host function; only the ints leave the host.

### 2.4 Quantizer
- **DC** (coefficient (0,0) = `8 * mean(w)`, range [0, 512]): uniform, no dead zone, `q0 = nearbyintf(C00 / S_dc)`, dequant `q0 * S_dc`, `S_dc = --dct-dc-step` (default 4, XUBC7's `XBC7_DC_QUANT`, `basisu_xbc7_decoder.h` L620-628 with its rationale: a DC step of 4 is ±0.5 of a weight step spread over the block). Round-half-even on both sides (`nearbyintf` / `__float2int_rn`, the `qes_index` convention, L354-359).
- **AC**: dead zone α = 0.5 with the first-order exemption, ported line for line from internal.h L1920-1962: for (1,0) and (0,1) `q = (int)roundf(d / L)` (L1924-1928; `roundf` is half-away-from-zero on both the CRT and CUDA), otherwise `s = |d|, tau = 0.5 L; q = s <= tau ? 0 : (int)floorf((s - tau) / L + 0.5f)` with the sign (L1934-1943); dequant `0`, `q * L` for the first-order pair, else `±(tau + |q| L)` (L1953, L1959-1961).
- Symbols: `int16` per coefficient, natural order `[u * 8 + v]`; `|q| <= 512` by construction (an orthonormal AC of a [0,64] plane is at most `||w||_2 <= 512` and every step is >= 1), so int16 never saturates and no DPCM fallback (XUASTC's 255 cap, internal.h L1537) is needed in the MVP.

### 2.5 The per-block scale: probe and code (decision 4, option (b))
**Probe**, run on the host, once, at `dct_start_it`, from the model the decoder will actually see (`zdec()` for all levels; on `--cuda` after `download_model` + `qes_refresh`, the L2082/L2122 pattern):

For block `(bx, by)` (8×8 texels = 8×8 pixels, since level 0 is full-resolution nearest) and selector channel `c`:
- Positions: `P = 4` pixels at `(bx*8 + {2, 5}, by*8 + {2, 5})`. Each has its own `lv1local` offset and bilinear block-latent sample, so the probe averages over the position dependence PERCEPTUAL_PLAN.md section 3 item 2 warns about.
- At each position build the MLP input once with `D.features(D.zdec(), px, py, f)` (L644-659), then evaluate `mlp_forward(D.mlp, D.mlp.p.data(), f, out, clamp_out)` (L501-525) with `f[c]` set to the five points `s_j ∈ {-1, -0.5, 0, 0.5, 1}` (the selector's min, max and three intermediate points).
- Segment gain `j = 0..3`: `g_j = (255 / 16) * sqrt( sum_{o < 3T} cw[o] * (out_{j+1}[o] - out_j[o])^2 / wsum )`. Units: 8-bit LSB per [0,64]-plane unit (Δs = 0.5 is Δw = 16 units; `out` is in [0,1], ×255 gives LSB; the norm is the cw-weighted Euclidean norm over the 3T outputs, which for one texture is the RGB Euclidean distance XUASTC's `get_max_span_len` uses, L27344-27346). `cw`/`wsum`: L602-606, L1748-1750.
- Block gain: `g_b = sqrt( mean over the 16 (position, segment) pairs of g_j^2 )` (RMS).
- Cost: 20 MLP evaluations per block per channel = 0.31 decode-equivalents on a 1080² image, once per run.

**Code** (`k = 0..15`, one uint8 per block per channel):
```
g_floor = 14 / 64 = 0.21875 LSB/unit          (XUBC7 / XUASTC span floor 14, L27186, over 64)
g_k     = g_floor * 2^(0.6 k)                 (0.219 ... 112 LSB/unit)
k(g)    = clamp( lround( log2(max(g, g_floor) / g_floor) / 0.6 ), 0, 15 )
A_k     = 1 / g_k                              (A_0 = 64/14 = 4.571, XUBC7's floor case)
```
`k(g)` is computed in double; `A_k` and `level_scale_k = S * A_k` in the strict-FP table builder; the tables are ints. The range must reach 112 because a nearly binary decoder that swings black→white over Δs = 0.1 has g ≈ 80 LSB/unit (k = 14), far above BC7's 4 LSB/unit (DCT_SELECTOR_PLAN.md 3.2(b)).

**Determinism.** The probe runs `mlp_forward` compiled `/fp:fast` (CMakeLists.txt L22), so a CPU run and a `--cuda` run may in principle pick different codes for a block whose `g` sits on a code boundary; that only changes that block's step table and is within the "display-precision agreement" expectation of the CPU-vs-GPU `--rng hash` comparison (README "Regression testing"). Within one run the codes are fitted once, on one side (the host), stored, uploaded, and compared exactly by check 7 (a copy check, as the codes are never recomputed on the device).

**When.** At `dct_start_it = ceil(--dct-start * iters)` (the `qes_start_it` computation, L1891), frozen after; `--load` of a v13 file restores them frozen (like the qes ranges, L1877-1882). No refit option in the MVP.

### 2.6 The snap (host, strict FP; `dct_snap_level0`)
For every block and channel: clamp the 64 shadow values to [-1,1] and write them back (as `qes_snap_level` does with `clamp`, L388), `w = (s + 1) * 32`, forward DCT, DC quantize/dequantize, AC quantize/dequantize with `step[c][code][u][v]`, inverse DCT, `zq = clamp(w' / 32 - 1, -1, 1)`; store the 64 symbols; return the number of changed `zq` values (the `qes_snap_level` contract, L383-393). The level-0 layout is `(y, x, c)` with `c` fastest (L307): the gather/scatter of a block is 64 strided reads/writes, no de-interleaving buffer is needed for C0 = 1, and the same loops work for C0 > 1.

### 2.7 Symbolization for the bit simulator (decision 6)
Per channel, per block, in block raster order: DC symbol `q0`; then the 63 ACs in the 8×8 zigzag (`generate_zigzag_order`, L26875-26930, ported as a 64-entry table `DCT_ZIGZAG` filled once) as `(run, sign, |q| - 1)` triples where `run` = zeros since the previous nonzero (0..62), with `EOB` = run symbol 64 (XUASTC `DCT_RUN_LEN_EOB_SYM_INDEX`, internal.h L1536; the `dct_run_len_model(65)` alphabet of the arithmetic profile) emitted when the remaining ACs are all zero (an all-zero block is DC + EOB; a block whose last nonzero is at zigzag 63 has no EOB, as in JPEG).

## 3. Data structures and functions (main.cpp)

All new code goes into ONE region, `// ---------------------------------------------------------------- DCT: transform-coded level 0 [DCT]`, inserted after `qes_spec` (L396-398) and before the tap section (L400). ~330 lines:

- `static float DCT_BASIS[8][8]; static int DCT_ZIGZAG[64]; static void dct_init_tables();`: filled once from `main` when `--dct-q` or `--dct-selftest` is given (like `qat_init_grid()` at L1716). Zigzag port of L26875-26930 (W = H = 8).
- `static const int DCT_JPEG_Y[8][8]`: `g_baseline_jpeg_y` verbatim (L26932-26943).
- `struct DctLevel { bool on = false, live = false; int N = 8, dc_step = 4; std::vector<int> q; /* per channel */ int BW = 0, BH = 0, C = 0; std::vector<int16_t> sym; /* [BH*BW][C][64] natural order */ std::vector<uint8_t> code; /* [BH*BW][C] */ std::vector<int> step; /* [C][16][64], entry [..][0] = dc_step */ int dc_raw_bits = 0; }` on `Decoder`.
- `STRICT_FP_BEGIN ... STRICT_FP_END` (the macro at L333-342; it already has `fp_contract(off)` on MSVC, required here because the transform has real multiply-add pairs; on GCC/Clang add `_Pragma("GCC optimize(\"fp-contract=off\")")` / `_Pragma("clang fp contract(off)")` to the macro: QES_NOTES.md item 9 records that the qes expressions have no contractible pair, so this is arithmetic-neutral for existing code):
  - `float dct_level_scale(int q, int k)` (2.3 + 2.5), `void dct_build_steps(int q, int N, int dc_step, std::vector<int>& step /* 16*64 */)` (2.3; `sample_quant_table` port with `sx = 8/N`).
  - `void dct_fwd8(const float* w, float* C)`, `void dct_inv8(const float* C, float* w)` (2.2, exact loop order above).
  - `int dct_quant_ac(float d, int L, int u, int v)`, `float dct_dequant_ac(int q, int L, int u, int v)`, `int dct_quant_dc(float d, int S)`, `float dct_dequant_dc(int q, int S)` (2.4).
  - `size_t dct_snap_level0(DctLevel&, const Latent& L0, float* z /* shadow, clamped in place */, float* zq)` (2.6).
  - `void dct_recon_level0(const DctLevel&, const Latent& L0, float* zq)`: dequantize + IDCT + clamp from `sym` alone (the loader and `ntc_decode` path; no shadow involved).
  - `void dct_clamp_level0(const Latent& L0, float* z)`: the pre-start clamp to [-1,1].
- Outside strict FP (they are estimates, not decode arithmetic): `void dct_probe_codes(const Decoder& D, DctLevel&)` (2.5), `struct DctBits { double raw, h0, ctx, code_raw, code_h0, code_ctx, header; double nz_per_block, eob_only_frac; }` and `DctBits dct_bits(const DctLevel&)` (section 4.5), `int dct_selftest()` (section 8.4), `std::string dct_spec(const DctLevel&)` (`"50"` or `"50,30"`).

`Decoder` (L596-637) gains `DctLevel dct;` and two marked hooks:
- `zdec()` / `zdec_mut()` (L618-619): `(qes_live || dct.live) ? zq.data() : lat.z.data()`.
- `qes_refresh()` (L621-630): the early return becomes `if (!qes_live && !dct.live) { if (dct.on) dct_clamp_level0(lat.lv[0], lat.z.data()); return 0; }`, and after the level loop `if (dct.live) changed += dct_snap_level0(dct, lat.lv[0], lat.z.data(), zq.data());`. Because the level loop already copies a non-qes level from the shadow into `zq` (L627), level 0 is first copied then overwritten by the snap; every existing caller of `qes_refresh` (L961, L1057, L1442, L1448-1449, L1487, L1880, L2122) therefore gets the DCT snap without being touched.
- `void dct_fit_all()` (next to `qes_fit_all`, L632-637): `dct_probe_codes(*this, dct); zq.resize(lat.size()); dct.live = true; qes_refresh();`.

## 4. Hooks at existing sites (main.cpp)

### 4.1 Level setup (after L1735)
`D.dct.on = o.dct_q > 0; if (D.dct.on) { dct_init_tables(); D.dct.C = o.LC; D.dct.q.assign(o.LC, o.dct_q); D.dct.N = 8; D.dct.dc_step = o.dct_dc_step; D.dct.BW = D.W / 8; D.dct.BH = D.H / 8; D.dct.sym.assign(...); D.dct.code.assign(..., 0); dct_build_steps per channel; }`.

### 4.2 Schedule (the qes pattern)
- `const int dct_start_it = o.dct_q > 0 ? (int)ceil(o.dct_start * o.iters) : -1;` next to L1891.
- `auto dct_begin = [&](int it) { D.dct_fit_all(); print "iter %6d  dct: scale codes fitted from the decoder sensitivity probe (histogram k0..k15: ...), level 0 decodes from the DCT-snapped plane from here; nz %.2f/blk, eob-only %.1f%%"; }` next to `qes_begin` (L1892-1900). Order matters: `qes_begin` first (so the probe sees the snapped block latent), then `dct_begin`; at `it == 0` before the CUDA init (L1901), in the loop inside the block at L2078-2091 with its own `changed` flag and `dct_sync_device()` call.
- `--load` v13: codes and symbols restored, `dct_recon_level0` fills `zq` level 0, the shadow's level 0 is set equal to it (no fp32 shadow is stored, the qes precedent L1109), `dct.live = true`, no probe, no re-snap (a re-snap of a clamped reconstruction is not guaranteed to reproduce the symbols; the file's symbols are the truth until the first latent step). With `--iters 0` nothing re-snaps, so `--load --iters 0` equals the run's final line. The `--resave` path (L2169) then writes v13 again from the same symbols.

### 4.3 CUDA sync
`dct_sync_device()` next to `qes_sync_device` (L1907-1923): builds `ntc_cuda::DctDesc` (section 5), calls `cu.set_dct(desc, D.dct.code.data(), D.dct.sym.data(), D.dct.step.data())`, which uploads and reconstructs the device plane from the symbols. Called after `qes_sync_device()` at L1948 and at the in-loop switch. `ModelDesc` fill (L1925-1944): `md.dct = D.dct.on; md.dct_C = D.dct.C; md.dct_dc_step; md.dct_q[c]`.

### 4.4 Training loop and prints
- `LatentTrainer::step` L896-963: no change (level 0 ES-active at L916; `D.qes_refresh()` at L961 now snaps through the DCT).
- `qat_search` is never called (`o.qat == 0`).
- The progress line (L2138-2149): with `D.dct.live` append ` | dct nz %.2f/blk eob-only %.1f%% codes ...` (mean nonzero ACs per block, fraction of EOB-only blocks; both from `dct_bits`). `qlabel` (L2035) becomes `"q" + qbits` until live, then `"dct" + dct_spec` (a lambda instead of a const).
- `done:` line (L2188-2198): `| --dct-q 50 level 0 (bit simulator) + --qes 0,8 latent + fp16 mlp: psnr ... at X bpp raw, Y bpp entropy-coded, Z bpp with contexts [level 0 alone: raw a / h0 b / ctx c; scale codes d bpp]`. The three totals keep their raw / entropy / context meaning so tables stay comparable.

### 4.5 `bitrate_stats` (L1173-1229)
One marked branch: `const bool dct = (l == 0 && D.dct.live);`. For that level skip the per-channel histogram loop (L1195-1222) and instead: `DctBits b = dct_bits(D.dct); raw_bits += b.raw + b.code_raw; ent_bits += b.h0 + b.code_h0; ctx_bits += b.ctx + b.code_ctx; s.bpp_ctx_level0 += (b.ctx + b.code_ctx) / npix; header_bits += b.header; memcpy(q, zd, nl * sizeof(float));` (the returned quantized latent for level 0 is the snapped plane, as for a qes level at L1207). Before `dct.live` level 0 falls through the existing continuous branch at `--qbits` (the same "not yet quantized" state the qes label logic handles at L2192); a checkpoint saved before the start stores level 0 continuous in a v12 file, accepted, as QES_NOTES.md deviation 2.

`dct_bits` (in the region), per channel:
- **raw** (fixed length): `nblk * dc_raw_bits + nnz * (7 + 9 + 1) + neob * 7`, with `dc_raw_bits = ceil(log2(floor(512 / dc_step) + 1))` (8 at step 4), run/EOB symbol 7 bits (65 symbols), magnitude 9 bits (`|q| - 1` in 0..511), sign 1.
- **h0**: zeroth-order entropy of the run stream (65 symbols), of the magnitude stream (`|q|`), of the DC residual against the left block's DC (first column against the block above; 0 for the first block), plus 1 bit per sign (what XUASTC's adaptive order-0 arithmetic profile achieves, transcoder.cpp L28849-28861).
- **ctx**: run and magnitude conditioned on the zigzag position of the previous nonzero (0 for the first token of a block), by the pair/triple hash counting of `context_entropy_bits` (L1149-1164) generalized to `(context, symbol)`; DC and signs as in h0. Cheap: one pass over the tokens.
- **Scale codes**: `code_raw = 4 * nblk`, `code_h0` = order-0 entropy of the 16-symbol code plane, `code_ctx = context_entropy_bits(codes, BW, BH)` (the existing function applied verbatim to the BW×BH code grid).
- **header** = 3 × 32 (q, N, dc_step) per channel.
- `nz_per_block`, `eob_only_frac` for the prints.
The banner's raw-bpp accounting (L1976-1990) cannot know the symbols before training; print `level 0: DCT-coded (rate from the bit simulator once live)` and charge level 0 at 0 raw bits there, with the note.

### 4.6 Model file v13 (`save_model` L1075-1112, loader L1752-1888)
Decision for item 8: **(b) write v12 when off, v13 when on.** Reasons: (i) the firewall: with the flag off the bytes written are those of today, so the V08-style byte comparisons of `model.bin` (V08_NOTES.md section 3, R1) stay valid and `ntc_decode`'s v12 path is not exercised differently; (ii) removal is the deletion of one branch in each of the three loaders; (iii) v13's payload differs structurally (int16 symbols replace level 0's floats), so a "v13 with zero dct fields" would still need two payload layouts in every loader; (a) buys nothing. Consequences: both loaders accept v9..v13 (`main.cpp` L1774; `ntc_decode.cpp` L104); `ntc_decode`'s "v9..v12 supported" message becomes v13; `--resave` (L2169) writes v13 only when the run has `--dct-q`.

Layout, magic `0x4E54433D`: identical to v12 through the per-level qes fields (L1100-1106; level 0's qes int is 0), then `int dct_N, dct_dc_step, dct_C (= C0)`, then `dct_C` ints `q[c]`, then the hidden widths and spec as now (L1107-1108), then the payload: `int16 sym[BH*BW][dct_C][64]` (natural order), `uint8 code[BH*BW][dct_C]`, then the floats of levels >= 1 only (`fwrite(D.zdec() + lv[1].off, ..., lat.z.size() - lv[0].size())`), then the MLP. `BW, BH` derive from the v11 image ints (L1093) and `N`. Adding channels later is `dct_C > 1` with no format change. Loader: read the three ints after the qes block (inside the `v12` conditional's successor `v13`), validate `N == 8`, `1 <= dc_step <= 64`, `dct_C == hdr[3]`, `1 <= q <= 100`, `saved_qat == 0`, `saved_qes[0] == 0`; the mismatch message (L1862-1863) gains ` --dct-q 50` / ` (no --dct-q)`; a v13 file loaded without `--dct-q`, or with a different q, is refused (unlike qes's warm start: the level-0 shadow does not exist in the file; loading a continuous v12 level 0 with `--dct-q` is the warm start, snapped at `dct_start_it`). `hdr_ok` bounds `sym` reads by `BW*BH*dct_C*64`.

### 4.7 `--cuda-check` check 7 (after check 6, L1482-1497)
Requires `--dct-start 0` (else print "dct snap: skipped (not live at iteration 0)" and pass). `compare()`: `cu.download_model(zs, p); cu.download_zq(zqd); cu.download_dct(symd, coded); D.lat.z = zs; D.qes_refresh();` then count `D.dct.sym[i] != symd[i]`, `D.zq[level 0][i] != zqd[i]` (exact; also the max ulp for the report), `D.dct.code[i] != coded[i]`. Run before and after `cu.lat_step(3, o.lat_sigma, o.lat_lr, o.lat_pairs)` (check 6 used noise step 2). Line: `  dct snap        : symbols %zu of %zu differ, plane %zu of %zu values differ (max %d ulp), codes %zu of %zu differ; after a latent step with lr %g: %zu / %zu / %zu  PASS`. Note that the first comparison's symbols are equal by construction (uploaded); the post-step comparison is the real parity test of `k_dct_snap`.

## 5. CUDA (`cuda/ntc_cuda.h` ~30 lines, `cuda/ntc_cuda.cu` ~170 lines, one region each)

**Header** (after `QesDesc`, L52-58): `static const int MAX_DCT_CH = 4;` `ModelDesc`: `int dct = 0, dct_C = 0, dct_dc_step = 4; int dct_q[MAX_DCT_CH];` `struct DctDesc { int live, C, BW, BH, dc_step; }`. `Trainer`: `void set_dct(const DctDesc&, const uint8_t* code, const int16_t* sym, const int* step /* C*16*64 ints */, const float* basis /* DCT_BASIS */);` `void download_dct(int16_t* sym, uint8_t* code);`.

**Device state** (`Impl`, L505-529): `bool dct = false; int16_t* d_dct_sym; uint8_t* d_dct_code;` `__constant__ DctDesc c_dct; __constant__ float c_dct_basis[64]; __constant__ int c_dct_step[MAX_DCT_CH][16][64];` (16 KB at 4 channels, within the 64 KB constant bank). The buffer allocation rule at L645-650 becomes `if (I.qes || I.dct)` → `d_zq` exists and `d_zdec = d_zq`; without either, `d_zdec = d_z` as now (the snapped-buffer rule, L24-31).

**Kernels** (after `k_snap`, L485-502):
- `dev_dct_fwd8 / dev_dct_inv8 / dev_dct_quant_* / dev_dct_dequant_*`: mirrors of section 2.2/2.4 written with `__fmul_rn` / `__fadd_rn` so nvcc's default contraction (`cuda/CMakeLists.txt` L16-17: FMA on, no fast-math) cannot fuse the multiply-add pairs; `__float2int_rn`, `roundf`, `floorf` as on the host.
- `k_dct_clamp0(float* z, n0)`: `--dct-q` on, not yet live: clamps level 0 of the shadow to [-1,1] (the host's `dct_clamp_level0`).
- `k_dct_snap(float* z, float* zq, int16_t* sym)`: one thread per (block, channel): gather 64 shadow values (clamp in place), `w = (s+1)*32`, forward, quantize with `c_dct_step[c][code][.]`, write `sym`, dequantize, inverse, `zq = clamp(w/32 - 1)`. 64 + 64 floats per thread in local memory; 18k threads at 1080², microseconds.
- `k_dct_recon(const int16_t* sym, float* zq, float* z)`: dequantize + inverse from the uploaded symbols into `d_zq` level 0 and `d_z` level 0 (the `set_dct` path; the host's `dct_recon_level0`).
- `snap_latent_impl` (L687-691): `if (!I.qes && !I.dct) return; k_snap(...) /* copies level 0 when qes has no bits there */; if (I.dct) { if (c_dct live) k_dct_snap else k_dct_clamp0 }`. Called, unchanged, from `lat_step` (L854), `upload_model` (L697), `set_qes` (L705).
- `set_dct`: uploads `c_dct`, basis, steps, codes, symbols; launches `k_dct_recon` (not `k_dct_snap`) so the device plane equals the host's reconstruction bit for bit at the switch and after `--load`.
- `download_dct`: copies `d_dct_sym`, `d_dct_code`.
- `init` (L550-676): allocate `d_dct_sym` (`BW*BH*C*64` int16), `d_dct_code`; refuse `dct_C > MAX_DCT_CH`; `c_dct.live = 0` until `set_dct`.
- `k_lat_pair` / `k_lat_gather` / `lat_step`'s mask (L835): unchanged; level 0 is in the mask because `qat == 0`.
- Destructor (L532-542): free the two buffers.

## 6. `ntc_decode.cpp` (~170 lines, one region + hooks)

- `Model` (L68-85): `int dct_N = 0, dct_dc_step = 0, dct_C = 0; std::vector<int> dct_q; std::vector<int16_t> dct_sym; std::vector<uint8_t> dct_code;`.
- `load_model` (L95-159): accept v13 (L104 message → "v9..v13"); after the qes block (L136-144) read the three ints and `q[]` when `v >= 13`; after the spec (L149-151), when `dct_N > 0`: read `sym` and `code` (sizes from `imgW/8 * imgH/8 * dct_C`; require `m.version >= 11`, `dct_N == 8`, `imgW % 8 == 0`, `lv[0]` nearest full-resolution, `qat_ch.empty()`, `qes_bits[0] == 0`), then the floats of levels >= 1 into `m.z + lv[1].off` and leave level 0 zeroed; then the MLP; the trailing-bytes check (L157) as now.
- New strict-FP region after `q8_quantize` (L285-303): verbatim copies of `DCT_BASIS` init (or the table itself), `dct_build_steps`, `dct_inv8`, `dct_dequant_*`, and `dct_recon_level0(Model&)` which fills `m.z` level 0 block by block (dequantize, zero-skipping IDCT, clamp) and reports `nnz` totals. Called in the "latent preparation" section (L847-852) inside its own `steady_clock` bracket; `level_quantized` (L283) gains `|| (l == 0 && m.dct_N > 0)`. The existing planar copy (L861-890) then works unchanged, and `ref_features` (L232-247) reads the same `m.z` for `--verify`.
- Summary (L922-933): level 0 line `dct 8x8, q 50, dc step 4, 4-bit scale codes, 5.1 nonzero ACs/block, 41% EOB-only, fp32 planar`; a new line `idct     : %.2f ms on %d thread(s) (%d blocks; %.1f MAC/texel with zero-skipping, 16 dense)` (single-threaded until September 8, 2026; since then the block rows run on the decode stage's thread pool) printed in `--bench` as its own stage (the timing note at L954 says the timed loop excludes it, so it is reported separately rather than folded into the table). `--pack-selectors` with a DCT level 0 prints "note: --pack-selectors does not apply to a DCT-coded level 0" (extend the L891 note). `--q8` / `--fp32-latent` do not touch level 0 (the `level_quantized` skip at L288).
- Load-time cost (decision 11): dense IDCT = 2 × 8·8·8 = 1024 MACs per 64 texels = **16 MAC/texel**; with zero-skipping in the first pass, `(8·nnz + 512) / 64 ≈ 8 + nnz/8` MAC/texel (≈ 9.3 at 10 nonzero ACs); dequantization = `nnz` multiplies per block (< 0.2/texel); the scale lookup is one row select of a 64-int table per block, no MAC. Against the 7,27,27,3 decoder of the model15/image3 layouts (7·27 + 27·27 + 27·3 = 999 MACs/texel; `Model::mlp_macs`, L83) that is 1.6% dense; against 7,17,17,3 (459 MACs) 3.5%.

## 7. Firewall: what "off" guarantees and what removal deletes

**Structural off** (`o.dct_q == 0`): `D.dct.on == false`, `D.dct.live == false`, so `zdec()` evaluates `qes_live ? zq : lat.z` exactly as at L618; `qes_refresh()` takes the same early return at L622 (the added clamp is behind `dct.on`); no `dct_init_tables()` call; `save_model` writes v12 (magic L1084 unchanged); `bitrate_stats` takes the existing branches (`dct` false); the loop's `rng` sees the same draws (the probe draws nothing and runs only when on; no new `std::normal_distribution` or `ntc_gauss` call exists on the off path); on the device `I.dct == false` → no buffer, `d_zdec` aliasing unchanged, no `k_dct_*` launch, `snap_latent_impl`'s first line still returns when `!I.qes`; `ntc_decode` reads `dct_N` only for v13. The regression pin and the five `--cuda-check` lines therefore exercise no new arithmetic; the `model.bin` byte comparison of V08_NOTES.md section 3 (R1, `/fp:precise` builds of the old and new `main.cpp`) is the tool if MSVC's contraction choices move under `/fp:fast`.

**Removal list** (delete these, rebuild, re-run section 8's regression items):
1. `main.cpp`: the six `Options` fields and the five parser lines and the usage lines; the validation block of section 1; the whole `[DCT]` region (section 3); `Decoder::dct` and `dct_fit_all`; the two hook expressions in `zdec()/zdec_mut()` and the two in `qes_refresh()` (restore L618-630 verbatim); the level-setup block (4.1); `dct_start_it`, `dct_begin`, `dct_sync_device`, the in-loop switch lines, the `ModelDesc` fills; the `qlabel` lambda (restore L2035); the progress/done-line additions; the `bitrate_stats` branch; the v13 branches of `save_model` and the loader and the magic in the loader's accept list; check 7; the `[DCT]` bullets in the DEPENDENCY notes (L18-59).
2. `cuda/ntc_cuda.h`: `MAX_DCT_CH`, the four `ModelDesc` fields, `DctDesc`, `set_dct`, `download_dct`.
3. `cuda/ntc_cuda.cu`: the `c_dct*` constants, the kernel region, the two `Impl` fields, `set_dct`, `download_dct`, the `|| I.dct` in the allocation condition (L645) and in `snap_latent_impl` (L688), the two `cudaFree`s, the `init` checks, the DEPENDENCY bullet.
4. `ntc_decode.cpp`: the `Model` fields, the v13 reads, the strict-FP DCT region, the `level_quantized` clause, the summary/idct/pack-selectors lines, the header comment bullet.
5. Docs: the README flag rows, the regression line, DCT_NOTES.md; DCT_SELECTOR_PLAN.md / this plan stay as records.

## 8. Verification

1. **Regression pin**: `ntc --crop 512 --mlp-pairs 32 --iters 200 --print-every 200 --save-every 200 --out out_reg` prints `psnr 23.83 dB`; the file magic is v12 (`4e54433c`); `ntc_decode out_reg/model.bin --compare out_reg/recon_q_final.png --verify` reads as in V08_NOTES.md R1 (PSNR 88.70 dB, max |diff| 1, verify 0 ulp pre-nonlinearity).
2. **The five README `--cuda-check` lines**: each `cuda-check: all passed`, six PASS lines (five for the layout without a selector level), and check 7 printing "skipped" or absent.
3. **`ntc_decode --compare --verify` unchanged** on `out_model10_b6_c1q2_c4bilin_mlp17_cuda8k_fd50` (v10), `out_model12_b8_c1q2_c4bilin_mlp27_cuda8k_fd50` (v11), `out_image3_b8_c1q4_c4bilin_qes8_mlp27_cuda8k_fd50` (v12), `out_m1234` (v12, 4 textures): the exact figures of V08_NOTES.md R3 (90.42 / 89.53 / 90.66 / 89.45-92.04 dB, max |diff| 1, verify 0 ulp). This also covers the `STRICT_FP` macro change in `ntc_decode.cpp` (section 3): `ref_mlp`'s `s += w[i] * cur[i]` (L259, L268) must still match the SIMD path at 0 ulp.
4. **`--dct-selftest`**: (a) `dct_inv8(dct_fwd8(w)) == w` to 1e-4 on 1000 random [0,64] blocks (orthonormality of the folded table); (b) at `q = 100`, `dc_step = 1`, `k` with `A_k = 1`, the quantized round trip has RMS error < 0.5 plane units (every step is 1: DC error <= 0.5, AC bin-center error <= 0.5); (c) `DCT_ZIGZAG` equals the JPEG 8×8 zigzag literal (0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63); (d) `dct_build_steps(q = 50, level_scale = 1)` equals Table K.1 in every AC entry (S(50) = 1, `(int)(base + 0.5)` = base); the DC entry is `dc_step`, not K.1's 16 nor basisu's 4; (e) `k(g_k) == k` for `k = 0..15` and `k` is monotone in `g`; (f) the first-order exemption: `dct_quant_ac(0.4 L, L, 1, 0) == 0`, `dct_quant_ac(0.6 L, L, 1, 0) == 1`, `dct_quant_ac(0.6 L, L, 2, 0) == 0`, `dct_quant_ac(1.1 L, L, 2, 0) == 1`, dequant of 1 at (2,0) = 1.5 L.
5. **New `--cuda-check` line**: `ntc chief1.png --cuda --cuda-check --block 8 --latent 0 0 1 --latent2 0 0 4 --filter nearest,bilinear --pos lv1local --dct-q 50 --dct-start 0 --qes 0,8 --qes-start 0 --mlp 17,17 --leak 0.0009765625 --iters 5 --print-every 5 --out out_chk`: checks 1-4 and 6 PASS, check 7 `0 / 0 / 0` before and after the latent step (no check 5: no `--qat`). Also the same line with `--dct-q 90` and `--dct-q 20` (step tables at both ends).
6. **`--load --iters 0` equality**: the 5-iteration run of line 5 without `--cuda-check` writes `out_chk/model.bin` with the v13 magic; `ntc chief1.png <same options> --load out_chk/model.bin --iters 0` prints the run's final `psnr` and the same bpp figures on CPU and with `--cuda`; loading it without `--dct-q` or with `--dct-q 30` is refused with the mismatch message; `ntc_decode out_chk/model.bin --compare out_chk/recon_q_final.png --verify`: max |diff| <= 1, PSNR > 85 dB, verify 0 ulp pre-nonlinearity; `--q8` and `--fp32-latent` outputs `cmp` identical.
7. **CPU vs GPU, 500 iterations, `--rng hash`** on chief1: `chief1.png --rng hash --iters 500 --lr-anneal 0.5 0.05 --mlp-fd 0.5 --mlp-pairs 32 --print-every 100 --block 8 --latent 0 0 1 --latent2 0 0 4 --filter nearest,bilinear --pos lv1local --dct-q 50 --qes 0,8 --mlp 17,17 --leak 0.0009765625`, with and without `--cuda`: display-precision agreement of PSNR and bpp (the QES_NOTES.md section 3 pattern, where 31.30 vs 31.29 dB was the observed spread); the `dct: codes fitted` lines print the same or near-identical code histograms.
8. **The first experiment** (basisu PSNR; simulated level-0 bits): for Q in {30, 50, 70, 90}:
   ```
   ntc model15.png --cuda --block 8 --latent 0 0 1 --latent2 0 0 3 --filter nearest,bilinear --pos lv1local --dct-q Q --qes 0,8 --mlp 27,27 --mlp-pairs 256 --iters 8000 --lr-anneal 0.5 0.05 --mlp-fd 0.5 --leak 0.0009765625 --out out_model15_b8_dctQ_c3bilin_qes8_mlp27_cuda8k_fd50
   ntc image3.png  --cuda --block 8 --latent 0 0 1 --latent2 0 0 4 --filter nearest,bilinear --pos lv1local --dct-q Q --qes 0,8 --mlp 27,27 --mlp-pairs 256 --iters 8000 --lr-anneal 0.5 0.05 --mlp-fd 0.5 --leak 0.0009765625 --out out_image3_b8_dctQ_c4bilin_qes8_mlp27_cuda8k_fd50
   ```
   against the logged per-texel points (DCT_SELECTOR_PLAN.md section 6 / the logs' `done:` lines): model15 `c1q1` 28.56 dB, level 0 alone 0.506 bpp ctx, total 0.838 ctx / 1.389 raw; `c1q2` 32.31 dB, 1.141, 1.487 / 2.389; image3 `c1q4` 49.29 dB, 0.866, 1.295 / 4.508. Report per run: PSNR, level-0 raw / h0 / ctx, scale-code bpp, total raw / ent / ctx, mean nonzero ACs per block, EOB-only fraction, the `dct: codes fitted` histogram, and `ntc_decode --bench` with its idct line; the `side_by_side.png` and `recon_q_final.png` (written at L2176-2186) are what Richard judges by eye. One variant worth a run: `--dct-start 0.25` (the default 0.5 puts the DCT switch, the qes fit and the FD switch all at iteration 4000).

## 9. Implementation order (each step builds and is checked)

1. Options, validation, usage, banner, `--dct-selftest` stub (~70 lines). Check: refusals of section 1, item 8.1.
2. The `[DCT]` region: tables, transform, table builder, quantizer, zigzag, self-test (~200 lines). Check: 8.4; 8.1.
3. `DctLevel` on `Decoder`, the `zdec` / `qes_refresh` hooks, `dct_snap_level0`, `dct_recon_level0`, the probe, `dct_fit_all`, the schedule (~130 lines). Check: CPU `chief1` layout of 8.7 for 300 iterations with `--dct-q 100 --dct-start 0.5`: at the switch the PSNR moves by a fraction of a dB (steps of 1/64 of the plane) and the `dct:` line prints; `--dct-q 30`: a visible drop then recovery; the regression pin.
4. `dct_bits` and the `bitrate_stats` branch, progress/done lines (~110 lines). Check: hand-count one block's tokens from a printed `sym` dump against `raw`; an all-zero plane gives `nblk * (8 + 7)` raw bits.
5. v13 save/load (~80 lines). Check: 8.6 on the CPU; v10/v11/v12 files still load in the trainer (V08_NOTES.md R4 commands).
6. CUDA mirror, `set_dct`/`download_dct`, check 7 (~200 lines across the three files). Check: 8.2, 8.5, 8.6 with `--cuda`, 8.7.
7. `ntc_decode` (~170 lines). Check: 8.3, 8.6's decoder lines, `--bench` idct line.
8. Docs: README flag rows and the regression line, the three DEPENDENCY notes, `DCT_NOTES.md` in the QES_NOTES.md form (every number from program output).
9. Experiments (8.8), logs committed with the naming scheme above.

Scope: `main.cpp` ~560 lines, `cuda/ntc_cuda.cu` ~170, `cuda/ntc_cuda.h` ~30, `ntc_decode.cpp` ~170, docs ~100; about 1000 lines, of which the deletable region/hook structure of section 7 is all of it.

## 10. Risks

- **Enum / parity (R7 of DCT_SELECTOR_PLAN.md).** The transform has multiply-add pairs; a single contracted FMA on either side flips a symbol at a bin edge. Mitigation is structural: `__fmul_rn`/`__fadd_rn` on the device, `fp_contract(off)` on the host (already in `STRICT_FP_BEGIN` on MSVC, L334; to be added for GCC/Clang), integer tables and the basis copied, `k_dct_recon` from symbols at every upload, and check 7 exact after a latent step. If check 7 ever reports plane differences with equal symbols, the IDCT order differs; if symbols differ, the forward path does. Fallback: XUBC7's fixed-point `dct2fx` (`basisu_xbc7_decoder.h` L23-50), bit-exact by construction.
- **The probe's determinism** across CPU and GPU runs (section 2.5): boundary blocks may get different codes; acceptable for the display-precision comparison, exact within a run. If it bothers the 500-iteration comparison, quantize `g` in double from a strict-FP copy of `mlp_forward` (a 30-line duplicate), not in the MVP.
- **ES convergence on a continuous plane.** Level 0 has 1.17M parameters trained by 4 antithetic pairs per step with per-texel attribution (L938-951); the snapped plane changes only when a shadow coefficient crosses a dead-zone edge (`0.5 L` plus a bin), so a block's high-frequency content can sit at zero for many steps while the shadow accumulates. If the q sweep shows PSNR still climbing at 8000 iterations or the nonzero count drifting, the first follow-up (per-texel central-difference gradient for level 0, two decodes, deterministic) replaces the ES estimate without changing anything else (DCT_SELECTOR_PLAN.md R1).
- **Ringing on hard edges.** model15 is dense texture; the clamp to [-1,1] and the MLP's co-adaptation from `dct-start` on are the only mitigations in the MVP; the side-by-sides at q = 90 and q = 30 show whether the decoder learns to tolerate overshoot or the plane needs the truncation RDO / a lower start fraction. frymire is the stress case to try after the two named images.
- **Three switches at one iteration** (`--dct-start 0.5`, `--qes-start 0.5`, `--mlp-fd 0.5`): the MLP Adam reset at L2072 coincides with the plane changing under it. `--dct-start 0.25` is the knob.
- **Rate honesty.** The simulator's h0/ctx figures are ideal adaptive-coder rates (as the existing `ctx` column is, L1128-1131); the raw figure is a real fixed-length cost. Nothing is byte-coded; every print says so.

## 11. Follow-ups, in order

1. Per-texel central-FD gradient for level 0 (two full decodes per step, the `k_lat_pair` feature-patch mechanism at L365-371 with ±h instead of ±σε; deterministic; replaces the 8-decode ES on level 0).
2. Soft rate penalty on the shadow (`lambda * log(1 + |c / step|)`, gradient by one IDCT per block, added before `adam.step` at L960; the printed simulator calibrates lambda), DCT_SELECTOR_PLAN.md 5.3.
3. AC-truncation RDO per block (XUBC7 `ac_truncate_rdo`, `encoder/basisu_xbc7_encode.cpp` L1937-2065): drop trailing nonzeros while `dD + lambda_bits * dR < 0`.
4. RD-searched scale code (option (f)): 16 block decodes per block, the exact discrete search the `qat_search` structure (L1007-1058, `k_qat_search` L444-479) already embodies; the probe becomes its initialization.
5. Multi-channel level 0 (`--latent 0 0 2`, `--dct-q 60,30`): the layouts of sections 3-6 already index `[block][channel]`; lift the `C0 == 1` refusal, per-channel q, `MAX_DCT_CH`.
6. DCT-coded block latent (level 1): the same `DctLevel` on a bilinear level; `zdec()` already serves whatever plane is snapped (DCT_SELECTOR_PLAN.md 4.3, PERCEPTUAL_PLAN.md item 4).
7. 4×4 blocks with XUBC7's matrix (`basisu_xbc7_decoder.h` L453-509, `g_base_4x4_quant`) or the resampled table with `sx = 2` (`--dct-block 4`; the zigzag and sampler are already generic in N).

## Critical files

- `main.cpp`: the QES block (L321-398) the new region follows; `Decoder::zdec` / `qes_refresh` (L618-637); `LatentTrainer::step` (L896-963, untouched); `save_model` / loader (L1075-1112, L1752-1888); `bitrate_stats` (L1173-1229); `cuda_check` (L1332-1500); the schedule and prints (L1891-1901, L2004-2010, L2078-2091, L2138-2149, L2188-2198)
- `cuda/ntc_cuda.cu`: `k_snap` (L485-502), `Impl` and `init` (L505-676), `snap_latent_impl` / `set_qes` (L687-706), `lat_step` (L829-856)
- `cuda/ntc_cuda.h`: `ModelDesc`, `QesDesc`, `Trainer` (L28-104)
- `ntc_decode.cpp`: loader (L95-159), `level_quantized` / `q8_quantize` (L283-303), latent preparation and planar copy (L847-891), summary (L917-939)
- `C:\dev\bu_8_31\basis_universal\transcoder\basisu_transcoder.cpp` (L26613-26721 `dct2f`, L26875-26943 zigzag and table, L27167-27252 quality scaling and sampling) and `transcoder\basisu_transcoder_internal.h` (L1839, L1920-1962 dead-zone quantizer)

## 12. Review 2 (experiment usability), September 6, 2026: adopted

Findings of the usability review, folded in as requirements. Sections 0, 1.4, 4.1, 6 and 8.8 are amended by item F below.

**A. The q axis is compressed per block by the adaptive scale; print the effective steps.** `step = max(1, (int)(base * S(q) / g_k + 0.5))`: a block at k = 14 has step(0,1) = 1 for every q >= 5 (q does nothing there); a block at k = 0 is DC-only until q ~ 95; k = 7 spreads sensibly (step(0,1) = 28 / 14 / 7 / 3 / 1 at q = 5 / 10 / 20 / 50 / 90). Print at the fit and in the final block: the code histogram, `step[k][zig 1]` and `step[k][zig 63]` per code with blocks, the fraction of blocks whose AC table is all ones ("lossless table"), and the fraction whose smallest AC step exceeds 64 ("DC-only by table"). Without these, "q 70 and q 90 gave the same rate" cannot be told apart from "the trainer stopped using ACs".

**B. Statistics.**
- Progress line, appended after the `(ent, ctx)` group, one token group: `| dct nz 5.13 lnz 9.4 eob0 41.2% clamp 0.8% bits/blk 67.9/31.2/24.8`. `nz` mean nonzero ACs per block; `lnz` mean zigzag index of the last nonzero over blocks with >= 1 nonzero; `eob0` fraction of blocks with no nonzero AC; `clamp` fraction of texels clamped after the IDCT (ringing indicator); `bits/blk` raw/h0/ctx per block including the scale code. `qlabel` = `dct50`.
- Final block, printed just before `done:` (which stays the last line), every line `dct <keyword> key value ...`, grep-able; also every print with `--dct-stats`:
  `dct cfg q N dc_step blocks (BWxBH) ch codes-fitted-at`;
  `dct nz mean median max zero% (interior% trailing%)` (interior zeros are paid by run symbols, trailing by one EOB);
  `dct nzhist 0 1 2 3-4 5-8 9-16 17-32 33-63` (fixed log buckets, also the map legend);
  `dct lnz mean median max hist 0 1-2 3-5 6-9 10-20 21-35 36-63 firstorder% <=2nd%`;
  `dct run mean run0% (runs = nonzero tokens; EOB tokens)`;
  `dct dc range mean h0 hres` (residual vs left, first column vs above);
  `dct pnz` 63 values (fraction of blocks with a nonzero at each zigzag position) plus `dct pnz_lo` (first 16);
  `dct codes` histogram, mean k, `step1 by k`, `all-ones%`, `dc-only-by-table%`;
  `dct blkbits raw mean median max | h0 mean median max | hist(h0)` (per-block h0 from the global order-0 code lengths);
  `dct bits raw|h0|ctx dc run mag sign eob code hdr total /blk = bpp` (three lines; EOB charged its own code length so the split sums to the total; the totals are exactly what is folded into the raw / ent / ctx figures);
  `dct plane rms max (units of 64) clamped% changed%` (the codec's own distortion against the clamped shadow: if PSNR falls while plane rms is flat, the trainer is the cause).
- `done:` line: keep `[level 0 alone: %.3f]` byte-identical and append a separate bracket `[dct: raw h0 ctx code bpp; nz/blk eob0%]`; keep the `psnr ... at ... bpp raw, ... entropy-coded, ... with an (up, left) context on level 0` text verbatim.
- Banner: do not print a raw total with level 0 at 0 bits; print `--dct-q 50 level 0 (rate from the bit simulator once live) + ... : levels >= 1 X bpp raw` and omit the total.
- Switch line: `dct_begin` prints PSNR before and after the first snap (`psnr 36.12 -> 35.40 dB`, two decodes once) and forces a progress print at `dct_start_it`, so the drop is attributable despite the three coincident switches at 0.5.
- `ntc_decode` prints the cheap subset only (nz mean, eob0%, lnz mean, code histogram, raw bits/blk and raw bpp, computed while dequantizing) plus `nnz` on the idct bench line; no entropy tables in the decoder (a third hand-written copy to keep in sync).

**C. Visualization PNGs.** `dct_map_NNNNNN.png` with the snapshots, `D.W x D.H` RGB, one colour per 8x8 block, no colour bar (keeps pixel alignment with `recon_*.png`); default `--dct-map nz` through the fixed 8-bucket ramp 0/1/2/3-4/5-8/9-16/17-32/33-63 = black/navy/blue/cyan/green/yellow/orange/white, documented once in the log at the fit and in the README; `--dct-map all` at the final iteration also writes `code` (16-step grey), `bits` (log buckets), `lnz` (diagonal buckets), `dconly` (binary); `dct_side_by_side.png` = `recon_q_final` | map; channels side by side for multi-channel. Plane-domain PNGs at every snapshot: `latent_q_NNNNNN.png` = the snapped plane through `save_latent_png`, `dct_resid_NNNNNN.png` = `(shadow - snapped) * 8 + 0.5` (gray 128 = 0; shows ringing and what the dead zone removed). The existing `latent_NNNNNN.png` remains the fp32 shadow.

**D. Sweep.** Bracket first: q = 100, 50, 10; then 1, 2, 5, 20, 30, 70, 90, 95 (the dense part belongs at 1-20 and 95-100, not 30-90). Also `--dct-start 0.25` at q 50, and `--dct-dc-step 8` and `2` at q 50 (the DC step is independent of q; at q 1 the DC is nearly the whole rate). Expected: q 100 within ~0.3 dB of the same layout trained with a continuous level 0 (its fp32 psnr column), raw bits enormous, h0/ctx the comparable figures; q 1 `eob0` near 100%, `lnz` near 0, PSNR a little above a block-latent-only run. Failure signatures: plane rms flat while PSNR falls (trainer); nz drifting through the second half (ES not settling); changed = 0 at the switch (snap not wired); one code for every block or all-ones 100% at q <= 50 (probe scale off); clamp > 5% (ringing; lower `--dct-start`); CPU vs GPU code histograms differing by more than a few percent per bin. Comparison points already logged (PSNR / raw / ent / ctx / level-0 ctx): model15 b8 c3 `c1q1` 28.56 / 1.389 / 1.333 / 0.838 / 0.506, `c1q2` 32.31 / 2.389 / 2.319 / 1.487 / 1.141; model15 b6 c4 `c1q2` 34.61 / 2.904 / 2.764 / 2.236 / 1.456, `c1q3` 37.23 / 3.904 / 3.722 / 3.047 / 2.259; model15 b4 c4 `c1q4` (`_psnr8`) 42.19 / 6.015 / 5.531 / 4.578 / 2.864; image3 b8 c4 `c1q4 qes8` 49.29 / 4.508 / 4.146 / 1.295 / 0.866, `qes6` 48.54 / 4.383 / 4.063 / 1.351 / 1.045, `qes10` 49.31 / 4.633 / 4.231 / 1.353 / 0.804. Missing and needed before the sweep: a float-selector run per layout (same command without `--dct-q`) as the q 100 reference, and a block-latent-only run as the q 1 floor.

**E. Fast sweep without retraining.** `--load <v12 float-selector model> --dct-q Q --iters 0` is an explicit, documented mode: the warm-start rule fits the codes and snaps at iteration 0 (one probe plus two statistics decodes per q, ten q values in about a minute). It gives the codec's own R-D on the plane the decoder wants with the decoder frozen (a lower bound on PSNR for that q, an upper bound on the rate a trained run needs), not the co-adaptation gain, ES settling or ringing tolerance; use it to choose the q values that get 8000-iteration runs. Re-quantizing a v13 at a different q stays refused (no shadow in the file). Options added: `--dct-stats`, `--dct-map [nz|code|bits|lnz|dconly|all]`; usage text says "100 = every AC step 1 (near lossless), 1 = coarsest; 0 = off" and that the DC step is independent of q; `--resave` with `--dct-q` on a loaded v12 prints what happened.

**F. Corrections to earlier sections.** Sections 0, 1 item 4, 4.1 (`BW = D.W / 8`), 6 (`imgW % 8 == 0`) and 8.8 still assume `--block % 8 == 0`; the terminology amendment supersedes them: pad to `lcm(--block, --dct-block)`, `BW = D.W / dct_block`, and the 8.8 sweep includes the `--block 6` and `--block 4` layouts of the comparison points. Everything else was found consistent with the stated decisions.

## 13. Review 1 (correctness, parity, firewall), September 6, 2026: adopted

Every line reference was re-verified by the reviewer against the tree and the basisu sources. Items below override the earlier sections where they conflict.

1. **`k_dct_recon` writes only `d_zq` level 0, never `d_z`** (section 5). Otherwise the device shadow becomes the reconstruction at every `set_dct` while the host keeps its fp32 shadow, the CPU/GPU trajectories diverge for a non-arithmetic reason, and check 7's pre-step comparison becomes ill-posed (a re-snap of a clamped reconstruction need not reproduce the symbols). The `--load` v13 case already sets the host shadow's level 0 to the reconstruction before `upload_model`, so nothing else is needed.
2. **The pre-start clamp must be applied identically on both sides and before the first decode.** Host: clamp level 0 to [-1,1] at the top of `qes_refresh` whenever `dct.on && !dct.live`, before either branch (the early return is not taken when `--qes` is already live, which is every `--qes-start 0` check line). Device: `k_snap` copies level 0 into `d_zq` before `k_dct_clamp0` clamps `d_z`, so decodes would read the unclamped copy; clamp inside the copy, or clamp both `d_z` and `d_zq` level 0. CPU-only: call the clamp once right after the loader so the `iter 0 (initial)` decode sees it. Check 6 then covers the not-live state.
3. **`bitrate_stats` branch placement.** The continuous raw charge (`raw_bits += nl * lbits`) precedes the histogram loop and the 64-bit min/max header follows it; neither is `fixed`/`qes` for a DCT level 0. Make the DCT branch the first statement of the level body (`if (dct) { ...; memcpy(q, zd, ...); continue; }`) and ensure the header charge is not reached, or the headline raw bpp is ~8 bpp too high.
4. **Gate the lcm padding on `--dct-q`.** `if (o.dct_q > 0) mult = lcm(mult, o.dct_block)` at the padding site, with the printed note updated; unconditional lcm would change every `--block 6` run with the flag off (firewall). Validation: both `D.W % 8 == 0` and `D.H % 8 == 0` after padding (trainer and `ntc_decode`, which checked only the width). State the consequences: padded area grows (up to 23 columns/rows at 24), bpp per padded texel, PSNR on the original, an auto-sized level 3 (`2 * block`) composes (24 is a multiple of 12).
5. **`DCT_BASIS` is 64 hex `uint32_t` literals in both `main.cpp` and `ntc_decode.cpp`** (plus the upload to CUDA), never recomputed with `cosf` (the commit-4941c22 rule). The self-test checks each literal against `sqrtf(...) * cosf(...)` within 1 ulp as a sanity check, not as the source of truth. Folding alpha into the table changes rounding relative to basisu, which is fine: nothing must match basisu bit for bit, only itself.
6. **Dead-zone description corrected.** `quantize_deadzone` gives `q = floor(|d| / L)`: the zero bin is `|d| < L` (not `<= 0.5 L`), `|q| = 1` covers `[L, 2L)` and dequantizes to `1.5 L`. Self-test (b) becomes "max error <= 1 step for non-exempt ACs, <= 0.5 for DC and the first-order pair". Guard every `(int)` cast against NaN/inf by clamping in float first (the `qes_index` rule).
7. **The GCC/Clang `fp-contract=off` addition to `STRICT_FP_BEGIN` is not arithmetic-neutral** there (the qes grid value is a multiply-add; on GCC `-ffp-contract=fast` is the default under `-march=native`). It is a latent fix for non-MSVC builds, where the regression items are not pinned anyway; on MSVC nothing changes. Say so.
8. **Training dynamics.** The plane-domain ES attribution needs no orthonormality argument: the perturbation is added to the snapped plane and scattered per texel, so it is the plane-domain gradient at the snapped point whatever the snap is (the same straight-through rule as `--qes`). Quantified at q 50, k 7: steps (0,1) 3, (2,2) 4, (4,4) 17, (7,7) 25 plane units; zero bins ±L; a unit coefficient moves a texel by at most 0.25 units; the shadow moves up to 0.02 in `s` per step, so a coherent pattern crosses the widest bin in ~5 steps at full lr and ~100 at the annealed 0.001: no stall. The real effect is a straight-through limit cycle when a coefficient crosses `L` (plane jumps by 1.5 L), damped by the annealed lr, which is why `--dct-start 0.5` coinciding with `--lr-anneal 0.5` is the right default; per-texel FD stays a follow-up. Print on the progress line the number of symbols (and `zq` values) that changed since the last print, the one diagnostic that separates "ES still climbing" from "plane churning".
9. **Probe.** Sound and cheap (0.31 decode-equivalents); needs no DC (it reads fixed selector points, never the block's symbols), so running it at the switch before any symbols exist is consistent; the multi-texture `/wsum` normalization is a heuristic (XUASTC sums channel squares), say so. The codes are fitted once and frozen while the MLP keeps adapting for 4000 iterations: re-run the probe at the end without applying it and print the would-be histogram and the fraction of blocks whose code would move by >= 2, to decide whether a refit option is next.
10. **DC step units.** DC = 8 x mean here (range [0,512]); XUBC7's rationale was for a 4x4 DC = 4 x mean, so `--dct-dc-step 4` is 0.5 mean units, finer than either precedent; 8 reproduces XUBC7's mean resolution (65 levels, 7 raw bits). `|q0| <= 128`; the AC magnitude bound is 256, not 512 (`||w - mean||_2 <= 256`), so the raw figure uses 8 magnitude bits (XUASTC's alphabet); int16 stays.
11. **Table orientation.** `g_baseline_jpeg_y` is not symmetric; with `dct2f`'s pass order the coefficient at `[u * 8 + v]` has `u` the vertical (row) frequency and `step[u][v] = K1[u][v]`; self-test (d) must compare with that indexing or it passes for the transpose too. The zigzag literal was checked by hand for the first three diagonals.
12. **Check 7.** Its pre-step comparison is a real test once item 1 holds: the host plane comes from the full snap and the device plane from `k_dct_recon` on the same symbols, so "plane 0 of N differ" proves the two IDCT + clamp paths agree; the post-step comparison adds the forward path and the quantizer. Say so in the PASS line.
13. **Device arithmetic.** `__fmul_rn` / `__fadd_rn` for every product and sum, including `s - 0.5f*L` and `tau + |q|*L` (both contractible), `__fdiv_rn` for the divisions; `__float2int_rn` / `nearbyintf` for DC, `roundf` on both sides for the first-order pair; zero-skipping is bit-exact (+0.0f sums). `ntc_decode`'s strict block lacks `fp_contract(off)` on MSVC but that target is built without `/arch`, so no FMA can be emitted; state that as the reason.
14. **Firewall.** Holds line by line with item 4 fixed. `save_model` keys v13 on `dct.live`, not `dct.on`. Removal list additions: the padding gate and its note, the loader-side clamp call, the `--dct-selftest` dispatch line. `usage()` text changes with the flag off (documented).
15. **v13 loader.** Add `imgH % 8 == 0`; in the trainer derive `BW, BH` from the header dims after `dims_ok` and read `sym`/`code` only then; set `D.zq = D.lat.z` before `dct_recon_level0` on a v13 load or `zq` is empty. The refusal set is otherwise complete.
16. **Hooks.** All confirmed at the cited lines. Note: `--load --iters 0` of a v12 with `--dct-q` runs `dct_begin(0)` (ceil(0.5 * 0) = 0), which is the fast-sweep mode of section 12.E, intended; `qlabel` becomes `dct50,q8`-style so the qes column survives; the `latent_*.png` dumps show the shadow, as for qes.
17. **Miscellany.** Pick one pre-start bpp convention (banner and progress line both charge level 0 at `--qbits` until the switch, then the simulator); the probe, IDCT and zero-skipping cost figures are correct; scope is more likely 650-700 lines in `main.cpp`; there are two loaders, not three; parallelize the host snap over blocks with the existing OpenMP idiom (deterministic, blocks independent; ~32 MAC/texel per refresh).
