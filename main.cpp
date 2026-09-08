// ntc - neural texture compression testbed trained with Evolution Strategies.
//
//   I_hat(u,v) = MLP( bilinear(Z, u, v), phi(u,v) )
//
// Z   : low-res latent texture (LW x LH x LC floats), optionally plus a second,
//       coarser level (--latent2) whose bilinear sample is concatenated onto it.
//       Up to 4 same-size RGB textures (a material) share Z and the MLP, which
//       then has 3 outputs per texture; per-texture loss weights via --weights.
// MLP : tiny fully connected net, < ~2000 weights
// phi : (u,v) in [-1,1] (`uv`), or the level-1 cell offset (`lv1local`), or nothing (`none`)
//
// Both Z and the MLP are trained with antithetic ES (no backprop).
//   MLP    : global ES on a random pixel minibatch shared across all pairs.
//   Latent : all texels perturbed at once; each texel's loss change is measured
//            only over the pixels its bilinear footprint touches, so one
//            full-image decode pair yields a gradient estimate for every texel.
//
// DEPENDENCY: ntc_decode.cpp (the standalone SSE4.1 CPU decoder) re-implements the decode
// path of this file and must be kept in sync. If you change any of the following, update
// ntc_decode.cpp and re-run its verification (CPU_DECODER_NOTES.md, --compare / --verify):
//   - the model file layout: save_model() and the loader (the magic / version cascade and its
//     version floor (v9), header fields, per-level filter flags, leak, qat bits, v11 size /
//     crop / clamp ints, latent and parameter order)
//   - the sample geometry: bilinear_tap(), sample_latent(), the (px + 0.5) / W uv convention,
//     edge clamping, and the nearest "cells are blocks" rule
//   - the positional inputs: PosEnc::encode() and the feature order in features()
//   - the MLP: mlp_forward() (sequential accumulation order, bias placement, the leaky ReLU
//     and its slope, the sigmoid / --clamp output), activate(), the weight layout in MLP
//   - the quantization grid: QAT_GRID / qat_init_grid(), qat_value(), qat_snap()
//   - the --qes quantizer (qes_index(), qes_fit(), qes_snap_level(): per-channel min / max
//     range, round-half-even index, grid lo + k / levels * range, strict floating point) and
//     the v12 file fields (per level: --qes bits, C lo and C hi floats); a --qes level is
//     stored as its on-grid values and must not be re-quantized by the decoder
//   - the post-hoc 8-bit latent quantization in bitrate_stats() (per-channel min / max,
//     lround, 1e-6 range floor), which defines recon_q_final.png and the quoted bitrates
//     for the levels that are not quantized in training
//   - save_png() rounding and pad_image() (which sizes the decoded image)
//   - [DCT] the experimental --dct-q level 0 (the "DCT" region): the DCT_BASIS_BITS / DCT_AK_BITS literals,
// [DCT]     dct_build_steps() (q, N, dc_step, code -> integer steps), dct_inv8(), dct_dequant_dc/ac(), dct_recon_block()
// [DCT]     (zq = clamp(w' / 32 - 1)), and the v13 file fields and payload (N, dc_step, C = 1..4, q[C]; int16 symbols
// [DCT]     [block][channel][64] and uint8 scale codes [block][channel], block-major with the channel fastest, then the
// [DCT]     floats of levels >= 1 only). Off by default: v12 is written and nothing changes.
//   - [NTCB] the bit-packed container (--write-ntcb, the "NTCB" region; NTCB_PLAN.md section 1): the header fields and
// [NTCB]    their ranges, the per-level records (mode 0 q8 / 1 qes / 2 qat palette / 3 dct), the 8-byte section headers,
// [NTCB]    the LSB-first bit packing, the grid dequantization lo + k / (float)levels * range (strict FP, as qes_rebuild),
// [NTCB]    the DC / (run, |q| - 1, sign) / EOB token stream (the pass-2 loop of dct_analyze) and the fp16 MLP section.
// [NTCB]    The writer (ntcb_write) lives here, the readers are ntcb_restore here and load_ntcb in ntc_decode.cpp; the
// [NTCB]    three must stay in step, and the file: line must print OK (NTCB_NOTES.md). Off by default.
//
// DEPENDENCY: cuda/ntc_cuda.cu (the --cuda training backend) mirrors the whole training
// path of this file, not just decoding: bilinear_tap() / sample_latent(), PosEnc::encode()
// (`uv`, `lv1local`), features(), mlp_forward() / activate(),
// the antithetic ES step and its footprint attribution (LatentTrainer, nearest and bilinear),
// the central-difference MLP step (--mlp-fd), Adam, the learning-rate schedule, the
// minibatch draw, the hash RNG streams (cuda/ntc_noise.h,
// shared with --rng hash here), the QAT grid and qat_search(), the --qes snap (k_snap =
// qes_snap_level(); the index is recomputed on the device with the same expression, the grid
// values are looked up from the host table uploaded by set_qes) and the model description
// handed over in ntc_cuda::ModelDesc. Snapped decode buffer rule: with a live --qes level every
// decode on either side reads the snapped copy (Decoder::zdec() here, d_zdec there), the ES
// perturbation is added to that copy and Adam updates the fp32 shadow (lat.z / d_z), which is
// re-snapped after every latent step; without --qes both aliases are the shadow itself.
// [DCT] The experimental --dct-q level 0 uses the same rule: k_dct_snap mirrors dct_snap_block (clamp, forward DCT,
// [DCT] dead-zone quantizer, inverse DCT into the decode copy), k_dct_recon mirrors dct_recon_block (from uploaded
// [DCT] symbols, decode copy only, never the shadow), k_dct_clamp0 the pre-start dct_clamp_level0; the basis and the
// [DCT] integer step tables are uploaded, the scale codes are host-fitted and copied; check 7 of --cuda-check compares
// [DCT] symbols, plane and codes exactly. Off by default: no buffer, no kernel, no aliasing change.
// [DCT] --dct-lambda (DCT_RATE_PLAN.md): the integer token cost model (dct_c_run / dct_c_mag / dct_block_bits), the
// [DCT] snap-time truncation dct_truncate_block (inside dct_quantize_block, three strict-FP float operations) and the
// [DCT] ES rate term dct_rate_block / the LatentTrainer::step rate pass are mirrored by dev_dct_* / dev_dct_quantize_block,
// [DCT] k_dct_rate_pair and k_dct_rate_gather; lam_t16[16] and rate_scale are host-computed floats uploaded in DctDesc.
// [DCT] --cuda-check compares the proxy bits, the pair bit differences and the truncation counts (the `dct rate` line).
// Nothing is shared by compilation except ntc_noise.h;
// every device function is a hand-written copy of a host function with the same name and
// the same floating-point evaluation order. So: any change to one of those functions, to a
// positional kind, to the Options that reach ModelDesc, or to the model file must be made
// on both sides, and `--cuda-check` (README "Regression testing") must pass afterwards. It
// compares decode, ES gradients, FD gradients, latent gradients and the selector search on
// identical inputs (plus the snapped copy, check 6); `--rng hash` lets a full CPU run be
// compared against a GPU run as well.

#define _CRT_SECURE_NO_WARNINGS
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif

#include "ntc_noise.h"   // counter-based noise: the GPU backend's RNG, also usable on the CPU (--rng hash)
#ifdef NTC_CUDA
#include "ntc_cuda.h"    // GPU backend (--cuda); see CUDA_PLAN_B.md
#endif


// ---------------------------------------------------------------- options
struct Options {
    std::vector<std::string> inputs;    // positional PNGs, 1..MAXT; empty after parsing -> kodim23.png (checked in)
    std::vector<float> weights;         // per-texture loss weight; all 1 unless --weights was given
    bool weights_given = false;
    std::string outdir = "out";
    int crop = 0;              // --crop N: center-crop input to N x N (0 = none, the default: the whole image is compressed)
    int block = 0;             // --block N: pad the (cropped) image to a multiple of N x N by duplicating its last column / row
    int orig_w = 0, orig_h = 0; // source size before --block padding (after --crop); stored in the model file (v11)
    int LW = 64, LH = 64, LC = 4;
    int LW2 = 0, LH2 = 0, LC2 = 0;             // optional second (coarser) latent level; 0 0 0 = off
    int LW3 = 0, LH3 = 0, LC3 = 0;             // optional third (coarser still) latent level; needs --latent2
    std::string filter = "bilinear";           // per-level sampling: "bilinear" or "nearest", comma-separated per level
    std::vector<int> hidden = { 24, 24 };  // MLP hidden layer widths
    float leak = 0.01f;                    // negative-side slope of the leaky ReLU (the only hidden activation since v0.8)
    std::string pos = "uv";    // positional encoding spec, see PosEnc
    bool clamp_out = false;    // hard clamp instead of sigmoid
    int iters = 3000;
    int save_every = 0;        // --save-every N: PNG snapshots every N iters; 0 (default) = two snapshots, at the middle and the end
    int print_every = 0;       // --print-every N: stats every N iters; 0 (default) = about once per second of wall time, plus the last iteration
    unsigned seed = 1;
    // MLP ES
    int mlp_pairs = 256;       // ES pairs per decoder step (256: on the GPU the extra pairs are nearly free and remove most of the ES-phase noise floor)
    int mlp_batch = 4096;
    float mlp_sigma = 0.02f;
    float mlp_lr = 0.005f;
    int mlp_every = 1;
    // Late-phase option: --mlp-fd START, a fraction of --iters (>= 1 = off). The phase is gated
    // by --mlp-every like the ES step.
    float mlp_fd_start = 1.0f;     // from here on, replace the MLP ES step with central finite differences
    float mlp_fd_h = 1e-3f;        // finite-difference step per weight
    bool mlp_fd_h_set = false;
    // latent ES
    int lat_pairs = 4;
    float lat_sigma = 0.05f;   // one sigma covers every ES-trained level
    float lat_lr = 0.02f;
    float lat_init = 0.1f;
    bool lat_init_image = false;   // [INIT] --lat-init-image: initialise channel 0 of level 0 from the image's luma (mean over textures) scaled to [-1, 1] instead of the Gaussian draw (the other channels and levels keep the draw); off = unchanged
    // Annealing: multiplier decays linearly from 1 at (start * iters) to final at the end.
    float lr_anneal_start = 1.0f, lr_anneal_final = 1.0f;       // applies to mlp_lr and lat_lr
    int threads = 0;
    int qbits = 8;             // latent bit depth used for the reported quantized stats
    int qat = 0;               // --qat: largest per-channel bit depth of level 0 (0 = off); level 0 is held on fixed grids and updated by exhaustive search
    std::vector<int> qat_ch;   // --qat B or B1,B2,...: bits per level-0 channel (a single value applies to every channel)
    int qat_every = 1;         // run that search every N iterations
    // --qes: quantization-aware ES for continuous (ES-trained) levels. Per level: bits (0 = plain
    // continuous); one entry applies to every level without --qat. Resolved to one entry per level
    // after the levels are known. See "QES" below.
    std::vector<int> qes_ch;
    bool qes = false;          // any level has --qes bits
    float qes_start = 0.5f;    // fraction of --iters at which the ranges are fitted, frozen and snapping begins (0 = from the start)
    // [DCT] --dct-q: level 0 is an 8x8 DCT-coded selector plane (see the "DCT" region). 0 = off: no behaviour change anywhere.
    int dct_q = 0;             // [DCT] 1..100 (JPEG quality; 100 = every AC step 1, near lossless; 1 = coarsest); the maximum of dct_q_ch (every `o.dct_q > 0` gate keys off it)
    std::vector<int> dct_q_ch; // [DCT] per level-0 channel (size 1 = every channel; expanded to o.LC after the layout is known), the --qat B1,B2 convention
    float dct_start = 0.5f;    // [DCT] fraction of --iters at which the scale codes are fitted, frozen, and snapping begins (0 = from the start)
    int dct_refit = 500;       // [DCT] --dct-refit N: re-probe the codes and re-snap every N iterations after the switch (0 = fit once, freeze)
    float dct_refit_until = 1.0f; // [DCT] --dct-refit-until F: no refit after ceil(F * iters), so the final codes are stable for the last stretch
    int dct_dc_step = 4;       // [DCT] uniform DC quantizer step in [0,64]-plane DC units (DC = 8 x block mean), independent of q
    int dct_deadzone = 0;      // [DCT] --dct-deadzone 0|1 (default 0 since v0.12.3, his call): 1 = XUASTC's dead-zone AC quantizer (alpha 0.5, first-order exemption); 0 = plain rounding on every AC (model file v14 / ntcb v2 carry the flag)
    bool dct_deadzone_given = false;   // [DCT] --dct-deadzone was given; a loaded DCT file's flag is adopted otherwise
    float dct_shadow_pull = 0.01f;     // [DCT] --dct-shadow-pull F (default 0.01 since v0.13, his call): after every latent step (plane live) move the level-0 shadow a fraction F toward its snapped value (sub-threshold coefficients decay instead of random-walking); 0 = off
    bool dct_shadow_reset = false;     // [DCT] --dct-shadow-reset: at every refit set the level-0 shadow equal to the snapped plane
    int dct_block = 8;         // [DCT] transform block size; 8 is the only value accepted in the MVP
    bool dct_selftest = false; // [DCT] --dct-selftest: run the transform / table / zigzag / quantizer self-test and exit
    bool dct_stats = false;    // [DCT] --dct-stats: print the full `dct <keyword>` statistics block at every progress print
    std::string dct_map = "nz"; // [DCT] --dct-map nz|code|bits|lnz|dconly|all: the per-block map PNG written with the snapshots
    bool dct_opts_given = false; // [DCT] any --dct-* option other than --dct-q was given (for the "ignored without --dct-q" note)
    float dct_lambda = -1.0f;     // [DCT] --dct-lambda L: rate term in the loss, L in LSB^2 per bpp (0 = off: no rate term, no truncation). Without the flag (dct_lambda_given false) the recipe is resolved after --dct-q is known: L = min(0.5, 7.5 S(q)^2) with S the libjpeg scale, the smallest over the channels (0.5 up to q ~87, 0.3 at 90, 0.075 at 95, 0 at 100)
    bool dct_lambda_given = false;    // [DCT] --dct-lambda was on the command line (any value; otherwise the recipe applies)
    bool dct_lambda_recipe = false;   // [DCT] true when dct_lambda came from the recipe (the settings line says so)
    std::string dct_rate = "both"; // [DCT] --dct-rate es|trunc|both: which of the two parts --dct-lambda enables (A: ES rate term, B: snap-time truncation)
    bool dct_rate_given = false;  // [DCT] --dct-rate was given (the "without --dct-lambda" note)
    float dct_lambda_lo = 0.25f;  // [DCT] --dct-lambda-lo W: the first-order pair's tokens ((0,1), (1,0): zigzag 1, 2) cost W x their bits in the proxy (both parts), 0..1; 1 = no protection
    std::string load;          // load model.bin instead of random init
    bool resave = false;       // --resave: with --load --iters 0, write <out>/model.bin in the current format
    bool write_ntcb = false;   // [NTCB] --write-ntcb: also write <out>/model.ntcb (the bit-packed container) at the end; the final MLP weights are rounded to fp16 first
    bool eval_filter = false;  // --eval-filter: after training, compare decoding with interpolated inputs against bilinear filtering of decoded texels
    bool rng_hash = false;     // --rng hash: draw the ES perturbations and minibatches from the counter-based hash the GPU uses
    bool cuda = false;         // --cuda: train on the GPU (needs a build with NTC_CUDA)
    bool cuda_check = false;   // --cuda-check: compare the GPU kernels against the CPU code, then exit
};

static void usage() {
    printf(
        "ntc [options] [tex0.png [tex1.png ...]]   (default: kodim23.png; up to 4 same-size RGB textures of one material share the latent and MLP)\n"
        "  --weights w0,w1,...  per-texture loss weight, one entry per image (default 1 each). Relative:\n"
        "                       loss = sum_t w_t mse_t / sum_t w_t, so 2,2 == 1,1; a weight of 0 drops that\n"
        "                       texture from training (its PSNR is then meaningless). Not stored in model.bin:\n"
        "                       pass the same --weights with --load to get the same reported psnr.\n"
        "  --out DIR            output directory (default out)\n"
        "  --crop N             center-crop to NxN before anything else; 0 = none, the whole image (0)\n"
        "  --block N            pad the image to a multiple of NxN by duplicating its last column / row, as\n"
        "                       BC/ASTC encoders do; the padded image is the source for everything after\n"
        "                       (all statistics, outputs, bitrates). With it, a latent W or H of 0 means\n"
        "                       the padded size (--latent) or the padded size / N (--latent2)\n"
        "  --latent W H C       latent texture size (64 64 4)\n"
        "  --latent2 W H C      optional second (typically coarser) latent level, e.g. 32 32 4 (off)\n"
        "  --latent3 W H C      optional third latent level (needs --latent2); with --block, 0 0 = padded size / (2N)\n"
        "  --filter M[,M]       latent sampling per level: bilinear | nearest (bilinear). With nearest,\n"
        "                       every pixel of a cell reads the same texel, so add a cell-position\n"
        "                       feature (lv1local) or the cell decodes to one color\n"
        "  --mlp W1,W2,...      MLP hidden layer widths (24,24); e.g. --mlp 32 or --mlp 16,16,16\n"
        "  --leak F             negative-side slope of the leaky ReLU (0.01)\n"
        "  --pos SPEC           positional features, comma list (uv). Kinds: uv | lv1local | none\n"
        "                         (uv: global u,v in [-1,1]; lv1local: the second level's cell offset in [-1,1])\n"
        "  --clamp              hard-clamp output instead of sigmoid\n"
        "  --iters N            training iterations (3000)\n"
        "  --save-every N       write PNG snapshots (recon + latents) every N iters; 0 = only at the middle and the end (0)\n"
        "  --print-every N      print stats every N iters; 0 = about once per second, plus the last iteration (0)\n"
        "  --seed N\n"
        "  --mlp-pairs N --mlp-batch N --mlp-sigma F --mlp-lr F --mlp-every N\n"
        "  --mlp-fd START       from iteration START*iters on, train the MLP with central\n"
        "                       finite differences (2 evals per weight) instead of ES (off)\n"
        "  --mlp-fd-h F         finite-difference step size (1e-3)\n"
        "  --lat-pairs N --lat-sigma F --lat-lr F --lat-init F\n"
        "  --lat-init-image     start channel 0 of level 0 at the image's luma (mean over textures) scaled to [-1, 1] (off: Gaussian)\n"   // [INIT]
        "  --lr-anneal START FINAL     decay both learning rates linearly from 1x at\n"
        "                              iteration START*iters to FINAL x at the end (off)\n"
        "  --threads N\n"
        "  --qbits N            latent bit depth for reported quantized bitrate/psnr (8)\n"
        "  --qat B | B0,B1[,B2,B3]  hold level 0 on a 2^B-value grid in [-1,1] (B = 1..8, one value or one per\n"   // [DCT] hook: was `--qat B[,B,...]` (one list notation with --qes and --dct-q)
        "                       channel) and update it by an exact exhaustive per-texel search instead of\n"
        "                       ES (off). Needs --filter nearest on level 0; level 1 (if\n"
        "                       any) and the MLP train as before\n"
        "  --qat-every N        run the level-0 search every N iterations (1); each run costs sum over channels of 2^B image decodes\n"
        "  --qes B | B0,B1[,B2] quantization-aware ES: hold every continuous (ES-trained) level, or the listed levels\n"
        "                       (0 = leave continuous), on a per-channel min/max grid of 2^B values (B = 2..12).\n"
        "                       The level keeps training by ES on an fp32 shadow; every decode reads the snapped\n"
        "                       copy, the ES perturbation is added to the snapped point and the update goes to\n"
        "                       the shadow. A --qat level cannot also have --qes (off)\n"
        "  --qes-start F        fit the per-channel ranges and start snapping at iteration F*iters (0.5; 0 = from the start);\n"
        "                       the ranges are frozen from then on and the shadow is clamped to them after every step\n"
        "  --dct-q Q | Q0,Q1[,Q2,Q3]  [experimental] level 0 is an 8x8 DCT-coded selector plane per channel (XUASTC's\n"   // [DCT]
        "                       quantizer: JPEG luma table, libjpeg quality scaling, dead zone, per-block 4-bit scale code from\n"   // [DCT]
        "                       a decoder probe). Q = 100: every AC step 1 (near lossless), 1 = coarsest; 0 = off; one value\n"   // [DCT]
        "                       applies to every channel, a list gives each of the 1..4 channels of --latent 0 0 C its own\n"   // [DCT]
        "                       quality (the --dct-dc-step DC step is shared by the channels). Level 0 has no bit depth then:\n"   // [DCT]
        "                       it is charged by a bit simulator over the zigzag / run-length symbols (nothing is coded).\n"   // [DCT]
        "                       Needs --filter nearest, level 0 at full resolution and an image size that is a multiple of 8\n"   // [DCT]
        "                       (the padding multiple becomes lcm(--block, 8)); replaces --qat on level 0 (off)\n"   // [DCT]
        "  --dct-start F        fit the scale codes and start decoding from the DCT-snapped plane at iteration F*iters\n"   // [DCT]
        "                       (0.5; 0 = from the start); the shadow is clamped to [-1,1] from the first iteration\n"   // [DCT]
        "  --dct-refit N        re-run the probe and replace the scale codes (re-snapping level 0, shadow kept) every N\n"   // [DCT]
        "                       iterations after the switch (500; 0 = fit once and freeze); the file stores the last codes\n"   // [DCT]
        "  --dct-refit-until F  no refit after iteration F*iters (1.0), so the codes settle over the last stretch of the run\n"   // [DCT]
        "  --dct-dc-step N      uniform DC step in [0,64]-plane DC units (4; DC = 8 x block mean; independent of Q)\n"   // [DCT]
        "  --dct-block N        DCT block size (8; only 8 in this version)\n"   // [DCT]
        "  --dct-deadzone 0|1   AC quantizer: 1 = XUASTC's dead zone (alpha 0.5, |d| < L -> 0, |q| = 1 dequantizes to 1.5 L; the\n"   // [DCT]
        "                       first-order pair (1,0), (0,1) rounds plainly), 0 = plain rounding round(d / L) and q * L on every AC (0; a loaded DCT file's flag\n"   // [DCT]
        "                       is adopted when the option is not given)\n"   // [DCT]
        "  --dct-shadow-pull F  after every latent step move the level-0 fp32 shadow a fraction F toward its snapped plane (default 0.01),\n"   // [DCT]
        "                       so coefficients below their quantizer step decay instead of random-walking across it under Adam; 0 = off\n"   // [DCT]
        "  --dct-shadow-reset   at every refit set the level-0 shadow equal to its snapped plane (the weaker form of the pull; off by default)\n"   // [DCT]
        "  --dct-stats          print the full `dct <keyword>` statistics block at every progress print (else only at the end)\n"   // [DCT]
        "  --dct-map KIND       per-block map PNG written with the snapshots: nz (default) | code | bits | lnz | dconly | all\n"   // [DCT]
        "  --dct-lambda L       rate term in the latent loss, loss = mse + L * proxy bits / pixel (L in LSB^2 per bpp; 0 = off; without the\n"   // [DCT]
        "                       flag the recipe L = min(0.5, 7.5 S(q)^2) applies, smallest over the level-0 channels: 0.5 up to q ~87, 0.3\n"   // [DCT]
        "                       at q 90, 0.075 at 95, 0 at 100): the ES\n"   // [DCT]
        "                       objective charges Exp-Golomb token costs of the perturbed shadow's symbols, and the snap walks every\n"   // [DCT]
        "                       nonzero AC from zigzag 63 downward and zeroes each one whose distortion increase is below lambda_t(k) =\n"   // [DCT]
        "                       L / g_k^2 times its exact bit saving; start at lambda*/4 ~ 7.5 S(q)^2 (20 at q 30, 80 at q 15, 200 at q 10)\n"   // [DCT]
        "  --dct-rate MODE      es | trunc | both (both): which of the two --dct-lambda parts is on (A: ES rate term, B: truncation)\n"   // [DCT]
        "  --dct-lambda-lo W    the two first-order ACs ((0,1), (1,0)) cost W x their bits in the rate proxy, W in [0, 1] (0.25): protects\n"   // [DCT]
        "                       the lowest frequencies from --dct-lambda (1 = no protection, 0 = never charged); both parts, both backends\n"   // [DCT]
        "  --dct-selftest       run the DCT transform / table / zigzag / quantizer self-test and exit\n"   // [DCT]
        "  --load model.bin     start from a saved model (use --iters 0 to just evaluate)\n"
        "  --resave             with --load --iters 0: also write <out>/model.bin in the current file format (v12; v14 with --dct-q)\n"   // [DCT] hook: v14 (was v13)
        "                       fast q sweep: --load <v12 model with a continuous level 0> --dct-q Q --iters 0 fits the scale codes,\n"
        "                       snaps the plane at Q and prints the statistics with the decoder frozen (about a second per Q)\n"   // [DCT]
        "  --write-ntcb         also write <out>/model.ntcb: the bit-packed container (no entropy coding; NTCB_PLAN.md) whose byte count\n"   // [NTCB]
        "                       is the raw bitrate; the final MLP weights are rounded to fp16 first so model.bin, the PNGs, the done:\n"   // [NTCB]
        "                       line and the file agree; the file: line reconciles its content bits with the simulator (off)\n"   // [NTCB]
        "  --rng MODE           ES noise source: mt (std::mt19937 + normal_distribution, platform-dependent) or\n"
        "                       hash (the GPU backend's counter-based generator; a CPU run then uses the same\n"
        "                       perturbations as a --cuda run with the same seed) (mt)\n"
        "  --eval-filter        evaluation only: at several sub-texel offsets, decode once per sample with the level-0\n"
        "                       (selector) inputs interpolated between the four nearest texels and compare against\n"
        "                       bilinear filtering of the decoded texels; PNGs go to <out>/filter/, stats to stdout\n"
        "  --cuda               train on the GPU (same outputs and model file)\n"
        "  --cuda-check         with --cuda: compare the GPU kernels against the CPU code and exit\n");
}

// ---------------------------------------------------------------- image
struct Image {
    int w = 0, h = 0, nc = 3;   // nc = 3 * textures; per pixel [r0 g0 b0 r1 g1 b1 ...]
    std::vector<float> rgb;     // w*h*nc, [0,1] (all textures interleaved, despite the name)
    float& at(int x, int y, int c) { return rgb[((size_t)y * w + x) * nc + c]; }
    float at(int x, int y, int c) const { return rgb[((size_t)y * w + x) * nc + c]; }
};

static bool load_png(const std::string& path, Image& img, int crop) {
    int w, h, n;
    unsigned char* data = stbi_load(path.c_str(), &w, &h, &n, 3);
    if (!data) return false;
    img.nc = 3;
    int cw = w, ch = h, ox = 0, oy = 0;
    if (crop > 0 && (w > crop || h > crop)) {
        cw = std::min(w, crop); ch = std::min(h, crop);
        ox = (w - cw) / 2; oy = (h - ch) / 2;
    }
    img.w = cw; img.h = ch; img.rgb.resize((size_t)cw * ch * 3);
    for (int y = 0; y < ch; y++)
        for (int x = 0; x < cw; x++)
            for (int c = 0; c < 3; c++)
                img.at(x, y, c) = data[((size_t)(y + oy) * w + (x + ox)) * 3 + c] / 255.0f;
    stbi_image_free(data);
    return true;
}

// Write texture t of img (any nc) as a 3-channel PNG. For nc == 3, t == 0 this
// is a plain RGB write.
static void save_png(const std::string& path, const Image& img, int t = 0) {
    std::vector<unsigned char> buf((size_t)img.w * img.h * 3);
    for (size_t p = 0; p < (size_t)img.w * img.h; p++)
        for (int c = 0; c < 3; c++)
            buf[p * 3 + c] = (unsigned char)std::lround(std::min(1.0f, std::max(0.0f, img.rgb[p * img.nc + 3 * t + c])) * 255.0f);
    stbi_write_png(path.c_str(), img.w, img.h, 3, buf.data(), img.w * 3);
}

// Write texture t of img cropped to the source extent sw x sh (the --block padding removed), so target.png and
// recon_q_final.png have the same size as the source and as ntc_decode's ntcb_decoded.png (which writes the source extent).
static void save_png_src(const std::string& path, const Image& img, int sw, int sh, int t = 0) {
    if (sw <= 0 || sh <= 0 || (sw == img.w && sh == img.h)) { save_png(path, img, t); return; }
    Image c; c.w = std::min(sw, img.w); c.h = std::min(sh, img.h); c.nc = img.nc; c.rgb.resize((size_t)c.w * c.h * c.nc);
    for (int y = 0; y < c.h; y++) memcpy(&c.rgb[(size_t)y * c.w * c.nc], &img.rgb[(size_t)y * img.w * c.nc], (size_t)c.w * c.nc * sizeof(float));
    save_png(path, c, t);
}

// Extend an image to Wp x Hp by repeating its last column and last row.
static void pad_image(Image& img, int Wp, int Hp) {
    if (Wp == img.w && Hp == img.h) return;
    Image out; out.w = Wp; out.h = Hp; out.nc = img.nc; out.rgb.resize((size_t)Wp * Hp * img.nc);
    for (int y = 0; y < Hp; y++) {
        const int sy = std::min(y, img.h - 1);
        for (int x = 0; x < Wp; x++) {
            const int sx = std::min(x, img.w - 1);
            for (int c = 0; c < img.nc; c++) out.rgb[((size_t)y * Wp + x) * img.nc + c] = img.rgb[((size_t)sy * img.w + sx) * img.nc + c];
        }
    }
    img = out;
}

// Interleave T same-size RGB images into one nc = 3T image.
static void pack_textures(const std::vector<Image>& tex, Image& out) {
    const int T = (int)tex.size();
    out.w = tex[0].w; out.h = tex[0].h; out.nc = 3 * T;
    out.rgb.resize((size_t)out.w * out.h * out.nc);
    for (size_t p = 0; p < (size_t)out.w * out.h; p++)
        for (int t = 0; t < T; t++)
            for (int c = 0; c < 3; c++) out.rgb[p * out.nc + 3 * t + c] = tex[t].rgb[p * 3 + c];
}

// Texture t of a packed image as a plain 3-channel image.
static void slice_texture(const Image& img, int t, Image& out) {
    out.w = img.w; out.h = img.h; out.nc = 3;
    out.rgb.resize((size_t)img.w * img.h * 3);
    for (size_t p = 0; p < (size_t)img.w * img.h; p++)
        for (int c = 0; c < 3; c++) out.rgb[p * 3 + c] = img.rgb[p * img.nc + 3 * t + c];
}

// Try the path as given, then relative to the executable's directory and its
// parents, so running from build/Release still finds the checked-in images.
// On success `name` is rewritten to the path that loaded.
static bool find_and_load(const char* argv0, std::string& name, Image& img, int crop) {
    std::vector<std::string> candidates = { name };
    std::string exe = argv0;
    size_t slash = exe.find_last_of("/\\");
    std::string dir = (slash == std::string::npos) ? "." : exe.substr(0, slash);
    for (int up = 0; up < 4; up++) { candidates.push_back(dir + "/" + name); dir += "/.."; }
    for (const std::string& c : candidates)
        if (load_png(c, img, crop)) { name = c; return true; }
    return false;
}

// ---------------------------------------------------------------- latent
struct Latent {
    int W = 0, H = 0, C = 0;
    size_t off = 0;        // index of this level's first value in LatentSet::z
    bool nearest = false;  // nearest-texel sampling instead of bilinear (a cell = a block)
    size_t size() const { return (size_t)W * H * C; }
};

// All latent levels in one flat parameter vector, so one Adam state and one ES
// epsilon cover everything. Level 0 is --latent (off = 0, guaranteed by add());
// level 1, if present, is --latent2 (typically coarser, but any size works).
// Level l occupies z[off, off + size()) in (y, x, c) order with c fastest.
struct LatentSet {
    std::vector<Latent> lv;
    std::vector<float> z;
    void add(int W, int H, int C, bool nearest = false) {
        Latent L; L.W = W; L.H = H; L.C = C; L.off = z.size(); L.nearest = nearest;
        lv.push_back(L);
        z.resize(z.size() + L.size());
    }
    size_t size() const { return z.size(); }
    int channels() const { int c = 0; for (const Latent& L : lv) c += L.C; return c; }
    const float* level(int l) const { return z.data() + lv[l].off; }
};

// ---------------------------------------------------------------- QES: quantization-aware ES
// --qes B: a continuous (ES-trained) level is held on a per-channel min/max grid of 2^B values,
//   g_c(k) = lo_c + k / (2^B - 1) * range_c,  range_c = max(hi_c - lo_c, 1e-6),  k = 0..2^B-1,
// the grid bitrate_stats() / ntc_decode apply post hoc at B = 8. The level keeps its fp32 shadow
// (LatentSet::z, Adam state unchanged) and a snapped copy zq = snap(z) that every decode reads
// (Decoder::zdec()). ES perturbs the snapped point (zp = zq + sigma * eps) and applies the update
// to the shadow: the straight-through estimator, independent of the grid step. Ranges are fitted
// from the shadow at --qes-start and frozen at once (a refit / freeze schedule was removed in
// v0.8, never used); once frozen the shadow is clamped to [lo, hi] after every Adam step. The functions below are compiled
// with strict floating point (no contraction, no reassociation) so the index and the grid values
// are the same bit patterns here, in cuda/ntc_cuda.cu (index recomputed with the same expression,
// grid values looked up from this table) and in ntc_decode.cpp.
#if defined(_MSC_VER) && !defined(__clang__)
#define STRICT_FP_BEGIN __pragma(float_control(precise, on, push)) __pragma(fp_contract(off))
#define STRICT_FP_END   __pragma(float_control(pop))
#elif defined(__clang__)
#define STRICT_FP_BEGIN _Pragma("float_control(precise, on, push)") _Pragma("clang fp contract(off)")   // [DCT] contract(off) added: the DCT has multiply-add pairs (not arithmetic-neutral for the qes grid on non-MSVC builds, which are not pinned)
#define STRICT_FP_END   _Pragma("float_control(pop)")
#else
#define STRICT_FP_BEGIN _Pragma("GCC push_options") _Pragma("GCC optimize(\"no-fast-math\")") _Pragma("GCC optimize(\"fp-contract=off\")")   // [DCT] fp-contract=off added, same note
#define STRICT_FP_END   _Pragma("GCC pop_options")
#endif

struct QesLevel {
    int bits = 0;                 // 0 = plain continuous level
    std::vector<float> lo, hi, range;   // per channel (size C when fitted)
    std::vector<float> grid;      // C x 2^bits grid values, channel c at c * 2^bits
    int levels() const { return (1 << bits) - 1; }
    bool fitted() const { return bits > 0 && !lo.empty(); }
    bool from_ntcb = false;       // [NTCB] a container's mode-0 (post-hoc) grid reinstalled by the loader: save_model writes it as a continuous level, ntcb_write as mode 0 again
};

STRICT_FP_BEGIN
// Grid index of a shadow value: round-half-even, as the device's __float2int_rn.
static inline int qes_index(float v, float lo, float range, int levels) {
    // Clamp in float before the cast: (int) of an out-of-range or NaN float is undefined on the
    // host while the device saturates, and the two must agree on every input.
    const float r = std::nearbyintf(((v - lo) / range) * (float)levels);
    return r <= 0.0f ? 0 : (r >= (float)levels ? levels : (int)r);
}
// Fit the per-channel min/max of level L from the shadow z (level base pointer) and rebuild the grid table.
static void qes_fit(QesLevel& Q, const Latent& L, const float* z) {
    const int levels = Q.levels();
    Q.lo.assign(L.C, 1e30f); Q.hi.assign(L.C, -1e30f); Q.range.assign(L.C, 0.0f);
    Q.grid.assign((size_t)L.C * (levels + 1), 0.0f);
    for (size_t i = 0; i < L.size(); i++) { const int c = (int)(i % L.C); Q.lo[c] = std::min(Q.lo[c], z[i]); Q.hi[c] = std::max(Q.hi[c], z[i]); }
    for (int c = 0; c < L.C; c++) {
        Q.range[c] = std::max(Q.hi[c] - Q.lo[c], 1e-6f);
        for (int k = 0; k <= levels; k++) Q.grid[(size_t)c * (levels + 1) + k] = Q.lo[c] + k / (float)levels * Q.range[c];
    }
}
// Rebuild the grid table from stored lo/hi (a loaded file).
static void qes_rebuild(QesLevel& Q, int C) {
    const int levels = Q.levels();
    Q.range.assign(C, 0.0f);
    Q.grid.assign((size_t)C * (levels + 1), 0.0f);
    for (int c = 0; c < C; c++) {
        Q.range[c] = std::max(Q.hi[c] - Q.lo[c], 1e-6f);
        for (int k = 0; k <= levels; k++) Q.grid[(size_t)c * (levels + 1) + k] = Q.lo[c] + k / (float)levels * Q.range[c];
    }
}
// [NTCB] The post-hoc quantization of a continuous level (bitrate_stats), in strict FP so that the value it charges and returns is
// [NTCB] the one both container readers reconstruct from the stored index: under /fp:fast the same expressions in bitrate_stats gave
// [NTCB] values an ulp off the strict ones (September 7, 2026; ntcb_write refuses such a level, and did, before this move).
static inline int q8_index(float z, float lo, float range, int levels) {   // [NTCB] was bitrate_stats' `lround((z - lo) / range * levels)`, clamped
    const int qi = (int)std::lround((z - lo) / range * levels);   // [NTCB]
    return std::max(0, std::min(levels, qi));   // [NTCB]
}   // [NTCB]
static inline float q8_value(int k, float lo, float range, int levels) { return lo + k / (float)levels * range; }   // [NTCB] was bitrate_stats' `lo + qi / (float)levels * range`
// zq = snap(z) for one level; with `clamp` the shadow is first clamped to [lo, hi] (frozen range).
// Returns how many snapped values changed.
static size_t qes_snap_level(const QesLevel& Q, const Latent& L, float* z, float* zq, bool clamp) {
    const int levels = Q.levels(); size_t changed = 0;
    for (size_t i = 0; i < L.size(); i++) {
        const int c = (int)(i % L.C);
        float v = z[i];
        if (clamp) { v = std::min(std::max(v, Q.lo[c]), Q.hi[c]); z[i] = v; }
        const float s = Q.grid[(size_t)c * (levels + 1) + qes_index(v, Q.lo[c], Q.range[c], levels)];
        if (s != zq[i]) { zq[i] = s; changed++; }
    }
    return changed;
}
STRICT_FP_END

static std::string qes_spec(const std::vector<int>& bits) {
    std::string r; for (size_t l = 0; l < bits.size(); l++) r += (l ? "," : "") + std::to_string(bits[l]); return r;
}

// Map a pixel (px,py) of a W x H image to latent-space texel coordinates.
// Bilinear: UV = pixel center in [0,1]; texel centers sit at (i+0.5)/LW and the
// tap blends the 4 surrounding texels. Nearest: the cell is the texel's own
// footprint [i/LW, (i+1)/LW), x0 == x1, and fx,fy is the offset within that
// cell in [0,1), so a cell behaves like a block of r x r pixels.
struct BilinearTap {
    int x0, x1, y0, y1;   // clamped texel indices
    float fx, fy;
};

static inline BilinearTap bilinear_tap(const Latent& L, float u, float v) {
    if (L.nearest) {
        float x = u * L.W, y = v * L.H;
        int ix = (int)std::floor(x), iy = (int)std::floor(y);
        BilinearTap t;
        t.fx = x - ix; t.fy = y - iy;
        t.x0 = t.x1 = std::max(0, std::min(L.W - 1, ix));
        t.y0 = t.y1 = std::max(0, std::min(L.H - 1, iy));
        return t;
    }
    float x = u * L.W - 0.5f, y = v * L.H - 0.5f;
    int x0 = (int)std::floor(x), y0 = (int)std::floor(y);
    BilinearTap t;
    t.fx = x - x0; t.fy = y - y0;
    t.x0 = std::max(0, std::min(L.W - 1, x0));
    t.x1 = std::max(0, std::min(L.W - 1, x0 + 1));
    t.y0 = std::max(0, std::min(L.H - 1, y0));
    t.y1 = std::max(0, std::min(L.H - 1, y0 + 1));
    return t;
}

static inline void sample_latent(const Latent& L, const float* z, const BilinearTap& t, float* out) {
    if (L.nearest) {
        const float* a = &z[((size_t)t.y0 * L.W + t.x0) * L.C];
        for (int k = 0; k < L.C; k++) out[k] = a[k];
        return;
    }
    const float* a = &z[((size_t)t.y0 * L.W + t.x0) * L.C];
    const float* b = &z[((size_t)t.y0 * L.W + t.x1) * L.C];
    const float* c = &z[((size_t)t.y1 * L.W + t.x0) * L.C];
    const float* d = &z[((size_t)t.y1 * L.W + t.x1) * L.C];
    float w00 = (1 - t.fx) * (1 - t.fy), w10 = t.fx * (1 - t.fy);
    float w01 = (1 - t.fx) * t.fy, w11 = t.fx * t.fy;
    for (int k = 0; k < L.C; k++)
        out[k] = w00 * a[k] + w10 * b[k] + w01 * c[k] + w11 * d[k];
}

// ---------------------------------------------------------------- MLP
// Fully connected net with an arbitrary list of hidden widths, e.g. {24,24}.
// Flat parameter layout, layer by layer: W[out*in] (row-major, one row per
// output unit) followed by b[out].
static const int MAXT = 4;        // max textures per material (all share the latent and the MLP)
static const int MAXOUT = 3 * MAXT; // max MLP outputs: 3 RGB channels per texture
static const int MAXH = 128;      // max units in any layer (incl. the input)
static const int MAXL = 8;        // max hidden layers

// Hidden activation: leaky ReLU (relu / tanh / sine were removed in v0.8; the model file's
// activation int is still written, always 0, and any other value is refused by the loader).
static inline float activate(float x, float leak) { return x > 0 ? x : leak * x; }

struct MLP {
    int nin = 0, nout = 3;
    std::vector<int> hidden;       // hidden layer widths
    float leak = 0.01f;            // leaky ReLU negative-side slope
    std::vector<float> p;
    size_t size() const { return p.size(); }

    // Widths of every layer, input first, output last.
    std::vector<int> widths() const {
        std::vector<int> w; w.push_back(nin);
        for (int h : hidden) w.push_back(h);
        w.push_back(nout);
        return w;
    }
    static size_t count(const std::vector<int>& w) {
        size_t n = 0;
        for (size_t l = 1; l < w.size(); l++) n += (size_t)w[l] * w[l - 1] + w[l];
        return n;
    }
    std::string describe() const {
        std::string s = std::to_string(nin);
        for (int h : hidden) s += " -> " + std::to_string(h);
        return s + " -> " + std::to_string(nout);
    }
    void init(int nin_, const std::vector<int>& hidden_, std::mt19937& rng) {
        nin = nin_; hidden = hidden_;
        std::vector<int> w = widths();
        p.assign(count(w), 0.0f);
        std::normal_distribution<float> N(0.0f, 1.0f);
        size_t o = 0;
        for (size_t l = 1; l < w.size(); l++) {
            bool last = (l + 1 == w.size());
            // He init for hidden layers, smaller for the output so it starts near mid-gray.
            float s = last ? std::sqrt(1.0f / w[l - 1]) : std::sqrt(2.0f / w[l - 1]);
            for (int i = 0; i < w[l] * w[l - 1]; i++) p[o++] = N(rng) * s;
            o += w[l]; // biases stay zero
        }
    }
};

// Forward pass with an explicit parameter pointer so perturbed copies can be used.
static inline void mlp_forward(const MLP& m, const float* p, const float* in, float* out, bool clamp_out) {
    float bufA[MAXH], bufB[MAXH];
    const float* cur = in;
    int ncur = m.nin;
    float* nxt = bufA;
    for (size_t l = 0; l < m.hidden.size(); l++) {
        int nh = m.hidden[l];
        const float* W = p; const float* b = W + (size_t)nh * ncur;
        for (int j = 0; j < nh; j++) {
            float s = b[j]; const float* w = W + (size_t)j * ncur;
            for (int i = 0; i < ncur; i++) s += w[i] * cur[i];
            nxt[j] = activate(s, m.leak);
        }
        p = b + nh;
        cur = nxt; ncur = nh;
        nxt = (nxt == bufA) ? bufB : bufA;
    }
    const float* W = p; const float* b = W + (size_t)m.nout * ncur;
    for (int j = 0; j < m.nout; j++) {
        float s = b[j]; const float* w = W + (size_t)j * ncur;
        for (int i = 0; i < ncur; i++) s += w[i] * cur[i];
        if (clamp_out) out[j] = std::min(1.0f, std::max(0.0f, s + 0.5f));
        else out[j] = 1.0f / (1.0f + std::exp(-s));
    }
}

// ---------------------------------------------------------------- DCT: transform-coded level 0 [DCT]
// [DCT] Experimental, off by default: with --dct-q absent (o.dct_q == 0) nothing in this region runs and
// [DCT] no hook outside it changes behaviour (DCT_MVP_PLAN.md section 7; DCT_NOTES.md has the removal list).
// [DCT] Level 0 (a full-resolution nearest selector level with C0 = 1..4 channels, each its own DCT-coded plane with
// [DCT] its own q, per-block scale codes and symbol stream; every array below is [block][channel]) is trained
// [DCT] through the --qes mechanism with a spatial shadow: the fp32 shadow lat.z is clamped to [-1,1],
// [DCT] every decode reads the snapped copy zq = clamp(IDCT(dequant(quant(DCT((s + 1) * 32)))) / 32 - 1)
// [DCT] per 8x8 block and channel, the ES perturbation is added to the snapped point and Adam updates the
// [DCT] shadow (Decoder::qes_refresh re-snaps after every latent step). The quantizer is XUASTC's
// [DCT] (basisu transcoder/basisu_transcoder.cpp compute_level_scale L27167-27206, sample_quant_table
// [DCT] L27208-27252, transcoder_internal.h quantize_deadzone / dequant_deadzone L1920-1962): JPEG Annex K
// [DCT] luminance table, libjpeg quality scaling S(q), [0,64] plane normalization, dead zone alpha 0.5 with
// [DCT] the first-order exemption ((1,0) and (0,1) round plainly), the DC coefficient (8 x block mean, range
// [DCT] [0,512]) at a fixed uniform step --dct-dc-step, and a per-block scale A_k = 1 / g_k from a 4-bit log
// [DCT] code k that the trainer fits once by probing the decoder's sensitivity to the selector
// [DCT] (dct_probe_codes). A DCT-coded level 0 has no bit depth: it is quantized only through the symbols.
// [DCT] The rate is a bit simulator over the DC symbol and the zigzag run-length tokens (dct_analyze):
// [DCT] fixed-length, order-0 and previous-nonzero-position context figures; nothing is entropy coded.
// [DCT] The transform and the quantizer are strict FP; their tables are float bit patterns copied verbatim
// [DCT] by cuda/ntc_cuda.cu (k_dct_snap / k_dct_recon) and ntc_decode.cpp (dct_recon_level0).
static const int MAX_DCT_CH = 4;            // [DCT] selector channels every table / file field is indexed for (--latent 0 0 C with C = 1..4; = ntc_cuda.h MAX_DCT_CH and ntc_decode's `dct_C <= 4`)
static const int DCT_N = 8;                 // [DCT] transform block size (the only value accepted in the MVP; --dct-block)
static const int DCT_NN = DCT_N * DCT_N;
static const int DCT_EOB = 64;              // [DCT] run symbol of the end-of-block token (XUASTC DCT_RUN_LEN_EOB_SYM_INDEX, internal.h L1536)
// [DCT] Orthonormal DCT-II basis with alpha folded in, B[u][x] = alpha(u) cos(pi (2x + 1) u / 16), alpha(0) =
// [DCT] sqrt(1/8), alpha(u > 0) = sqrt(2/8) (basisu dct2f::init L26613-26663), as 64 float bit patterns (the
// [DCT] double-precision products rounded once). Shared verbatim with ntc_decode.cpp and uploaded to the
// [DCT] device; never recomputed with cosf (the commit-4941c22 rule at the top of this file).
static const uint32_t DCT_BASIS_BITS[DCT_NN] = {
    0x3EB504F3u, 0x3EB504F3u, 0x3EB504F3u, 0x3EB504F3u, 0x3EB504F3u, 0x3EB504F3u, 0x3EB504F3u, 0x3EB504F3u,
    0x3EFB14BEu, 0x3ED4DB31u, 0x3E8E39DAu, 0x3DC7C5C2u, 0xBDC7C5C2u, 0xBE8E39DAu, 0xBED4DB31u, 0xBEFB14BEu,
    0x3EEC835Eu, 0x3E43EF15u, 0xBE43EF15u, 0xBEEC835Eu, 0xBEEC835Eu, 0xBE43EF15u, 0x3E43EF15u, 0x3EEC835Eu,
    0x3ED4DB31u, 0xBDC7C5C2u, 0xBEFB14BEu, 0xBE8E39DAu, 0x3E8E39DAu, 0x3EFB14BEu, 0x3DC7C5C2u, 0xBED4DB31u,
    0x3EB504F3u, 0xBEB504F3u, 0xBEB504F3u, 0x3EB504F3u, 0x3EB504F3u, 0xBEB504F3u, 0xBEB504F3u, 0x3EB504F3u,
    0x3E8E39DAu, 0xBEFB14BEu, 0x3DC7C5C2u, 0x3ED4DB31u, 0xBED4DB31u, 0xBDC7C5C2u, 0x3EFB14BEu, 0xBE8E39DAu,
    0x3E43EF15u, 0xBEEC835Eu, 0x3EEC835Eu, 0xBE43EF15u, 0xBE43EF15u, 0x3EEC835Eu, 0xBEEC835Eu, 0x3E43EF15u,
    0x3DC7C5C2u, 0xBE8E39DAu, 0x3ED4DB31u, 0xBEFB14BEu, 0x3EFB14BEu, 0xBED4DB31u, 0x3E8E39DAu, 0xBDC7C5C2u,
};
// [DCT] A_k = 1 / g_k, g_k = (14 / 64) 2^(0.6 k), k = 0..15 (A_0 = 64 / 14 = 4.571, the XUBC7 / XUASTC span floor
// [DCT] over 64; g_15 = 112 LSB per plane unit), as float bit patterns (double values rounded once); shared with
// [DCT] ntc_decode.cpp, which rebuilds the same integer step tables from (q, N, dc_step, k).
static const uint32_t DCT_AK_BITS[16] = {
    0x40924925u, 0x40410671u, 0x3FFEB2BFu, 0x3FA809C5u, 0x3F5DBA49u, 0x3F124925u, 0x3EC10671u, 0x3E7EB2BFu,
    0x3E2809C5u, 0x3DDDBA49u, 0x3D924925u, 0x3D410671u, 0x3CFEB2BFu, 0x3CA809C5u, 0x3C5DBA49u, 0x3C124925u,
};
// [DCT] basisu g_baseline_jpeg_y (transcoder.cpp L26932-26943): JPEG Annex K Table K.1 with the DC entry 4.
// [DCT] Row = vertical frequency u, column = horizontal frequency v (basisu samples [ry][rx]); the DC entry is
// [DCT] never read here (the DC has its own step).
static const int DCT_JPEG_Y[8][8] = {
    {  4, 11, 10, 16, 24, 40, 51, 61 },
    { 12, 12, 14, 19, 26, 58, 60, 55 },
    { 14, 13, 16, 24, 40, 57, 69, 56 },
    { 14, 17, 22, 29, 51, 87, 80, 62 },
    { 18, 22, 37, 56, 68,109,103, 77 },
    { 24, 35, 55, 64, 81,104,113, 92 },
    { 49, 64, 78, 87,103,121,120,101 },
    { 72, 92, 95, 98,112,100,103, 99 },
};
static float DCT_BASIS[DCT_N][DCT_N];   // [DCT] filled from DCT_BASIS_BITS by dct_init_tables()
static float DCT_AK[16];                // [DCT] filled from DCT_AK_BITS
static int DCT_ZIGZAG[DCT_NN];          // [DCT] zigzag position -> natural index u * 8 + v (basisu generate_zigzag_order L26875-26930, W = H = 8)
static bool DCT_TABLES_READY = false;
static void dct_init_tables() {
    if (DCT_TABLES_READY) return;
    memcpy(&DCT_BASIS[0][0], DCT_BASIS_BITS, sizeof(DCT_BASIS));
    memcpy(DCT_AK, DCT_AK_BITS, sizeof(DCT_AK));
    int idx = 0;
    for (int s = 0; s < 2 * DCT_N - 1; s++) {   // anti-diagonals, direction alternating (odd diagonals reversed)
        const int x_start = (s < DCT_N) ? 0 : (s - DCT_N + 1), x_end = (s < DCT_N) ? s : (DCT_N - 1);
        const int n = x_end - x_start + 1;
        int diag[DCT_N];
        for (int x = x_start, j = 0; x <= x_end; x++, j++) diag[j] = x + (s - x) * DCT_N;   // x = column (v), y = s - x = row (u)
        if (s & 1) for (int k = n - 1; k >= 0; k--) DCT_ZIGZAG[idx++] = diag[k];
        else for (int k = 0; k < n; k++) DCT_ZIGZAG[idx++] = diag[k];
    }
    DCT_TABLES_READY = true;
}

// [DCT] State of a DCT-coded level 0 (on Decoder). Every array is indexed [block][channel] (block raster order,
// [DCT] channel fastest); C0 = 1..4 since section 7 of DCT_NOTES.md, which lifted the MVP's C0 == 1 refusal without a layout change.
struct DctLevel {
    bool on = false;            // --dct-q given: level 0 is DCT-coded; the shadow is clamped to [-1,1] from the start
    bool live = false;          // codes fitted (or restored): every decode reads the snapped plane
    int N = DCT_N, dc_step = 4;
    bool deadzone = true;       // [DCT] --dct-deadzone: true = the dead-zone AC quantizer with the first-order exemption, false = plain rounding on every AC
    std::vector<int> q;         // per channel (1..100)
    int BW = 0, BH = 0, C = 0;  // blocks across / down, channels
    std::vector<int16_t> sym;   // [BH*BW][C][64] natural order [u * 8 + v]; [0] = DC symbol
    std::vector<uint8_t> code;  // [BH*BW][C] scale code k = 0..15
    std::vector<int> step;      // [C][16][64] integer steps; entry [..][0] = dc_step (the DC never reads it)
    int dc_raw_bits = 8;        // ceil(log2(floor(512 / dc_step) + 1)): fixed-length bits of the DC symbol
    int fitted_at = -1;         // iteration the codes were fitted at (-1: restored from a v13 file)
    int refit_every = 0;        // [DCT] --dct-refit: re-probe and re-snap every N iterations after the switch (0 = frozen)
    int refit_until_it = -1;    // [DCT] --dct-refit-until: last iteration a refit may happen at
    int refits = 0, refitted_at = -1;   // [DCT] refits so far and the iteration of the last one
    size_t last_clamped = 0;    // texels whose IDCT output fell outside [-1,1] in the last snap / recon (ringing indicator)
    size_t last_clamped_c[MAX_DCT_CH] = { 0, 0, 0, 0 };   // [DCT] the same per channel (the dct[c] lines; sums to last_clamped)
    // [DCT] --dct-lambda (DCT_RATE_PLAN.md): lambda in LSB^2 per bpp; rate_es = part A (the ES rate term), rate_trunc = part B
    // [DCT] (snap-time truncation); lam_t16[k] = lambda / (16 g_k^2) per scale code (multiplies the 1/16-bit saving directly);
    // [DCT] bits = the proxy bits of every snapped (block, channel) in 1/16 bit; the truncation counts of the last snap and the
    // [DCT] hit count of the last latent step (pair evaluations whose bit difference was nonzero: the signal of part A).
    float lambda = 0.0f; bool rate_es = false, rate_trunc = false; float lam_t16[16] = { 0 };   // [DCT]
    int lo_w16 = 16;   // [DCT] --dct-lambda-lo W as round(16 W): the first-order pair's token cost is (token * lo_w16 + 8) >> 4 in both parts (16 = unweighted, byte-identical to before)
    std::vector<int32_t> bits;   // [DCT] [block][channel] proxy bits of the snapped block, 1/16 bit (filled by the snap and the recon when lambda > 0)
    size_t last_truncated = 0, last_truncated_c[MAX_DCT_CH] = { 0, 0, 0, 0 }, last_nz_before = 0;   // [DCT] ACs zeroed by part B in the last snap (total, per channel) and the nonzero ACs before it
    size_t rate_hits = 0, rate_evals = 0;   // [DCT] last latent step: (block, channel, pair) evaluations with a nonzero bit difference, and their count
    size_t nblk() const { return (size_t)BW * BH; }
    size_t ntex() const { return nblk() * DCT_NN; }
    const int* steps(int c, int k) const { return &step[((size_t)c * 16 + k) * DCT_NN]; }
    int16_t* symbols(size_t b, int c) { return &sym[(b * C + c) * DCT_NN]; }
    const int16_t* symbols(size_t b, int c) const { return &sym[(b * C + c) * DCT_NN]; }
};

// [DCT] Strict floating point: the step tables, the transform and the quantizer must produce the same bit
// [DCT] patterns here, in ntc_decode.cpp (same source text under the same macro) and on the device (every
// [DCT] product and sum written with __fmul_rn / __fadd_rn so nvcc cannot contract the multiply-add pairs).
STRICT_FP_BEGIN
// libjpeg quality scaling (XUASTC compute_level_scale L27177-27183): S = q < 50 ? 5000 / q : 200 - 2q, over 100.
static float dct_quality_scale(int q) {
    q = std::max(1, std::min(100, q));
    float S = q < 50 ? 5000.0f / (float)q : 200.0f - 2.0f * (float)q;
    S *= (1.0f / 100.0f);
    return S;
}
// level_scale_k = S(q) * A_k (L27201 with the span term replaced by the block gain g_k; no lattice factor).
static float dct_level_scale(int q, int k) { return dct_quality_scale(q) * DCT_AK[k]; }
// One 8x8 step table for a level_scale: sample_quant_table (L27208-27252) with sx = sy = 8 / N (the identity at
// N = 8), entry = max(1, (int)(base * level_scale + 0.5)), every AC 1 at q >= 100 (compute_quant_table L27262-27269);
// entry [0] = dc_step (the DC has its own uniform step; it is not K.1's 16 nor basisu's 4).
static void dct_build_steps_scaled(int q, float level_scale, int N, int dc_step, int* out /* N * N, [u * N + v] */) {
    const float sx = 8.0f / (float)N;
    for (int u = 0; u < N; u++)
        for (int v = 0; v < N; v++) {
            if (u == 0 && v == 0) { out[0] = dc_step; continue; }
            if (q >= 100) { out[u * N + v] = 1; continue; }
            float i = std::min((float)v * sx, 7.0f), j = std::min((float)u * sx, 7.0f);   // i = table column (rx), j = table row (ry)
            const int i0 = (int)i, j0 = (int)j, i1 = std::min(i0 + 1, 7), j1 = std::min(j0 + 1, 7);
            const float ti = i - (float)i0, tj = j - (float)j0;
            const float a = (1.0f - ti) * (float)DCT_JPEG_Y[j0][i0] + ti * (float)DCT_JPEG_Y[j0][i1];
            const float b = (1.0f - ti) * (float)DCT_JPEG_Y[j1][i0] + ti * (float)DCT_JPEG_Y[j1][i1];
            const float base = (1.0f - tj) * a + tj * b;
            float sf = base * level_scale + 0.5f;
            if (!(sf >= 1.0f)) sf = 1.0f;              // (int) of a NaN / out-of-range float is undefined: clamp in float first
            if (sf > 1048576.0f) sf = 1048576.0f;
            out[u * N + v] = std::max(1, (int)sf);
        }
}
// The 16 step tables of one channel (one per scale code).
static void dct_build_steps(int q, int N, int dc_step, int* out /* 16 * N * N */) {
    for (int k = 0; k < 16; k++) dct_build_steps_scaled(q, dct_level_scale(q, k), N, dc_step, out + (size_t)k * N * N);
}
// Forward / inverse 8x8 DCT with the basis products folded in (basisu dct2f::forward L26680-26721, alpha applied
// per pass there, here inside the table). w[x * 8 + y]: x = row (vertical), y = column; C[u * 8 + v]: u = vertical
// frequency. Fixed evaluation order: sequential float sums from 0.0f, inner index ascending; mirrored verbatim.
static void dct_fwd8(const float* w, float* C) {
    float T[DCT_NN];
    for (int x = 0; x < 8; x++)
        for (int v = 0; v < 8; v++) { float s = 0.0f; for (int y = 0; y < 8; y++) s += w[x * 8 + y] * DCT_BASIS[v][y]; T[x * 8 + v] = s; }
    for (int u = 0; u < 8; u++)
        for (int v = 0; v < 8; v++) { float s = 0.0f; for (int x = 0; x < 8; x++) s += T[x * 8 + v] * DCT_BASIS[u][x]; C[u * 8 + v] = s; }
}
static void dct_inv8(const float* C, float* w) {
    float T[DCT_NN];
    for (int x = 0; x < 8; x++)
        for (int v = 0; v < 8; v++) { float s = 0.0f; for (int u = 0; u < 8; u++) s += DCT_BASIS[u][x] * C[u * 8 + v]; T[x * 8 + v] = s; }
    for (int x = 0; x < 8; x++)
        for (int y = 0; y < 8; y++) { float s = 0.0f; for (int v = 0; v < 8; v++) s += DCT_BASIS[v][y] * T[x * 8 + v]; w[x * 8 + y] = s; }
}
// Float clamp that maps NaN to lo (the same expression on the device), used before every (int) cast.
static inline float dct_clampf(float v, float lo, float hi) { if (!(v >= lo)) return lo; if (v > hi) return hi; return v; }
// DC: uniform, no dead zone, round-half-even (nearbyintf / __float2int_rn), dequant q * S.
static inline int dct_quant_dc(float d, int S) { return (int)dct_clampf(std::nearbyintf(d / (float)S), -1024.0f, 1024.0f); }
static inline float dct_dequant_dc(int q, int S) { return (float)q * (float)S; }
// AC (XUASTC quantize_deadzone, alpha = 0.5): the first-order pair (1,0), (0,1) rounds plainly (roundf, half away
// from zero on the CRT and CUDA); otherwise s = |d|, tau = 0.5 L, s <= tau -> 0, else floor((s - tau) / L + 0.5) with
// the sign, i.e. the zero bin is |d| < L, |q| = 1 covers [L, 2L) and dequantizes to the bin centre tau + |q| L.
// [DCT] dz = false (--dct-deadzone 0): every AC takes the first-order pair's path, roundf(d / L) with the +-1024 clamp and q * L back.
static inline bool dct_first_order(int u, int v) { return (u == 1 && v == 0) || (u == 0 && v == 1); }
static inline int dct_quant_ac(float d, int L, int u, int v, bool dz) {   // [DCT] hook: the dz parameter (--dct-deadzone)
    if (!dz || dct_first_order(u, v)) return (int)dct_clampf(std::roundf(d / (float)L), -1024.0f, 1024.0f);   // [DCT] hook: `!dz ||`
    const float s = std::fabsf(d), tau = 0.5f * (float)L;
    if (s <= tau) return 0;
    const float qf = (s - tau) / (float)L;
    const int q = (int)dct_clampf(std::floorf(qf + 0.5f), 0.0f, 1024.0f);
    return d < 0.0f ? -q : q;
}
static inline float dct_dequant_ac(int q, int L, int u, int v, bool dz) {   // [DCT] hook: the dz parameter (--dct-deadzone)
    if (!dz || dct_first_order(u, v)) return (float)q * (float)L;   // [DCT] hook: `!dz ||`
    if (q == 0) return 0.0f;
    const float tau = 0.5f * (float)L;
    const float mag = tau + (float)std::abs(q) * (float)L;
    return q < 0 ? -mag : mag;
}
// Dequantize one block's symbols and reconstruct its plane values w' (before the / 32 - 1 mapping).
static inline void dct_dequant_block(const int16_t* sym, const int* st, int dc_step, float* Cf, bool dz) {   // [DCT] hook: dz
    for (int k = 0; k < DCT_NN; k++) Cf[k] = k == 0 ? dct_dequant_dc(sym[0], dc_step) : dct_dequant_ac(sym[k], st[k], k / 8, k % 8, dz);   // [DCT] hook: dz
}
// [DCT] --dct-lambda, the token cost model (DCT_RATE_PLAN.md 2.2; integer, 1/16 bit; mirrored verbatim by dev_dct_* on the
// [DCT] device): Exp-Golomb code lengths 2 floor(log2(v)) + 1 for the run (v = run + 1) and the magnitude (v = |q|, capped at
// [DCT] 256 as in dct_analyze), 1 bit per sign, 2 bits per EOB (absent when the last nonzero sits at zigzag 63); the DC is
// [DCT] excluded. Keep in step with dct_analyze pass 2 and ntcb mode 3 (the same tokens; only the costs differ).
static const int DCT_C_SIGN = 16, DCT_C_EOB = 32;   // [DCT]
static inline int dct_floor_log2(unsigned v) { int k = 0; while (v >>= 1) k++; return k; }   // [DCT] v >= 1
static inline int dct_c_run(int r) { return 16 * (2 * dct_floor_log2((unsigned)(r + 1)) + 1); }   // [DCT] r = 0..62
static inline int dct_c_mag(int m) { return 16 * (2 * dct_floor_log2((unsigned)m) + 1); }   // [DCT] m = |q| >= 1
static inline int dct_lo_weight(int tok, int p, int lo16) { return p <= 2 ? (tok * lo16 + 8) >> 4 : tok; }   // [DCT] --dct-lambda-lo: the first-order pair (zigzag 1, 2) is charged lo16/16 of its token
static int dct_block_bits(const int16_t* sym, int lo16) {   // [DCT] the proxy bits of one (block, channel)'s 63 ACs in zigzag order
    int bits = 0, run = 0, lnz = 0;   // [DCT]
    for (int p = 1; p < DCT_NN; p++) {   // [DCT]
        const int q = sym[DCT_ZIGZAG[p]];   // [DCT]
        if (q == 0) { run++; continue; }   // [DCT]
        bits += dct_lo_weight(dct_c_run(run) + dct_c_mag(std::min(256, std::abs(q))) + DCT_C_SIGN, p, lo16);   // [DCT]
        lnz = p; run = 0;   // [DCT]
    }   // [DCT]
    if (lnz < DCT_NN - 1) bits += DCT_C_EOB;   // [DCT]
    return bits;   // [DCT]
}   // [DCT]
// [DCT] Part B (2.4): walk the nonzero ACs from zigzag 63 downward and zero every one whose exact plane-domain distortion
// [DCT] increase dD = c^2 - (c - v)^2 (c = the coefficient, v = its dequantized value) is below lam16 times its exact token
// [DCT] saving (its own token, the run merge into the next surviving token, the EOB that appears when zigzag 63 empties).
// [DCT] Greedy, one pass, never revisits. The three float operations are the parity surface (strict FP here, _rn on the device).
static int dct_truncate_block(const float* Cf, const int* st, float lam16, int lo16, bool dz, int16_t* sym) {   // [DCT] hook: dz (the dD below uses whichever dequantization the level runs)
    int ntrunc = 0, next = DCT_NN;   // [DCT] next = nearest surviving nonzero above p (64 = none: p is the last nonzero)
    for (int p = DCT_NN - 1; p >= 1; p--) {   // [DCT]
        const int k = DCT_ZIGZAG[p], q = sym[k];   // [DCT]
        if (q == 0) continue;   // [DCT]
        int prev = 0;   // [DCT] nearest nonzero below p in the current symbols (0 = none)
        for (int j = p - 1; j >= 1; j--) if (sym[DCT_ZIGZAG[j]] != 0) { prev = j; break; }   // [DCT]
        const int r1 = p - prev - 1;   // [DCT]
        const int token = dct_lo_weight(dct_c_run(r1) + dct_c_mag(std::min(256, std::abs(q))) + DCT_C_SIGN, p, lo16);   // [DCT] the pair's own token is weighted; the next token's run change is not
        const int saved = next == DCT_NN ? token - (p == DCT_NN - 1 ? DCT_C_EOB : 0)   // [DCT] last nonzero: an EOB appears iff p was 63
                                         : token + dct_c_run(next - p - 1) - dct_c_run(next - p - 1 + r1 + 1);   // [DCT] interior: the next token's run grows
        const float v = dct_dequant_ac(q, st[k], k / 8, k % 8, dz), c = Cf[k];   // [DCT] hook: dz
        const float dD = c * c - (c - v) * (c - v);   // [DCT] exact distortion increase in plane units^2 (contraction off in this block)
        if (dD < lam16 * (float)saved) { sym[k] = 0; ntrunc++; }   // [DCT]
        else next = p;   // [DCT]
    }   // [DCT]
    return ntrunc;   // [DCT]
}   // [DCT]
// [DCT] The quantize loop of dct_snap_block plus part B; with_bits: return dct_block_bits of the final symbols (the ES rate
// [DCT] evaluations), else 0 (the snap: dct_recon_block fills Q.bits; the lambda 0 path is the old one-line loop and nothing
// [DCT] more). ntrunc / nz_before receive the truncation count and the nonzero ACs before it (0 when lam16 == 0).
static int dct_quantize_block(const float* Cf, const int* st, int dc_step, float lam16, int lo16, bool dz, bool with_bits, int16_t* sym, int* ntrunc, int* nz_before) {   // [DCT] hook: dz (--dct-deadzone)
    for (int k = 0; k < DCT_NN; k++) sym[k] = (int16_t)(k == 0 ? dct_quant_dc(Cf[0], dc_step) : dct_quant_ac(Cf[k], st[k], k / 8, k % 8, dz));   // [DCT] was the one-line quantize loop of dct_snap_block; hook: dz
    int nt = 0, nz = 0;   // [DCT]
    if (lam16 > 0.0f) {   // [DCT]
        for (int k = 1; k < DCT_NN; k++) if (sym[k] != 0) nz++;   // [DCT]
        nt = dct_truncate_block(Cf, st, lam16, lo16, dz, sym);   // [DCT] hook: dz
    }   // [DCT]
    if (ntrunc) *ntrunc = nt;   // [DCT]
    if (nz_before) *nz_before = nz;   // [DCT]
    return with_bits ? dct_block_bits(sym, lo16) : 0;   // [DCT]
}   // [DCT]
// [DCT] Part A (2.3): the proxy bits of snap(clamp(z + sign * sg * eps)) for one (block, channel), evaluated on the perturbed
// [DCT] shadow z (not the snapped copy, whose perturbation never crosses a bin edge); z and eps are level-0 base pointers.
static int dct_rate_block(const DctLevel& Q, const Latent& L0, size_t b, int c, const float* z, const float* eps, float sg, float sign) {   // [DCT]
    float w[DCT_NN], Cf[DCT_NN]; int16_t sym[DCT_NN];   // [DCT]
    const int bx = (int)(b % Q.BW), by = (int)(b / Q.BW);   // [DCT]
    for (int y = 0; y < 8; y++)   // [DCT]
        for (int x = 0; x < 8; x++) {   // [DCT]
            const size_t i = ((size_t)(by * 8 + y) * L0.W + (bx * 8 + x)) * L0.C + c;   // [DCT]
            const float s = dct_clampf(z[i] + sign * (sg * eps[i]), -1.0f, 1.0f);   // [DCT] = k_dct_rate_pair's __fadd_rn / __fsub_rn of __fmul_rn(sg, e)
            w[y * 8 + x] = (s + 1.0f) * 32.0f;   // [DCT]
        }   // [DCT]
    dct_fwd8(w, Cf);   // [DCT]
    const int code = Q.code[b * Q.C + c];   // [DCT]
    return dct_quantize_block(Cf, Q.steps(c, code), Q.dc_step, Q.lam_t16[code], Q.lo_w16, Q.deadzone, true, sym, nullptr, nullptr);   // [DCT] hook: Q.deadzone
}   // [DCT]
// Reconstruct one block and channel of zq (level base pointer, (y, x, c) layout) from its symbols: dequantize,
// inverse DCT, zq = clamp(w' / 32 - 1). Returns the number of clamped texels; changed counts zq values that moved.
static size_t dct_recon_block(DctLevel& Q, const Latent& L0, size_t b, int c, float* zq, size_t* changed) {   // [DCT] hook: Q non-const (was const DctLevel&) for the bits fill below
    float Cf[DCT_NN], w[DCT_NN];
    if (Q.lambda > 0.0f && Q.bits.size() == Q.nblk() * (size_t)Q.C) Q.bits[b * Q.C + c] = dct_block_bits(Q.symbols(b, c), Q.lo_w16);   // [DCT] the proxy bits of the file's / the snap's symbols (so --load --iters 0 prints the dct rate line)
    dct_dequant_block(Q.symbols(b, c), Q.steps(c, Q.code[b * Q.C + c]), Q.dc_step, Cf, Q.deadzone);   // [DCT] hook: Q.deadzone
    dct_inv8(Cf, w);
    const int bx = (int)(b % Q.BW), by = (int)(b / Q.BW);
    size_t nclamp = 0;
    for (int y = 0; y < 8; y++)
        for (int x = 0; x < 8; x++) {
            const size_t i = ((size_t)(by * 8 + y) * L0.W + (bx * 8 + x)) * L0.C + c;
            const float raw = w[y * 8 + x] / 32.0f - 1.0f;
            const float v = dct_clampf(raw, -1.0f, 1.0f);
            if (v != raw) nclamp++;
            if (v != zq[i]) { zq[i] = v; if (changed) (*changed)++; }
        }
    return nclamp;
}
// Snap one block and channel: clamp the 64 shadow values to [-1,1] in place, w = (s + 1) * 32, forward DCT, quantize
// (DC at dc_step, ACs with the block's step table), store the symbols, then reconstruct through dct_recon_block.
static size_t dct_snap_block(DctLevel& Q, const Latent& L0, size_t b, int c, float* z, float* zq, size_t* changed, int* ntrunc = nullptr, int* nz_before = nullptr) {   // [DCT] hook: the two truncation counts (--dct-lambda)
    float w[DCT_NN], Cf[DCT_NN];
    const int bx = (int)(b % Q.BW), by = (int)(b / Q.BW);
    for (int y = 0; y < 8; y++)
        for (int x = 0; x < 8; x++) {
            const size_t i = ((size_t)(by * 8 + y) * L0.W + (bx * 8 + x)) * L0.C + c;
            const float s = dct_clampf(z[i], -1.0f, 1.0f);
            z[i] = s;
            w[y * 8 + x] = (s + 1.0f) * 32.0f;
        }
    dct_fwd8(w, Cf);
    const int code = Q.code[b * Q.C + c];   // [DCT]
    const int* st = Q.steps(c, code);   // [DCT] hook: was Q.steps(c, Q.code[b * Q.C + c])
    int16_t* sym = Q.symbols(b, c);
    dct_quantize_block(Cf, st, Q.dc_step, Q.lam_t16[code], Q.lo_w16, Q.deadzone, false, sym, ntrunc, nz_before);   // [DCT] hook: was the one-line quantize loop (now inside dct_quantize_block, followed by part B when lam_t16[code] > 0); Q.deadzone
    return dct_recon_block(Q, L0, b, c, zq, changed);
}
// The whole level (blocks independent, parallel): the qes_snap_level contract (returns changed zq values).
static size_t dct_snap_level0(DctLevel& Q, const Latent& L0, float* z, float* zq) {
    size_t changed = 0, nclamp = 0, ntrunc = 0, nzb = 0;   // [DCT] hook: ntrunc, nzb (--dct-lambda truncation counts) added to the reduction
    size_t nclamp_c[MAX_DCT_CH] = { 0, 0, 0, 0 };   // [DCT] per-channel clamp count (integer sums: order-independent)
    size_t ntrunc_c[MAX_DCT_CH] = { 0, 0, 0, 0 };   // [DCT] per-channel truncation count
    const long long nb = (long long)Q.nblk();
#pragma omp parallel for schedule(static) reduction(+:changed,nclamp,ntrunc,nzb)   // [DCT] hook: was reduction(+:changed,nclamp)
    for (long long b = 0; b < nb; b++)
        for (int c = 0; c < Q.C; c++) {   // [DCT] hook: was the one-line channel loop
            size_t ch = 0; int nt = 0, nz0 = 0;   // [DCT] hook: nt / nz0 = this (block, channel)'s truncation count and pre-truncation nonzero ACs
            const size_t nc = dct_snap_block(Q, L0, (size_t)b, c, z, zq, &ch, &nt, &nz0); nclamp += nc; changed += ch;   // [DCT] hook: nc kept for the per-channel count
            ntrunc += (size_t)nt; nzb += (size_t)nz0;   // [DCT]
#pragma omp atomic   // [DCT]
            nclamp_c[c] += nc;   // [DCT]
#pragma omp atomic   // [DCT]
            ntrunc_c[c] += (size_t)nt;   // [DCT]
        }   // [DCT]
    Q.last_clamped = nclamp;
    for (int c = 0; c < MAX_DCT_CH; c++) Q.last_clamped_c[c] = nclamp_c[c];   // [DCT]
    Q.last_truncated = ntrunc; Q.last_nz_before = nzb;   // [DCT]
    for (int c = 0; c < MAX_DCT_CH; c++) Q.last_truncated_c[c] = ntrunc_c[c];   // [DCT]
    return changed;
}
// zq level 0 from the symbols alone (the loader path; no shadow involved).
static void dct_recon_level0(DctLevel& Q, const Latent& L0, float* zq) {
    size_t nclamp = 0;
    size_t nclamp_c[MAX_DCT_CH] = { 0, 0, 0, 0 };   // [DCT]
    const long long nb = (long long)Q.nblk();
#pragma omp parallel for schedule(static) reduction(+:nclamp)
    for (long long b = 0; b < nb; b++)
        for (int c = 0; c < Q.C; c++) {   // [DCT] hook: was the one-line channel loop
            const size_t nc = dct_recon_block(Q, L0, (size_t)b, c, zq, nullptr); nclamp += nc;   // [DCT] hook: nc kept for the per-channel count
#pragma omp atomic   // [DCT]
            nclamp_c[c] += nc;   // [DCT]
        }   // [DCT]
    Q.last_clamped = nclamp;
    for (int c = 0; c < MAX_DCT_CH; c++) Q.last_clamped_c[c] = nclamp_c[c];   // [DCT]
}
// The pre-start clamp of the shadow's level 0 to [-1,1] (applied on the device by k_dct_clamp0).
static void dct_clamp_level0(const Latent& L0, float* z) {
    for (size_t i = 0; i < L0.size(); i++) z[i] = dct_clampf(z[i], -1.0f, 1.0f);
}
STRICT_FP_END

// [DCT] Scale code of a block gain g (8-bit LSB per [0,64]-plane unit): k = clamp(lround(log2(max(g, g_floor) / g_floor) / 0.6), 0, 15).
static const double DCT_G_FLOOR = 14.0 / 64.0;
static int dct_code_of_gain(double g) {
    if (!(g > DCT_G_FLOOR)) g = DCT_G_FLOOR;
    const long k = std::lround(std::log2(g / DCT_G_FLOOR) / 0.6);
    return (int)std::max(0L, std::min(15L, k));
}
static double dct_gain_of_code(int k) { return DCT_G_FLOOR * std::pow(2.0, 0.6 * k); }

// [DCT] The decoder sensitivity probe (DCT_MVP_PLAN.md 2.5): for every block and channel, at P = 4 pixel positions
// [DCT] (bx * 8 + {2, 5}, by * 8 + {2, 5}) build the MLP input once (`features(px, py, f)`, the caller's
// [DCT] Decoder::features on the decode buffer) and evaluate the MLP with f[c] at s = -1, -0.5, 0, 0.5, 1; the
// [DCT] segment gain g_j = (255 / 16) sqrt(sum_o cw[o] (out_{j+1}[o] - out_j[o])^2 / wsum) is the cw-weighted
// [DCT] RGB distance per plane unit (the / wsum normalization of a material is a heuristic; XUASTC sums channel
// [DCT] squares), the block gain is the RMS over the 16 (position, segment) pairs, and the code is
// [DCT] dct_code_of_gain. 20 MLP evaluations per block and channel; codes are compared exactly across
// [DCT] the host and the device (they are fitted once, on the host, and copied). Not strict FP: an estimate,
// [DCT] compiled like the rest of the trainer (a CPU and a --cuda run may differ on a code boundary).
// [DCT] With C > 1 channels only f[c] is swept and f[0..C-1] \ {c} stay at their current decode-buffer values
// [DCT] (features() fills every level-0 channel at the texel): channel c's gain is the decoder's sensitivity to
// [DCT] channel c with the other level-0 channels at their current values, a conditional sensitivity; the cost
// [DCT] is C x 20 evaluations per block (C x 0.31 decode-equivalents per probe).
template <class FEAT>
static void dct_probe_codes(const MLP& mlp, bool clamp_out, const std::vector<float>& cw, float wsum, const DctLevel& Q, FEAT features,
                            std::vector<uint8_t>& code, std::vector<float>* gain = nullptr) {
    static const int P[2] = { 2, 5 };
    static const float SJ[5] = { -1.0f, -0.5f, 0.0f, 0.5f, 1.0f };
    const size_t n = Q.nblk() * Q.C;
    code.assign(n, 0);
    if (gain) gain->assign(n, 0.0f);
    const int nout = mlp.nout;
#pragma omp parallel for schedule(dynamic)
    for (int by = 0; by < Q.BH; by++)
        for (int bx = 0; bx < Q.BW; bx++)
            for (int c = 0; c < Q.C; c++) {
                double acc = 0.0;
                for (int py = 0; py < 2; py++)
                    for (int px = 0; px < 2; px++) {
                        float f[MAXH], out[5][MAXOUT];
                        features(bx * 8 + P[px], by * 8 + P[py], f);
                        for (int j = 0; j < 5; j++) { f[c] = SJ[j]; mlp_forward(mlp, mlp.p.data(), f, out[j], clamp_out); }
                        for (int j = 0; j < 4; j++) {
                            double ss = 0.0;
                            for (int o = 0; o < nout; o++) { const double d = (double)out[j + 1][o] - (double)out[j][o]; ss += cw[o] * d * d; }
                            const double g = (255.0 / 16.0) * std::sqrt(ss / wsum);
                            acc += g * g;
                        }
                    }
                const double gb = std::sqrt(acc / 16.0);
                const size_t i = ((size_t)by * Q.BW + bx) * Q.C + c;
                code[i] = (uint8_t)dct_code_of_gain(gb);
                if (gain) (*gain)[i] = (float)gb;
            }
}

// [DCT] The bit simulator and the statistics of DCT_MVP_PLAN.md 4.5 / 12.B. Per channel the symbols are walked
// [DCT] in block raster order: the DC symbol, then the 63 ACs in zigzag order as (run, sign, |q| - 1) tokens with
// [DCT] run = zeros since the previous nonzero, and EOB (run symbol 64) when the remaining ACs are all zero (a
// [DCT] block whose last nonzero sits at zigzag 63 has no EOB, as in JPEG). Three rates:
// [DCT]   raw : fixed length: dc_raw_bits per block, 7 (run) + 8 (|q| - 1) + 1 (sign) per nonzero, 7 per EOB, 4 per scale code;
// [DCT]   h0  : order-0 entropy of the run stream (EOB included), of the |q| stream, of the DC residual against the
// [DCT]         left block's DC (first column: the block above; first block: 0), 1 bit per sign, order-0 entropy
// [DCT]         of the code plane (what XUASTC's adaptive order-0 arithmetic profile achieves);
// [DCT]   ctx : run and |q| conditioned on the zigzag position of the previous nonzero (0 for a block's first
// [DCT]         token), DC and signs as h0, the code plane under the (up, left) context of context_entropy_bits.
// [DCT] Each token is charged its code length under the stream's model, so the per-class split (dc / run / mag /
// [DCT] sign / eob / code) sums exactly to the total, and per-block h0 bits are the block's tokens under the
// [DCT] global order-0 code lengths. Ideal adaptive-coder rates: nothing is coded.
static bool g_ctx_stats = true;   // false during progress prints: the (up, left) context estimate hashes every texel and is only needed for the final block; callers fall back to the order-0 figure
static double context_entropy_bits(const std::vector<int>& idx, int W, int H);   // [DCT] forward declaration (defined with bitrate_stats)
struct DctStats {
    // bits, whole level (all channels), by class: 0 dc, 1 run, 2 mag, 3 sign, 4 eob, 5 code, 6 header
    double raw[7] = { 0 }, h0[7] = { 0 }, ctx[7] = { 0 };
    double raw_total = 0, h0_total = 0, ctx_total = 0;   // sums of the seven classes
    double code_raw = 0, code_h0 = 0, code_ctx = 0, header = 0;
    size_t nblk = 0, ntok_nz = 0, ntok_eob = 0;
    size_t nmag_over = 0;   // magnitudes above 256 (folded into the top bin; must stay 0 for an orthonormal AC of a [0,64] plane)
    // nonzero ACs per (block, channel)
    double nz_mean = 0; int nz_median = 0, nz_max = 0; size_t nzhist[8] = { 0 };   // buckets 0, 1, 2, 3-4, 5-8, 9-16, 17-32, 33-63
    double zero_pct = 0, zero_interior_pct = 0, zero_trailing_pct = 0;
    size_t eob_only = 0; double eob_only_frac = 0;
    // last nonzero zigzag index over blocks with >= 1 nonzero
    double lnz_mean = 0; int lnz_median = 0, lnz_max = 0; size_t lnzhist[7] = { 0 };   // 0, 1-2, 3-5, 6-9, 10-20, 21-35, 36-63
    double firstorder_pct = 0, second_pct = 0;   // blocks with >= 1 nonzero whose last nonzero is at zigzag <= 2 / <= 5
    double run_mean = 0, run0_pct = 0;
    int dc_min = 0, dc_max = 0; double dc_mean = 0, dc_h0_per_blk = 0, dcres_h0_per_blk = 0;
    double pnz[DCT_NN] = { 0 };   // fraction of (block, channel) with a nonzero at each zigzag position (1..63)
    size_t codehist[16] = { 0 }; double code_mean = 0, allones_pct = 0, dconly_pct = 0;
    double blkraw_mean = 0, blkraw_median = 0, blkraw_max = 0, blkh0_mean = 0, blkh0_median = 0, blkh0_max = 0;
    size_t blkbits_hist[8] = { 0 };   // per (block, channel) h0 bits: <8, <16, <32, <64, <128, <256, <512, >= 512
    std::vector<float> blk_h0;        // [block][channel]
    std::vector<uint8_t> blk_nz, blk_lnz;
};
static int dct_bucket8_nz(int nz) { return nz <= 2 ? nz : (nz <= 4 ? 3 : (nz <= 8 ? 4 : (nz <= 16 ? 5 : (nz <= 32 ? 6 : 7)))); }
static int dct_bucket7_lnz(int lnz) { return lnz == 0 ? 0 : (lnz <= 2 ? 1 : (lnz <= 5 ? 2 : (lnz <= 9 ? 3 : (lnz <= 20 ? 4 : (lnz <= 35 ? 5 : 6))))); }
static int dct_bucket8_bits(double b) { int k = 0; double t = 8; while (k < 7 && b >= t) { k++; t *= 2; } return k; }
static double dct_median_of(std::vector<double> v) { if (v.empty()) return 0; std::sort(v.begin(), v.end()); const size_t n = v.size(); return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]); }
// [DCT] only_c = -1: every channel, summed (the level's figures); only_c = c: that channel alone (the dct[c] lines). The
// [DCT] per-channel raw / h0 / ctx totals sum exactly to the summed call's (every count table, the 96-bit header and the
// [DCT] code bits are per channel either way); S.nblk stays the block count and the blk_* vectors keep the full
// [DCT] [block][channel] size (dct_map_image reads them with the summed stats), only the selected channel filled.
static DctStats dct_analyze(const DctLevel& Q, int only_c = -1) {   // [DCT] hook: the only_c filter
    DctStats S;
    const int c0 = only_c < 0 ? 0 : only_c, c1 = only_c < 0 ? Q.C : only_c + 1;   // [DCT]
    const size_t nb = Q.nblk(), nbc = nb * (size_t)(c1 - c0);   // [DCT] hook: was nb * Q.C
    auto bi = [&](size_t b, int c) { return only_c < 0 ? b * Q.C + c : b; };   // [DCT] index into the (block, channel) vectors of this call
    S.nblk = nb;
    S.blk_h0.assign(nb * Q.C, 0.0f); S.blk_nz.assign(nb * Q.C, 0); S.blk_lnz.assign(nb * Q.C, 0);   // [DCT] hook: full size (was nbc)
    std::vector<double> blkraw(nbc, 0.0), blkh0(nbc, 0.0);
    std::vector<int> nzs; nzs.reserve(nbc);
    std::vector<int> lnzs;
    size_t zeros_interior = 0, zeros_trailing = 0, run_sum = 0, run0 = 0;
    double dc_sum = 0; S.dc_min = 1 << 30; S.dc_max = -(1 << 30);
    size_t pnzc[DCT_NN] = { 0 };
    for (int c = c0; c < c1; c++) {   // [DCT] hook: was 0 .. Q.C
        // pass 1: counts
        std::vector<size_t> run_cnt(65, 0), mag_cnt(257, 0), ctxr_cnt(64, 0), ctxm_cnt(64, 0);
        std::vector<size_t> ctx_run((size_t)64 * 65, 0), ctx_mag((size_t)64 * 257, 0);
        std::vector<size_t> dc_cnt(2049, 0), dcres_cnt(4097, 0);
        size_t nsign = 0, nrun = 0, neob = 0;
        std::vector<int> dcs(nb, 0);
        for (size_t b = 0; b < nb; b++) {
            const int16_t* sym = Q.symbols(b, c);
            const int dc = sym[0];
            dcs[b] = dc;
            const int bx = (int)(b % Q.BW), by = (int)(b / Q.BW);
            const int pred = bx > 0 ? dcs[b - 1] : (by > 0 ? dcs[b - Q.BW] : 0);
            dc_cnt[(size_t)std::max(0, std::min(2048, dc + 1024))]++;
            dcres_cnt[(size_t)std::max(0, std::min(4096, dc - pred + 2048))]++;
            dc_sum += dc; S.dc_min = std::min(S.dc_min, dc); S.dc_max = std::max(S.dc_max, dc);
            int run = 0, prev = 0, nz = 0, lnz = 0;
            for (int p = 1; p < DCT_NN; p++) {
                const int q = sym[DCT_ZIGZAG[p]];
                if (q == 0) { run++; continue; }
                const int mag = std::min(256, std::abs(q)); if (std::abs(q) > 256) S.nmag_over++;
                run_cnt[run]++; mag_cnt[mag]++; ctx_run[(size_t)prev * 65 + run]++; ctx_mag[(size_t)prev * 257 + mag]++; ctxr_cnt[prev]++; ctxm_cnt[prev]++;
                run_sum += run; if (run == 0) run0++;
                nsign++; nrun++; nz++; lnz = p; prev = p; pnzc[p]++;
                run = 0;
            }
            if (lnz < DCT_NN - 1) { run_cnt[DCT_EOB]++; ctx_run[(size_t)prev * 65 + DCT_EOB]++; ctxr_cnt[prev]++; neob++; zeros_trailing += (size_t)(DCT_NN - 1 - lnz); }
            zeros_interior += (size_t)(lnz - nz);
            nzs.push_back(nz); if (nz) lnzs.push_back(lnz);
            S.blk_nz[b * Q.C + c] = (uint8_t)nz; S.blk_lnz[b * Q.C + c] = (uint8_t)lnz;
            S.nzhist[dct_bucket8_nz(nz)]++;
            if (nz == 0) S.eob_only++;
            else { S.lnzhist[dct_bucket7_lnz(lnz)]++; if (lnz <= 2) S.firstorder_pct += 1; if (lnz <= 5) S.second_pct += 1; }
        }
        S.ntok_nz += nrun; S.ntok_eob += neob;
        // code lengths (order-0 and context) from the counts
        const size_t nruntok = nrun + neob;
        auto cl0 = [](size_t cnt, size_t tot) { return cnt && tot ? -std::log2((double)cnt / (double)tot) : 0.0; };
        double dc_h0 = 0, dcres_h0 = 0;
        for (size_t k = 0; k < dc_cnt.size(); k++) if (dc_cnt[k]) dc_h0 += dc_cnt[k] * cl0(dc_cnt[k], nb);
        for (size_t k = 0; k < dcres_cnt.size(); k++) if (dcres_cnt[k]) dcres_h0 += dcres_cnt[k] * cl0(dcres_cnt[k], nb);
        S.dc_h0_per_blk += dc_h0; S.dcres_h0_per_blk += dcres_h0;
        // pass 2: charge every token; per class and per block
        for (size_t b = 0; b < nb; b++) {
            const int16_t* sym = Q.symbols(b, c);
            const int bx = (int)(b % Q.BW), by = (int)(b / Q.BW);
            const int pred = bx > 0 ? dcs[b - 1] : (by > 0 ? dcs[b - Q.BW] : 0);
            const double dcbits = cl0(dcres_cnt[(size_t)std::max(0, std::min(4096, sym[0] - pred + 2048))], nb);
            double braw = Q.dc_raw_bits, bh0 = dcbits, bctx = dcbits;
            S.raw[0] += Q.dc_raw_bits; S.h0[0] += dcbits; S.ctx[0] += dcbits;
            int run = 0, prev = 0, lnz = 0;
            for (int p = 1; p < DCT_NN; p++) {
                const int q = sym[DCT_ZIGZAG[p]];
                if (q == 0) { run++; continue; }
                const int mag = std::min(256, std::abs(q)); if (std::abs(q) > 256) S.nmag_over++;
                const double r0 = cl0(run_cnt[run], nruntok), m0 = cl0(mag_cnt[mag], nrun);
                const double rc = cl0(ctx_run[(size_t)prev * 65 + run], ctxr_cnt[prev]), mc = cl0(ctx_mag[(size_t)prev * 257 + mag], ctxm_cnt[prev]);
                S.raw[1] += 7; S.raw[2] += 8; S.raw[3] += 1; braw += 16;
                S.h0[1] += r0; S.h0[2] += m0; S.h0[3] += 1; bh0 += r0 + m0 + 1;
                S.ctx[1] += rc; S.ctx[2] += mc; S.ctx[3] += 1; bctx += rc + mc + 1;
                lnz = p; prev = p; run = 0;
            }
            if (lnz < DCT_NN - 1) {
                const double e0 = cl0(run_cnt[DCT_EOB], nruntok), ec = cl0(ctx_run[(size_t)prev * 65 + DCT_EOB], ctxr_cnt[prev]);
                S.raw[4] += 7; S.h0[4] += e0; S.ctx[4] += ec; braw += 7; bh0 += e0; bctx += ec;
            }
            blkraw[bi(b, c)] = braw; blkh0[bi(b, c)] = bh0; S.blk_h0[b * Q.C + c] = (float)bh0;   // [DCT] hook: bi() (was b * Q.C + c)
        }
        // scale codes: 4 bits raw, order-0 and (up, left) context entropy of the BW x BH code grid
        std::vector<int> cidx(nb); size_t chist[16] = { 0 };
        for (size_t b = 0; b < nb; b++) { const int k = Q.code[b * Q.C + c]; cidx[b] = k; chist[k]++; S.codehist[k]++; S.code_mean += k; }
        double ch0 = 0; for (int k = 0; k < 16; k++) if (chist[k]) ch0 += chist[k] * cl0(chist[k], nb);
        S.code_raw += 4.0 * nb; S.code_h0 += ch0; S.code_ctx += g_ctx_stats ? context_entropy_bits(cidx, Q.BW, Q.BH) : ch0;   // progress prints: order-0 stands in
        S.header += 3 * 32.0;
    }
    S.raw[5] = S.code_raw; S.h0[5] = S.code_h0; S.ctx[5] = S.code_ctx;
    S.raw[6] = S.h0[6] = S.ctx[6] = S.header;
    for (int k = 0; k < 7; k++) { S.raw_total += S.raw[k]; S.h0_total += S.h0[k]; S.ctx_total += S.ctx[k]; }
    // nonzero / last-nonzero / run / DC / code / per-block summaries
    {
        double s = 0; int mx = 0; for (int v : nzs) { s += v; mx = std::max(mx, v); }
        S.nz_mean = nbc ? s / nbc : 0; S.nz_max = mx;
        std::vector<double> t(nzs.begin(), nzs.end()); S.nz_median = (int)dct_median_of(t);
        const double nac = (double)nbc * 63.0;
        S.zero_pct = nac ? 100.0 * (nac - s) / nac : 0;
        S.zero_interior_pct = nac ? 100.0 * zeros_interior / nac : 0;
        S.zero_trailing_pct = nac ? 100.0 * zeros_trailing / nac : 0;
        S.eob_only_frac = nbc ? (double)S.eob_only / nbc : 0;
    }
    {
        double s = 0; int mx = 0; for (int v : lnzs) { s += v; mx = std::max(mx, v); }
        S.lnz_mean = lnzs.empty() ? 0 : s / lnzs.size(); S.lnz_max = mx;
        std::vector<double> t(lnzs.begin(), lnzs.end()); S.lnz_median = (int)dct_median_of(t);
        S.firstorder_pct = lnzs.empty() ? 0 : 100.0 * S.firstorder_pct / lnzs.size();
        S.second_pct = lnzs.empty() ? 0 : 100.0 * S.second_pct / lnzs.size();
    }
    S.run_mean = S.ntok_nz ? (double)run_sum / S.ntok_nz : 0; S.run0_pct = S.ntok_nz ? 100.0 * run0 / S.ntok_nz : 0;
    S.dc_mean = nbc ? dc_sum / nbc : 0; S.dc_h0_per_blk = nbc ? S.dc_h0_per_blk / nbc : 0; S.dcres_h0_per_blk = nbc ? S.dcres_h0_per_blk / nbc : 0;
    if (S.dc_min > S.dc_max) S.dc_min = S.dc_max = 0;
    for (int p = 0; p < DCT_NN; p++) S.pnz[p] = nbc ? (double)pnzc[p] / nbc : 0;
    S.code_mean = nbc ? S.code_mean / nbc : 0;
    {
        size_t allones = 0, dconly = 0;
        for (int c = c0; c < c1; c++)   // [DCT] hook: was 0 .. Q.C
            for (int k = 0; k < 16; k++) {
                const int* st = Q.steps(c, k);
                bool ones = true; int mn = 1 << 30;
                for (int i = 1; i < DCT_NN; i++) { if (st[i] != 1) ones = false; mn = std::min(mn, st[i]); }
                size_t cnt = 0; for (size_t b = 0; b < nb; b++) if (Q.code[b * Q.C + c] == k) cnt++;
                if (ones) allones += cnt;
                if (mn > 64) dconly += cnt;
            }
        S.allones_pct = nbc ? 100.0 * allones / nbc : 0; S.dconly_pct = nbc ? 100.0 * dconly / nbc : 0;
    }
    {
        double sr = 0, sh = 0, mr = 0, mh = 0;
        for (size_t i = 0; i < nbc; i++) { sr += blkraw[i]; sh += blkh0[i]; mr = std::max(mr, blkraw[i]); mh = std::max(mh, blkh0[i]); S.blkbits_hist[dct_bucket8_bits(blkh0[i])]++; }
        S.blkraw_mean = nbc ? sr / nbc : 0; S.blkraw_max = mr; S.blkraw_median = dct_median_of(blkraw);
        S.blkh0_mean = nbc ? sh / nbc : 0; S.blkh0_max = mh; S.blkh0_median = dct_median_of(blkh0);
    }
    return S;
}

// [DCT] The codec's own distortion against the clamped shadow: rms and max of (zq - z) * 32 in plane units, and the
// [DCT] fraction of level-0 values the snap moved (zq != z).
static void dct_plane_stats(const Latent& L0, const float* z, const float* zq, double& rms, double& mx, double& changed_pct, int only_c = -1) {   // [DCT] hook: only_c = one channel's values (the dct[c] lines)
    double s2 = 0; mx = 0; size_t ch = 0, n = 0;   // [DCT] hook: n = values counted (was L0.size())
    for (size_t i = 0; i < L0.size(); i++) {   // [DCT] hook: was a one-line loop
        if (only_c >= 0 && (int)(i % L0.C) != only_c) continue;   // [DCT]
        n++;   // [DCT]
        const double d = ((double)zq[i] - (double)z[i]) * 32.0; s2 += d * d; mx = std::max(mx, std::fabs(d)); if (zq[i] != z[i]) ch++;   // [DCT] unchanged body
    }   // [DCT]
    rms = n ? std::sqrt(s2 / n) : 0; changed_pct = n ? 100.0 * ch / n : 0;   // [DCT] hook: n (was L0.size())
}

// [DCT] "50" or "50,30": the per-channel quality list, the label of the progress line and the done: line.
static std::string dct_spec(const DctLevel& Q) {
    std::string r; for (size_t c = 0; c < Q.q.size(); c++) r += (c ? "," : "") + std::to_string(Q.q[c]); return r;
}
// [DCT] The per-channel mean of the probe's block gains (the decoder's sensitivity to that channel, in the 8-bit output
// [DCT] rms per selector half-step that dct_code_of_gain maps to the code) for the switch and refit lines when C > 1:
// [DCT] ` (mean gain ch a / b)`; empty when C == 1, so those lines are unchanged. gain is [block][channel] as dct_probe_codes fills it.
static std::string dct_gain_clause(const DctLevel& Q, const std::vector<float>& gain) {   // [DCT]
    if (Q.C <= 1 || gain.empty()) return "";   // [DCT]
    std::string s = " (mean gain ch";   // [DCT]
    for (int c = 0; c < Q.C; c++) {   // [DCT]
        double sum = 0.0; size_t n = 0;   // [DCT]
        for (size_t i = (size_t)c; i < gain.size(); i += (size_t)Q.C) { sum += gain[i]; n++; }   // [DCT]
        char buf[48]; snprintf(buf, sizeof buf, "%s %.2f", c ? " /" : "", n ? sum / n : 0.0); s += buf;   // [DCT]
    }   // [DCT]
    return s + ")";   // [DCT]
}   // [DCT]

// [DCT] The `dct codes` line (DCT_MVP_PLAN.md 12.A): code histogram, the effective steps at zigzag 1 and 63 per code
// [DCT] (channel c's tables; the line is labelled `dct codes[c]` when the level has more than one channel and S is
// [DCT] then that channel's dct_analyze), the fraction of blocks whose AC table is all ones and whose smallest AC step exceeds 64.
static void dct_print_codes(const DctLevel& Q, const DctStats& S, int c = 0) {   // [DCT] hook: the channel argument
    if (Q.C > 1) printf("dct codes[%d]", c); else   // [DCT]
    printf("dct codes");
    for (int k = 0; k < 16; k++) printf(" k%d %zu", k, S.codehist[k]);
    printf(" mean_k %.2f | step1 by k", S.code_mean);
    for (int k = 0; k < 16; k++) printf(" %d", Q.steps(c, k)[DCT_ZIGZAG[1]]);   // [DCT] hook: steps(c, k) (was channel 0)
    printf(" | step63 by k");
    for (int k = 0; k < 16; k++) printf(" %d", Q.steps(c, k)[DCT_ZIGZAG[63]]);   // [DCT] hook: steps(c, k)
    printf(" | all-ones %.1f%% dc-only-by-table %.1f%%\n", S.allones_pct, S.dconly_pct);
}
// [DCT] One channel: the summed line as before; C > 1: one `dct codes[c]` line per channel from that channel's statistics.
static void dct_print_codes_all(const DctLevel& Q, const DctStats& S) {   // [DCT]
    if (Q.C == 1) { dct_print_codes(Q, S, 0); return; }   // [DCT]
    for (int c = 0; c < Q.C; c++) dct_print_codes(Q, dct_analyze(Q, c), c);   // [DCT]
}   // [DCT]
// [DCT] --dct-lambda: the proxy bits of the snapped level (sum of Q.bits / 16) and per channel (per_c[MAX_DCT_CH], may be null).
static void dct_proxy_totals(const DctLevel& Q, double& bits, double* per_c) {   // [DCT]
    bits = 0.0; if (per_c) for (int c = 0; c < MAX_DCT_CH; c++) per_c[c] = 0.0;   // [DCT]
    if (Q.bits.size() != Q.nblk() * (size_t)Q.C) return;   // [DCT]
    for (size_t b = 0; b < Q.nblk(); b++)   // [DCT]
        for (int c = 0; c < Q.C; c++) { const double v = Q.bits[b * Q.C + c] / 16.0; bits += v; if (per_c) per_c[c] += v; }   // [DCT]
}   // [DCT]
static const char* dct_rate_mode(const DctLevel& Q) { return Q.rate_es && Q.rate_trunc ? "es+trunc" : (Q.rate_es ? "es-only" : "trunc-only"); }   // [DCT]
// [DCT] The `dct rate` line (DCT_RATE_PLAN.md section 6): the proxy total against the simulator's order-0 AC bits (run + mag +
// [DCT] sign + eob of the h0 line, the tokens the proxy prices; the DC, the codes and the headers are outside the proxy), the
// [DCT] lambda_t16 table, the last snap's truncation count and the last latent step's hit fraction. The second ratio leaves the EOB out on
// [DCT] both sides (proxy - 2 bits per EOB token, the h0 run + mag + sign bits): the proxy's fixed 2-bit EOB against an adaptive coder's
// [DCT] fraction of a bit is what pushes the first ratio above 1 on EOB-dominated streams. A part --dct-rate left out prints `off`.
static void dct_print_rate(const DctLevel& Q, const DctStats& S, double npix) {   // [DCT]
    double proxy; dct_proxy_totals(Q, proxy, nullptr);   // [DCT]
    const double nb = (double)std::max<size_t>(Q.nblk(), 1);   // [DCT]
    const double h0ac = S.h0[1] + S.h0[2] + S.h0[3] + S.h0[4], h0tok = S.h0[1] + S.h0[2] + S.h0[3];   // [DCT]
    const double proxy_tok = proxy - (DCT_C_EOB / 16.0) * (double)S.ntok_eob;   // [DCT] the proxy without its EOB tokens (one per (block, channel) whose last nonzero is below zigzag 63, = S.ntok_eob)
    printf("dct rate lambda %g mode %s proxy total %.0f /blk %.2f = %.4f bpp | h0 ac (run+mag+sign+eob) /blk %.2f ratio proxy/h0 %.3f (eob excluded %.3f)", Q.lambda, dct_rate_mode(Q), proxy, proxy / nb, proxy / npix, h0ac / nb, h0ac > 0 ? proxy / h0ac : 0.0, h0tok > 0 ? proxy_tok / h0tok : 0.0);   // [DCT]
    if (Q.rate_trunc) { printf(" | lambda_t16 by k"); for (int k = 0; k < 16; k++) printf(" %.4g", Q.lam_t16[k]); }   // [DCT]
    else printf(" | lambda_t16 off");   // [DCT]
    if (Q.rate_trunc) printf(" | trunc last-snap %zu of %zu (%.1f%%)", Q.last_truncated, Q.last_nz_before, Q.last_nz_before ? 100.0 * Q.last_truncated / Q.last_nz_before : 0.0);   // [DCT]
    else printf(" | trunc off");   // [DCT]
    if (Q.rate_es) printf(" | hits %.1f%% (of %zu)\n", Q.rate_evals ? 100.0 * Q.rate_hits / Q.rate_evals : 0.0, Q.rate_evals);   // [DCT]
    else printf(" | hits off\n");   // [DCT]
}   // [DCT]
// [DCT] The compact per-channel line of the final block (C > 1 only): `dct[c] q .. nz .. eob0 .. lnz .. codes k0..k15 .. |
// [DCT] bits raw / h0 / ctx (per block position and bpp) | dc range .. hres .. | plane rms .. clamped% ..`. The raw / h0 /
// [DCT] ctx totals of the C lines sum to the summed `dct bits` lines (each channel carries its own 96-bit header and code bits).
static void dct_print_channel(const DctLevel& Q, int c, double npix, const Latent& L0, const float* z, const float* zq) {   // [DCT]
    const DctStats S = dct_analyze(Q, c);   // [DCT]
    const double nb = (double)std::max<size_t>(S.nblk, 1);   // [DCT]
    printf("dct[%d] q %d nz mean %.2f eob0 %.1f%% lnz %.1f codes k0..k15", c, Q.q[c], S.nz_mean, 100.0 * S.eob_only_frac, S.lnz_mean);   // [DCT]
    for (int k = 0; k < 16; k++) printf(" %zu", S.codehist[k]);   // [DCT]
    printf(" mean_k %.2f | bits raw %.0f h0 %.0f ctx %.0f (/blk %.2f/%.2f/%.2f = %.4f/%.4f/%.4f bpp) | dc range %d..%d hres %.2f",   // [DCT]
        S.code_mean, S.raw_total, S.h0_total, S.ctx_total, S.raw_total / nb, S.h0_total / nb, S.ctx_total / nb, S.raw_total / npix, S.h0_total / npix, S.ctx_total / npix, S.dc_min, S.dc_max, S.dcres_h0_per_blk);   // [DCT]
    double rms, mx, chg; dct_plane_stats(L0, z, zq, rms, mx, chg, c);   // [DCT]
    const double nvals = L0.C ? (double)L0.size() / L0.C : 0.0;   // [DCT] values of this channel
    printf(" | plane rms %.3f max %.2f clamped%% %.2f changed%% %.1f", rms, mx, nvals ? 100.0 * Q.last_clamped_c[c] / nvals : 0.0, chg);   // [DCT] hook: no "\n" (the --dct-lambda clause may follow)
    if (Q.lambda > 0.0f) { double pb, pc[MAX_DCT_CH]; dct_proxy_totals(Q, pb, pc); printf(" proxy %.0f (%.2f/blk)", pc[c], pc[c] / nb); if (Q.rate_trunc) printf(" trunc %zu", Q.last_truncated_c[c]); else printf(" trunc off"); }   // [DCT] --dct-lambda: this channel's proxy bits and last-snap truncations
    printf("\n");   // [DCT]
}   // [DCT]
// [DCT] The full statistics block (12.B), every line `dct <keyword> ...`; npix = image pixels for the bpp column.
static void dct_print_stats(const DctLevel& Q, const DctStats& S, double npix, const Latent& L0, const float* z, const float* zq) {
    printf("dct cfg q %s N %d dc_step %d blocks %zu (%dx%d) ch %d codes-fitted-at %d refit-every %d refit-until %d refits %d last-refit-at %d dc_raw_bits %d\n", dct_spec(Q).c_str(), Q.N, Q.dc_step, S.nblk, Q.BW, Q.BH, Q.C, Q.fitted_at,   // [DCT] refit fields
        Q.refit_every, Q.refit_until_it, Q.refits, Q.refitted_at, Q.dc_raw_bits);   // [DCT]
    printf("dct nz mean %.2f median %d max %d zero%% %.1f (interior%% %.1f trailing%% %.1f) mag>256 %zu\n", S.nz_mean, S.nz_median, S.nz_max, S.zero_pct, S.zero_interior_pct, S.zero_trailing_pct, S.nmag_over);
    printf("dct nzhist 0 %zu 1 %zu 2 %zu 3-4 %zu 5-8 %zu 9-16 %zu 17-32 %zu 33-63 %zu\n", S.nzhist[0], S.nzhist[1], S.nzhist[2], S.nzhist[3], S.nzhist[4], S.nzhist[5], S.nzhist[6], S.nzhist[7]);
    printf("dct lnz mean %.1f median %d max %d hist 0 %zu 1-2 %zu 3-5 %zu 6-9 %zu 10-20 %zu 21-35 %zu 36-63 %zu firstorder%% %.1f <=2nd%% %.1f\n",
        S.lnz_mean, S.lnz_median, S.lnz_max, S.lnzhist[0], S.lnzhist[1], S.lnzhist[2], S.lnzhist[3], S.lnzhist[4], S.lnzhist[5], S.lnzhist[6], S.firstorder_pct, S.second_pct);
    printf("dct run mean %.2f run0%% %.1f (runs = nonzero tokens %zu; EOB tokens %zu)\n", S.run_mean, S.run0_pct, S.ntok_nz, S.ntok_eob);
    printf("dct dc range %d..%d mean %.2f h0 %.2f hres %.2f bits/blk\n", S.dc_min, S.dc_max, S.dc_mean, S.dc_h0_per_blk, S.dcres_h0_per_blk);
    printf("dct pnz"); for (int p = 1; p < DCT_NN; p++) printf(" %.3f", S.pnz[p]); printf("\n");
    printf("dct pnz_lo"); for (int p = 1; p <= 16; p++) printf(" %.3f", S.pnz[p]); printf("\n");
    dct_print_codes_all(Q, S);   // [DCT] hook: was dct_print_codes(Q, S) (the same line when C == 1; one line per channel otherwise)
    printf("dct blkbits raw mean %.1f median %.1f max %.0f | h0 mean %.1f median %.1f max %.1f | hist(h0) <8 %zu <16 %zu <32 %zu <64 %zu <128 %zu <256 %zu <512 %zu >=512 %zu\n",
        S.blkraw_mean, S.blkraw_median, S.blkraw_max, S.blkh0_mean, S.blkh0_median, S.blkh0_max,
        S.blkbits_hist[0], S.blkbits_hist[1], S.blkbits_hist[2], S.blkbits_hist[3], S.blkbits_hist[4], S.blkbits_hist[5], S.blkbits_hist[6], S.blkbits_hist[7]);
    const double nb = (double)std::max<size_t>(S.nblk, 1);
    printf("dct bits raw dc %.0f run %.0f mag %.0f sign %.0f eob %.0f code %.0f hdr %.0f total %.0f /blk %.2f = %.4f bpp\n", S.raw[0], S.raw[1], S.raw[2], S.raw[3], S.raw[4], S.raw[5], S.raw[6], S.raw_total, S.raw_total / nb, S.raw_total / npix);
    printf("dct bits h0  dc %.0f run %.0f mag %.0f sign %.0f eob %.0f code %.0f hdr %.0f total %.0f /blk %.2f = %.4f bpp\n", S.h0[0], S.h0[1], S.h0[2], S.h0[3], S.h0[4], S.h0[5], S.h0[6], S.h0_total, S.h0_total / nb, S.h0_total / npix);
    printf("dct bits ctx dc %.0f run %.0f mag %.0f sign %.0f eob %.0f code %.0f hdr %.0f total %.0f /blk %.2f = %.4f bpp\n", S.ctx[0], S.ctx[1], S.ctx[2], S.ctx[3], S.ctx[4], S.ctx[5], S.ctx[6], S.ctx_total, S.ctx_total / nb, S.ctx_total / npix);
    if (Q.lambda > 0.0f) dct_print_rate(Q, S, npix);   // [DCT] --dct-lambda: the dct rate line (before dct plane)
    double rms, mx, chg; dct_plane_stats(L0, z, zq, rms, mx, chg);
    printf("dct plane rms %.3f max %.2f (units of 64) clamped%% %.2f changed%% %.1f\n", rms, mx, L0.size() ? 100.0 * Q.last_clamped / L0.size() : 0.0, chg);
    if (Q.C > 1) for (int c = 0; c < Q.C; c++) dct_print_channel(Q, c, npix, L0, z, zq);   // [DCT] hook: the per-channel view (the lines above stay the summed level)
}

// [DCT] Visualization (12.C): one colour per 8x8 block, W x H pixels so the maps align with recon_*.png; channels side
// [DCT] by side. Kinds: nz (nonzero ACs through the fixed 8-bucket ramp 0/1/2/3-4/5-8/9-16/17-32/33-63 = black / navy /
// [DCT] blue / cyan / green / yellow / orange / white), code (16-step grey, k * 17), bits (per-block h0 bits, log buckets
// [DCT] <8 ... >= 512 on the same ramp), lnz (last nonzero zigzag index, 7 diagonal buckets on the ramp), dconly (white =
// [DCT] no nonzero AC).
static const unsigned char DCT_RAMP[8][3] = { { 0, 0, 0 }, { 0, 0, 128 }, { 0, 64, 255 }, { 0, 220, 255 }, { 0, 200, 0 }, { 255, 230, 0 }, { 255, 128, 0 }, { 255, 255, 255 } };
static bool dct_map_kind_ok(const std::string& k) { return k == "nz" || k == "code" || k == "bits" || k == "lnz" || k == "dconly" || k == "all"; }
static void dct_map_image(const DctLevel& Q, const DctStats& S, const std::string& kind, int W, int H, Image& img) {
    img.w = W * Q.C; img.h = H; img.nc = 3; img.rgb.assign((size_t)img.w * img.h * 3, 0.0f);
    for (int c = 0; c < Q.C; c++)
        for (size_t b = 0; b < Q.nblk(); b++) {
            const size_t i = b * Q.C + c;
            unsigned char rgb[3];
            if (kind == "code") { const unsigned char g = (unsigned char)(Q.code[i] * 17); rgb[0] = rgb[1] = rgb[2] = g; }
            else if (kind == "dconly") { const unsigned char g = S.blk_nz[i] == 0 ? 255 : 0; rgb[0] = rgb[1] = rgb[2] = g; }
            else {
                const int bk = kind == "bits" ? dct_bucket8_bits(S.blk_h0[i]) : (kind == "lnz" ? dct_bucket7_lnz(S.blk_lnz[i]) : dct_bucket8_nz(S.blk_nz[i]));
                memcpy(rgb, DCT_RAMP[bk], 3);
            }
            const int bx = (int)(b % Q.BW), by = (int)(b / Q.BW);
            for (int y = 0; y < 8; y++)
                for (int x = 0; x < 8; x++) {
                    float* d = &img.rgb[((size_t)(by * 8 + y) * img.w + c * W + bx * 8 + x) * 3];
                    for (int k = 0; k < 3; k++) d[k] = rgb[k] / 255.0f;
                }
        }
}
// dct_resid_*.png: (shadow - snapped) * 2 + 0.5 per channel side by side (gray 128 = 0): ringing and what the dead zone removed.
static void dct_save_resid_png(const std::string& path, const Latent& L0, const float* z, const float* zq) {
    const int w = L0.W * L0.C, h = L0.H;
    std::vector<unsigned char> buf((size_t)w * h);
    for (int y = 0; y < h; y++)
        for (int c = 0; c < L0.C; c++)
            for (int x = 0; x < L0.W; x++) {
                const size_t i = ((size_t)y * L0.W + x) * L0.C + c;
                const float m = std::min(1.0f, std::max(0.0f, (z[i] - zq[i]) * 2.0f + 0.5f));   // [DCT] gain 2: +-0.5 of the range spans the image (x8 saturated below q ~50)
                buf[(size_t)y * w + c * L0.W + x] = (unsigned char)std::lround(m * 255.0f);
            }
    stbi_write_png(path.c_str(), w, h, 1, buf.data(), w);
}

// [DCT] --dct-selftest (DCT_MVP_PLAN.md 8.4, amended by 13.5 / 13.6 / 13.11). Returns 0 when every item passes.
static int dct_selftest() {
    dct_init_tables();
    int fails = 0;
    auto report = [&](const char* name, bool ok, const char* detail) { printf("  %-28s %s%s%s\n", name, ok ? "PASS" : "FAIL", detail && *detail ? "  " : "", detail ? detail : ""); fails += !ok; };
    char d[256];
    // (0) the basis literals against double-precision sqrt * cos rounded to float (sanity, not the source of truth)
    {
        int bad = 0;
        for (int u = 0; u < 8; u++)
            for (int x = 0; x < 8; x++) {
                const double a = u == 0 ? std::sqrt(1.0 / 8.0) : std::sqrt(2.0 / 8.0);
                const float ref = (float)(a * std::cos(3.14159265358979323846 * (2 * x + 1) * u / 16.0));
                if (ref != DCT_BASIS[u][x]) bad++;
            }
        snprintf(d, sizeof(d), "%d of 64 literals differ from the double product rounded to float", bad);
        report("basis literals", bad == 0, d);
        int badk = 0;
        for (int k = 0; k < 16; k++) if ((float)(1.0 / dct_gain_of_code(k)) != DCT_AK[k]) badk++;
        snprintf(d, sizeof(d), "%d of 16 A_k literals differ from 1 / g_k rounded to float", badk);
        report("A_k literals", badk == 0, d);
    }
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> U(0.0f, 64.0f);
    // (a) orthonormality of the folded table: inv(fwd(w)) == w to 1e-4 on 1000 random [0,64] blocks
    {
        double mx = 0;
        for (int t = 0; t < 1000; t++) {
            float w[64], C[64], r[64];
            for (int i = 0; i < 64; i++) w[i] = U(rng);
            dct_fwd8(w, C); dct_inv8(C, r);
            for (int i = 0; i < 64; i++) mx = std::max(mx, (double)std::fabs(r[i] - w[i]));
        }
        snprintf(d, sizeof(d), "max |inv(fwd(w)) - w| = %.3e over 1000 blocks", mx);
        report("round trip", mx < 1e-4, d);
    }
    // (b) q = 100, dc_step = 1: every step 1; coefficient error <= 1 for non-exempt ACs, <= 0.5 for the DC and the
    //     first-order pair; the spatial round trip through quantization stays within rms 1 plane unit
    {
        int st[64]; dct_build_steps_scaled(100, 1.0f, 8, 1, st);
        bool ones = true; for (int i = 1; i < 64; i++) if (st[i] != 1) ones = false;
        double mx_ac = 0, mx_dc = 0, rms = 0; size_t n = 0;
        for (int t = 0; t < 1000; t++) {
            float w[64], C[64], D[64], r[64];
            for (int i = 0; i < 64; i++) w[i] = U(rng);
            dct_fwd8(w, C);
            for (int k = 0; k < 64; k++) {
                const int u = k / 8, v = k % 8;
                const int q = k == 0 ? dct_quant_dc(C[0], 1) : dct_quant_ac(C[k], st[k], u, v, true);   // [DCT] hook: true (the dead-zone quantizer, as before)
                D[k] = k == 0 ? dct_dequant_dc(q, 1) : dct_dequant_ac(q, st[k], u, v, true);   // [DCT] hook: true
                const double e = std::fabs((double)D[k] - (double)C[k]);
                if (k == 0 || dct_first_order(u, v)) mx_dc = std::max(mx_dc, e); else mx_ac = std::max(mx_ac, e);
            }
            dct_inv8(D, r);
            for (int i = 0; i < 64; i++) { const double e = (double)r[i] - (double)w[i]; rms += e * e; n++; }
        }
        rms = std::sqrt(rms / n);
        snprintf(d, sizeof(d), "steps all 1: %s; max |err| DC/first-order %.4f (<= 0.5), other ACs %.4f (<= 1), spatial rms %.4f (< 1)", ones ? "yes" : "no", mx_dc, mx_ac, rms);
        report("q 100 quantizer", ones && mx_dc <= 0.5 + 1e-5 && mx_ac <= 1.0 + 1e-5 && rms < 1.0, d);
    }
    // (c) the zigzag against the JPEG 8x8 literal
    {
        static const int ref[64] = { 0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63 };
        int bad = 0; for (int i = 0; i < 64; i++) if (DCT_ZIGZAG[i] != ref[i]) bad++;
        snprintf(d, sizeof(d), "%d of 64 entries differ from the JPEG literal", bad);
        report("zigzag", bad == 0, d);
    }
    // (d) q = 50 at level_scale 1 reproduces Table K.1 in every AC entry with [u][v] = K.1[u][v] (not its transpose); [0] = dc_step
    {
        int st[64]; dct_build_steps_scaled(50, 1.0f, 8, 4, st);
        int bad = 0, badT = 0;
        for (int u = 0; u < 8; u++) for (int v = 0; v < 8; v++) { if (u == 0 && v == 0) continue; if (st[u * 8 + v] != DCT_JPEG_Y[u][v]) bad++; if (st[u * 8 + v] != DCT_JPEG_Y[v][u]) badT++; }
        const bool S50 = dct_quality_scale(50) == 1.0f;
        snprintf(d, sizeof(d), "S(50) = %g; %d AC entries differ from K.1[u][v] (%d from the transpose, which must be nonzero); [0] = %d (dc_step 4)", dct_quality_scale(50), bad, badT, st[0]);
        report("table K.1 at q 50", S50 && bad == 0 && badT > 0 && st[0] == 4, d);
    }
    // (e) k(g_k) == k for k = 0..15, k monotone in g
    {
        int bad = 0; bool mono = true; int last = -1;
        for (int k = 0; k < 16; k++) if (dct_code_of_gain(dct_gain_of_code(k)) != k) bad++;
        for (double g = 0.01; g < 200.0; g *= 1.02) { const int k = dct_code_of_gain(g); if (k < last) mono = false; last = k; }
        snprintf(d, sizeof(d), "%d of 16 codes do not round-trip; monotone %s; g_0 %.4f g_15 %.1f", bad, mono ? "yes" : "no", dct_gain_of_code(0), dct_gain_of_code(15));
        report("scale codes", bad == 0 && mono, d);
    }
    // (f) the first-order exemption and the dead zone
    {
        const int L = 10;
        const int a = dct_quant_ac(0.4f * L, L, 1, 0, true), b = dct_quant_ac(0.6f * L, L, 1, 0, true), c = dct_quant_ac(0.6f * L, L, 2, 0, true), e = dct_quant_ac(1.1f * L, L, 2, 0, true);   // [DCT] hook: true
        const int z = dct_quant_ac(0.99f * L, L, 2, 0, true), one = dct_quant_ac(1.0f * L, L, 2, 0, true), neg = dct_quant_ac(-1.5f * L, L, 0, 2, true);   // [DCT] hook: true
        const float dq = dct_dequant_ac(1, L, 2, 0, true), dq1 = dct_dequant_ac(3, L, 1, 0, true), dqn = dct_dequant_ac(-2, L, 2, 3, true);   // [DCT] hook: true
        const bool ok = a == 0 && b == 1 && c == 0 && e == 1 && z == 0 && one == 1 && neg == -1 && dq == 1.5f * L && dq1 == 3.0f * L && dqn == -2.5f * L;
        snprintf(d, sizeof(d), "(1,0): 0.4L->%d 0.6L->%d; (2,0): 0.6L->%d 0.99L->%d 1.0L->%d 1.1L->%d; (0,2): -1.5L->%d; dequant 1@(2,0) = %.1fL, 3@(1,0) = %.1fL, -2@(2,3) = %.1fL", a, b, c, z, one, e, neg, dq / L, dq1 / L, dqn / L);
        report("dead zone", ok, d);
    }
    // [DCT] (f2) --dct-deadzone 0: plain rounding on every AC: 0.4L -> 0, 0.6L -> 1, 1.49L -> 1, 1.5L -> 2 at (2,0) (half away from zero,
    // [DCT]      symmetric: -0.6L -> -1, -1.5L -> -2), dequant q * L (no 1.5-step bin centre), +-1024 clamp (2000L -> 1024, -2000L -> -1024)
    {   // [DCT]
        const int L = 10;   // [DCT]
        const int a = dct_quant_ac(0.4f * L, L, 2, 0, false), b = dct_quant_ac(0.6f * L, L, 2, 0, false), c = dct_quant_ac(1.49f * L, L, 2, 0, false), e = dct_quant_ac(1.5f * L, L, 2, 0, false);   // [DCT]
        const int na = dct_quant_ac(-0.6f * L, L, 2, 0, false), nb = dct_quant_ac(-1.5f * L, L, 0, 2, false), hi = dct_quant_ac(2000.0f * L, L, 2, 3, false), lo = dct_quant_ac(-2000.0f * L, L, 2, 3, false);   // [DCT]
        const float dq = dct_dequant_ac(1, L, 2, 0, false), dq1 = dct_dequant_ac(3, L, 1, 0, false), dqn = dct_dequant_ac(-2, L, 2, 3, false);   // [DCT]
        bool sym_ok = true; for (int k = 1; k < 64; k++) for (int t = -30; t <= 30; t++) { const float x = 0.1f * (float)t * (float)L; if (dct_quant_ac(x, L, k / 8, k % 8, false) != -dct_quant_ac(-x, L, k / 8, k % 8, false)) sym_ok = false; }   // [DCT]
        const bool ok = a == 0 && b == 1 && c == 1 && e == 2 && na == -1 && nb == -2 && hi == 1024 && lo == -1024 && dq == 1.0f * L && dq1 == 3.0f * L && dqn == -2.0f * L && sym_ok;   // [DCT]
        snprintf(d, sizeof(d), "(2,0): 0.4L->%d 0.6L->%d 1.49L->%d 1.5L->%d -0.6L->%d; (0,2): -1.5L->%d; clamp 2000L->%d -2000L->%d; odd symmetry %s; dequant 1@(2,0) = %.1fL, 3@(1,0) = %.1fL, -2@(2,3) = %.1fL", a, b, c, e, na, nb, hi, lo, sym_ok ? "yes" : "no", dq / L, dq1 / L, dqn / L);   // [DCT]
        report("plain rounding (dz 0)", ok, d);   // [DCT]
    }   // [DCT]
    // (g) the DC symbol is round-half-even at its step and the step tables at the ends of the q range are sane
    {
        const int r0 = dct_quant_dc(2.0f, 4), r1 = dct_quant_dc(6.0f, 4), r2 = dct_quant_dc(10.0f, 4);
        std::vector<int> st(16 * 64); dct_build_steps(1, 8, 4, st.data());
        int mn = 1 << 30, mx = 0; for (int k = 0; k < 16; k++) for (int i = 1; i < 64; i++) { mn = std::min(mn, st[k * 64 + i]); mx = std::max(mx, st[k * 64 + i]); }
        std::vector<int> st90(16 * 64); dct_build_steps(90, 8, 4, st90.data());
        snprintf(d, sizeof(d), "DC 2/4 -> %d, 6/4 -> %d, 10/4 -> %d (half-even 0, 2, 2); q 1 steps %d..%d, q 90 k 7 step(0,1) = %d", r0, r1, r2, mn, mx, st90[7 * 64 + 1]);
        report("DC rounding / step range", r0 == 0 && r1 == 2 && r2 == 2 && mn >= 1 && mx > 1000, d);
    }
    // [DCT] (h) --dct-lambda, the token cost model: c_run(0..7) = c_mag(1..8) = 1,3,3,5,5,5,5,7 bits, monotone to 62 / 256, sign 1, EOB 2
    {   // [DCT]
        static const int ref[8] = { 1, 3, 3, 5, 5, 5, 5, 7 };   // [DCT]
        int bad = 0; bool mono = true;   // [DCT]
        for (int i = 0; i < 8; i++) { if (dct_c_run(i) != 16 * ref[i]) bad++; if (dct_c_mag(i + 1) != 16 * ref[i]) bad++; }   // [DCT]
        for (int r = 1; r <= 62; r++) if (dct_c_run(r) < dct_c_run(r - 1)) mono = false;   // [DCT]
        for (int m = 2; m <= 256; m++) if (dct_c_mag(m) < dct_c_mag(m - 1)) mono = false;   // [DCT]
        snprintf(d, sizeof(d), "%d of 16 table values differ from 1,3,3,5,5,5,5,7 bits; monotone %s; c_run(62) = %d c_mag(256) = %d bits; sign %d EOB %d bits", bad, mono ? "yes" : "no", dct_c_run(62) / 16, dct_c_mag(256) / 16, DCT_C_SIGN / 16, DCT_C_EOB / 16);   // [DCT]
        report("rate cost table", bad == 0 && mono && dct_c_run(62) == 176 && dct_c_mag(256) == 272, d);   // [DCT]
    }   // [DCT]
    // [DCT] (i) the proxy bits of a hand-built block: tokens (0,-3) (0,1) (2,7) (14,-1) + EOB = 5 + 3 + 9 + 9 + 2 = 28 bits; a lone |q| = 1
    // [DCT]     at zigzag 63 = c_run(62) + c_mag(1) + sign = 11 + 1 + 1 = 13 bits (no EOB); an empty block = 2 bits (EOB only)
    {   // [DCT]
        int16_t sym[64] = { 0 };   // [DCT]
        sym[DCT_ZIGZAG[1]] = -3; sym[DCT_ZIGZAG[2]] = 1; sym[DCT_ZIGZAG[5]] = 7; sym[DCT_ZIGZAG[20]] = -1;   // [DCT]
        const int b1 = dct_block_bits(sym, 16);   // [DCT]
        int16_t lone[64] = { 0 }; lone[DCT_ZIGZAG[63]] = 1; const int b2 = dct_block_bits(lone, 16);   // [DCT]
        int16_t empty[64] = { 0 }; const int b3 = dct_block_bits(empty, 16);   // [DCT]
        snprintf(d, sizeof(d), "hand-built block %d/16 = %.2f bits (hand count 28); lone |q| = 1 at zigzag 63: %.2f bits (13, no EOB); empty block %.2f bits (2)", b1, b1 / 16.0, b2 / 16.0, b3 / 16.0);   // [DCT]
        report("rate proxy bits", b1 == 28 * 16 && b2 == 13 * 16 && b3 == 2 * 16, d);   // [DCT]
    }   // [DCT]
    // [DCT] (j) truncation: (a) a lone |q| = 1 at zigzag 63 (L 10, c 12: v 15, dD 135) is dropped iff dD < lambda_t (c_run(62) + c_mag(1) + 16 -
    // [DCT]     c_eob) = 176 lambda_t (kept at 0.7, dropped at 0.8); (b) an interior coefficient's saving includes the run merge: p 3 between
    // [DCT]     p 1 and p 10 saves 80 + c_run(6) - c_run(8) = 48 (not its 80-bit token), so lambda_t 2 keeps it (135 >= 96) and 3 drops it
    // [DCT]     (135 < 144) while p 1 and p 10 survive; (c) lambda 0 truncates nothing; (d) a first-order coefficient at c = 0.5 L
    // [DCT]     (q 1, v L) has dD 0 exactly and is dropped by any lambda > 0
    {   // [DCT]
        int st[64]; for (int k = 0; k < 64; k++) st[k] = 10; st[0] = 4;   // [DCT]
        auto build = [&](float* Cf, int16_t* sym) { for (int k = 0; k < 64; k++) { Cf[k] = 0.0f; sym[k] = 0; } };   // [DCT]
        float Cf[64]; int16_t sym[64]; int nt = 0, nz = 0;   // [DCT]
        // (a)   [DCT]
        build(Cf, sym); Cf[DCT_ZIGZAG[63]] = 12.0f;   // [DCT]
        dct_quantize_block(Cf, st, 4, 0.7f, 16, true, false, sym, &nt, &nz); const int a_keep = sym[DCT_ZIGZAG[63]], a_nt0 = nt, a_nz = nz;   // [DCT] hook: dz true (the dead-zone quantizer; the truncation items are unchanged)
        dct_quantize_block(Cf, st, 4, 0.8f, 16, true, false, sym, &nt, &nz); const int a_drop = sym[DCT_ZIGZAG[63]], a_nt1 = nt;   // [DCT] hook: dz true (the dead-zone quantizer; the truncation items are unchanged)
        // (b)   [DCT]
        build(Cf, sym); Cf[DCT_ZIGZAG[1]] = 25.0f; Cf[DCT_ZIGZAG[3]] = 12.0f; Cf[DCT_ZIGZAG[10]] = 30.0f;   // [DCT] zigzag 1 = (0,1), first-order: q 3 v 30 dD 600; zigzag 3 = (2,0): q 1 v 15 dD 135; zigzag 10 = (4,0): q 3 v 35 dD 875
        dct_quantize_block(Cf, st, 4, 2.0f, 16, true, false, sym, &nt, &nz); const int b_k2 = sym[DCT_ZIGZAG[3]], b_nt2 = nt, b_nz = nz, b_1a = sym[DCT_ZIGZAG[1]], b_10a = sym[DCT_ZIGZAG[10]];   // [DCT] hook: dz true (the dead-zone quantizer; the truncation items are unchanged)
        dct_quantize_block(Cf, st, 4, 3.0f, 16, true, false, sym, &nt, &nz); const int b_k3 = sym[DCT_ZIGZAG[3]], b_nt3 = nt, b_1b = sym[DCT_ZIGZAG[1]], b_10b = sym[DCT_ZIGZAG[10]];   // [DCT] hook: dz true (the dead-zone quantizer; the truncation items are unchanged)
        // (c)   [DCT]
        dct_quantize_block(Cf, st, 4, 0.0f, 16, true, false, sym, &nt, &nz); const int c_nt = nt, c_nz = nz, c_k = sym[DCT_ZIGZAG[3]];   // [DCT] hook: dz true (the dead-zone quantizer; the truncation items are unchanged)
        // (d)   [DCT]
        build(Cf, sym); Cf[DCT_ZIGZAG[1]] = 5.0f;   // [DCT]
        dct_quantize_block(Cf, st, 4, 1e-6f, 16, true, false, sym, &nt, &nz); const int d_q = sym[DCT_ZIGZAG[1]], d_nt = nt, d_nz = nz;   // [DCT] hook: dz true (the dead-zone quantizer; the truncation items are unchanged)
        const bool ok = a_keep == 1 && a_nt0 == 0 && a_nz == 1 && a_drop == 0 && a_nt1 == 1   // [DCT]
            && b_k2 == 1 && b_nt2 == 0 && b_nz == 3 && b_1a == 3 && b_10a == 3 && b_k3 == 0 && b_nt3 == 1 && b_1b == 3 && b_10b == 3   // [DCT]
            && c_nt == 0 && c_nz == 0 && c_k == 1 && d_q == 0 && d_nt == 1 && d_nz == 1;   // [DCT]
        snprintf(d, sizeof(d), "(a) lone 63 at lambda_t 0.7: q %d trunc %d (of %d), at 0.8: q %d trunc %d; (b) interior p3 at 2: q %d trunc %d (of %d; p1 %d p10 %d), at 3: q %d trunc %d (p1 %d p10 %d); (c) lambda 0: trunc %d nz_before %d q %d; (d) first-order c = 0.5L: q %d trunc %d (of %d)",   // [DCT]
            a_keep, a_nt0, a_nz, a_drop, a_nt1, b_k2, b_nt2, b_nz, b_1a, b_10a, b_k3, b_nt3, b_1b, b_10b, c_nt, c_nz, c_k, d_q, d_nt, d_nz);   // [DCT]
        report("rate truncation", ok, d);   // [DCT]
    }   // [DCT]
    printf("dct-selftest: %s\n", fails ? "FAILED" : "all passed");
    return fails ? 1 : 0;
}
// ---------------------------------------------------------------- end of the DCT region [DCT]

// ---------------------------------------------------------------- positional encoding
// phi(u,v): a list of named feature generators, composed from --pos "a,b,...".
// To add one: add an enum value, a name in POS_NAMES, its output count in
// PosFeature::count(), and its evaluation in PosEnc::encode(). Fifteen experimental
// kinds (Fourier, cell DCT, block DCT, one-hot, BC7 partitions, ...) were removed in
// v0.8; tag pre-v0.8-removal has the code, README "Archived experiments" the numbers.
//
//   uv           global u,v mapped to [-1,1]                               (2)
//   lv1local     cell offset of the SECOND latent level, fx,fy -> [-1,1]         (2)
enum PosKind { POS_UV, POS_LV1LOCAL, POS_COUNT };   // same order as PosKindDev in cuda/ntc_cuda.h and PosKind in ntc_decode.cpp
static const char* POS_NAMES[POS_COUNT] = { "uv", "lv1local" };

struct PosFeature {
    PosKind kind;
    int count() const { return 2; }   // both kinds are an (x, y) pair
};

struct PosEnc {
    std::vector<PosFeature> feats;
    std::string spec;   // canonical text form, stored in the model file

    int count() const { int c = 0; for (auto& f : feats) c += f.count(); return c; }
    bool uses_level1() const { for (auto& f : feats) if (f.kind == POS_LV1LOCAL) return true; return false; }

    // Parse "uv,lv1local". "none" or "" gives no positional input.
    bool parse(const std::string& s) {
        feats.clear(); spec.clear();
        if (s == "none" || s.empty()) { spec = "none"; return true; }
        for (size_t a = 0; a < s.size();) {
            size_t e = s.find(',', a);
            if (e == std::string::npos) e = s.size();
            std::string item = s.substr(a, e - a);
            a = e + 1;
            if (item.empty()) continue;
            PosFeature f;
            const std::string& name = item;   // a removed kind (or a ":N" suffix) fails here: no name matches
            int k = 0;
            while (k < POS_COUNT && name != POS_NAMES[k]) k++;
            if (k == POS_COUNT) return false;
            f.kind = (PosKind)k;
            feats.push_back(f);
            if (!spec.empty()) spec += ",";
            spec += name;
        }
        if (spec.empty()) spec = "none";
        return true;
    }

    // u,v: global pixel-center UV in [0,1]. t: level 0's tap for that UV; t1: level 1's
    // tap (null without a second level; only lv1local reads it).
    inline int encode(float u, float v, const BilinearTap& t, const BilinearTap* t1, float* f) const {
        int k = 0;
        for (const PosFeature& pf : feats) {
            switch (pf.kind) {
            case POS_UV:
                f[k++] = u * 2.0f - 1.0f;
                f[k++] = v * 2.0f - 1.0f;
                break;
            case POS_LV1LOCAL:
                f[k++] = t1->fx * 2.0f - 1.0f;
                f[k++] = t1->fy * 2.0f - 1.0f;
                break;
            }
        }
        return k;
    }
};

// ---------------------------------------------------------------- decoder
struct Decoder {
    const Options* opt;
    LatentSet lat;
    MLP mlp;
    PosEnc pos;
    int W = 0, H = 0; // output image size
    // Material loss: per-output-channel weight cw[c] = weights[c / 3]; the loss
    // divisor is 3 * wsum * pixels. Dividing by the weight sum makes --weights
    // purely relative (2,2 == 1,1) and leaves one texture of weight 1 unchanged.
    std::vector<float> cw;
    float wsum = 1.0f;

    int qat_bits = 0;     // --qat: largest level-0 channel bit depth (0 = off), stored in the model file
    std::vector<int> qat_ch;   // bits per level-0 channel (size C0 when on)
    // --qes state (see the QES block above). zq is the snapped decode copy of the whole latent
    // (non-qes levels copied as they are); it exists once the ranges have been fitted (qes_live).
    std::vector<QesLevel> qes;   // one per level; bits 0 = continuous
    std::vector<float> zq;
    bool qes_on = false;         // some level has --qes bits
    bool qes_live = false;       // ranges fitted: every decode reads zq
    bool qes_frozen = false;     // ranges frozen (always, once live): the shadow is clamped to them on every refresh
    DctLevel dct;                // [DCT] --dct-q state (see the DCT region); dct.on / dct.live are false with the flag off
    // The latent every decode reads: the snapped copy once --qes is live, otherwise the shadow itself.
    const float* zdec() const { return (qes_live || dct.live) ? zq.data() : lat.z.data(); }   // [DCT] hook: || dct.live
    float* zdec_mut() { return (qes_live || dct.live) ? zq.data() : lat.z.data(); }           // [DCT] hook: || dct.live
    // Rebuild zq from the shadow (and clamp the shadow when frozen). Returns the number of snapped values that changed.
    size_t qes_refresh() {
        if (dct.on && !dct.live) dct_clamp_level0(lat.lv[0], lat.z.data());   // [DCT] hook: the pre-start clamp of the shadow's level 0 to [-1,1], before either branch (k_dct_clamp0 on the device)
        if (!qes_live && !dct.live) return 0;   // [DCT] hook: was `if (!qes_live) return 0;`
        size_t changed = 0;
        for (size_t l = 0; l < lat.lv.size(); l++) {
            const Latent& L = lat.lv[l];
            if (qes[l].fitted()) changed += qes_snap_level(qes[l], L, lat.z.data() + L.off, zq.data() + L.off, qes_frozen);
            else memcpy(zq.data() + L.off, lat.z.data() + L.off, L.size() * sizeof(float));
        }
        if (dct.live) changed += dct_snap_level0(dct, lat.lv[0], lat.z.data(), zq.data());   // [DCT] hook: level 0 was copied above, now overwritten by the DCT snap (shadow clamped in place)
        return changed;
    }
    // Fit the ranges of every --qes level from the shadow, freeze them, then snap.
    void qes_fit_all() {
        for (size_t l = 0; l < lat.lv.size(); l++) if (qes[l].bits > 0) qes_fit(qes[l], lat.lv[l], lat.z.data() + lat.lv[l].off);
        zq.resize(lat.size());
        qes_live = true; qes_frozen = true;
        qes_refresh();
    }
    // [DCT] Fit the scale codes from the decoder sensitivity probe (on the decode buffer, so a live --qes block latent is
    // [DCT] seen snapped), freeze them, and snap level 0 through the DCT from here on. `gain` optionally receives the block gains.
    void dct_probe(std::vector<uint8_t>& code_out, std::vector<float>* gain = nullptr) const {   // [DCT]
        const float* z = zdec();   // [DCT]
        dct_probe_codes(mlp, opt->clamp_out, cw, wsum, dct, [&](int px, int py, float* f) { features(z, px, py, f); }, code_out, gain);   // [DCT]
    }   // [DCT]
    void dct_fit_all(int it, std::vector<float>* gain = nullptr) {   // [DCT] hook: gain (optional) receives the probe's block gains
        dct_probe(dct.code, gain);   // [DCT]
        dct.fitted_at = it;   // [DCT]
        zq.resize(lat.size());   // [DCT]
        dct.live = true;   // [DCT]
        qes_refresh();   // [DCT]
    }   // [DCT]
    // [DCT] Refit (--dct-refit): re-run the probe on the current decoder, replace the codes and re-snap level 0 from the
    // [DCT] shadow (which is kept, so each block's rounding re-rolls by at most half a step). m1 / m2: codes that moved by >= 1 / >= 2.
    void dct_refit(int it, size_t& m1, size_t& m2, size_t* m1c = nullptr, std::vector<float>* gain = nullptr) {   // [DCT] hook: m1c[MAX_DCT_CH] (optional) = the >= 1 movers per channel (i % C); gain (optional) = the probe's block gains
        std::vector<uint8_t> nc; dct_probe(nc, gain);   // [DCT]
        m1 = m2 = 0;   // [DCT]
        if (m1c) for (int c = 0; c < MAX_DCT_CH; c++) m1c[c] = 0;   // [DCT]
        for (size_t i = 0; i < nc.size(); i++) { const int d = std::abs((int)nc[i] - (int)dct.code[i]); if (d >= 1) { m1++; if (m1c) m1c[i % dct.C]++; } if (d >= 2) m2++; }   // [DCT] hook: the per-channel count
        dct.code = nc;   // [DCT]
        dct.refits++; dct.refitted_at = it;   // [DCT]
        qes_refresh();   // [DCT]
    }   // [DCT]
    int nin() const { return lat.channels() + pos.count(); }

    // Build the MLP input for pixel (px,py) from a given flat latent array:
    // level 0's sample, then each further level's sample at the same UV, then
    // the positional features (cell-relative kinds use level 0's tap, the lv1*
    // kinds level 1's).
    inline void features(const float* z, int px, int py, float* f) const {
        float u = (px + 0.5f) / W, v = (py + 0.5f) / H;
        const Latent& L0 = lat.lv[0];
        BilinearTap t = bilinear_tap(L0, u, v);
        sample_latent(L0, z, t, f);
        int k = L0.C;
        BilinearTap t1;
        for (size_t l = 1; l < lat.lv.size(); l++) {
            const Latent& L = lat.lv[l];
            BilinearTap tl = bilinear_tap(L, u, v);
            if (l == 1) t1 = tl;
            sample_latent(L, z + L.off, tl, f + k);
            k += L.C;
        }
        pos.encode(u, v, t, lat.lv.size() > 1 ? &t1 : nullptr, f + k);
    }

    // One pixel: all mlp.nout outputs (3 per texture) into out[].
    inline void pixel(const float* p, const float* z, int px, int py, float* out) const {
        float f[MAXH];
        features(z, px, py, f);
        mlp_forward(mlp, p, f, out, opt->clamp_out);
    }

    // Full-image decode into img (parallel over rows).
    void decode_full(const float* p, const float* z, Image& img) const {
        img.w = W; img.h = H; img.nc = mlp.nout; img.rgb.resize((size_t)W * H * img.nc);
#pragma omp parallel for schedule(static)
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
                pixel(p, z, x, y, &img.rgb[((size_t)y * W + x) * img.nc]);
    }

    // Per-pixel weighted squared error sum_c cw[c] (out_c - target_c)^2 over all
    // 3T channels (unnormalized; LatentTrainer divides by 3 * wsum * W * H).
    void decode_err(const float* p, const float* z, const Image& target, std::vector<float>& err) const {
        const int nc = mlp.nout;
        err.resize((size_t)W * H);
#pragma omp parallel for schedule(static)
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++) {
                float out[MAXOUT];
                pixel(p, z, x, y, out);
                const float* t = &target.rgb[((size_t)y * W + x) * nc];
                float e = 0;
                for (int c = 0; c < nc; c++) { float d = out[c] - t[c]; e += cw[c] * (d * d); }
                err[(size_t)y * W + x] = e;
            }
    }

    // Weighted mean squared error on a pixel subset: sum_c cw[c] d^2 / (3 * wsum * |idx|).
    // Serial; called per-thread by the MLP trainers.
    double loss_subset(const float* p, const float* z, const Image& target, const std::vector<int>& idx) const {
        const int nc = mlp.nout;
        double s = 0;
        for (int i : idx) {
            int px = i % W, py = i / W;
            float out[MAXOUT];
            pixel(p, z, px, py, out);
            const float* t = &target.rgb[(size_t)i * nc];
            for (int c = 0; c < nc; c++) { float d = out[c] - t[c]; s += cw[c] * (d * d); }
        }
        return s / (3.0 * wsum * idx.size());
    }
};

// Reporting-only error measures: the weighted training objective and the
// unweighted per-texture MSE (comparable to single-texture runs).
struct Mse {
    double weighted = 0;        // sum_t w_t SSE_t / (3 wsum W H)
    double tex[MAXT] = { 0 };   // SSE_t / (3 W H)
};
// Reported error, defined to match Basis Universal's image_metrics::calc (RGB, avg_comp_error):
// both images rounded to 8-bit integers exactly as save_png does, |a - b| per channel, MSE over
// all pixels and all 3 channels of the evaluation extent ew x eh (the source image before
// --block padding; 0 = the whole image), in LSB^2. The training loss itself is the fp32 MSE
// inside the ES / FD steps; this function is only for the printed statistics.
static Mse mse_of(const Image& a, const Image& b, const std::vector<float>& w, float wsum, int ew = 0, int eh = 0) {
    const int T = a.nc / 3;
    if (ew <= 0 || ew > a.w) ew = a.w;
    if (eh <= 0 || eh > a.h) eh = a.h;
    double sse[MAXT] = { 0 };
    for (int yy = 0; yy < eh; yy++)
        for (int xx = 0; xx < ew; xx++) {
            const size_t p = (size_t)yy * a.w + xx;
            for (int t = 0; t < T; t++) {
                const float* x = &a.rgb[p * a.nc + 3 * t];
                const float* y = &b.rgb[p * b.nc + 3 * t];
                for (int c = 0; c < 3; c++) {
                    const int xi = (int)std::lround(std::min(1.0f, std::max(0.0f, x[c])) * 255.0f);
                    const int yi = (int)std::lround(std::min(1.0f, std::max(0.0f, y[c])) * 255.0f);
                    const double d = (double)(xi - yi); sse[t] += d * d;
                }
            }
        }
    const double npix = (double)ew * eh;
    Mse m;
    double ws = 0;
    for (int t = 0; t < T; t++) { m.tex[t] = sse[t] / (3.0 * npix); ws += (double)w[t] * sse[t]; }
    m.weighted = ws / (3.0 * wsum * npix);
    return m;
}
// PSNR as Basis Universal prints it: 20 log10(255 / rms), clamped to [0, 100], 100 for a zero error.
static double psnr_of(double mse) { return mse > 0 ? std::min(100.0, std::max(0.0, 10.0 * std::log10(255.0 * 255.0 / mse))) : 100.0; }

// ---------------------------------------------------------------- Adam
struct Adam {
    std::vector<float> m, v;
    float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
    int t = 0;
    void init(size_t n) { m.assign(n, 0); v.assign(n, 0); t = 0; }
    // Gradient-descent step: theta -= lr * adam(g)
    void step(std::vector<float>& theta, const std::vector<float>& g, float lr) {
        t++;
        float c1 = 1.0f - std::pow(b1, (float)t), c2 = 1.0f - std::pow(b2, (float)t);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < (int)theta.size(); i++) {
            m[i] = b1 * m[i] + (1 - b1) * g[i];
            v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i];
            float mh = m[i] / c1, vh = v[i] / c2;
            theta[i] -= lr * mh / (std::sqrt(vh) + eps);
        }
    }
};

// ---------------------------------------------------------------- ES: MLP
// Antithetic ES on the MLP weights, evaluated on one shared random minibatch.
// g = 1/(2 N sigma) * sum_i [L(p + s e_i) - L(p - s e_i)] e_i
struct MlpTrainer {
    Adam adam;
    std::vector<float> grad;
    std::vector<float> eps;   // N x P
    std::vector<int> batch;
    int cur_it = 0;           // iteration of the current step (keys the --rng hash draws, as on the GPU)

    void init(size_t P) { adam.init(P); grad.assign(P, 0); }

    // The pixel set one MLP step is evaluated on: a random minibatch (with replacement).
    void draw_batch(const Decoder& D, std::mt19937& rng, const Options& o) {
        batch.resize(o.mlp_batch);
        if (o.rng_hash) {
            for (size_t s = 0; s < batch.size(); s++) batch[s] = (int)ntc_noise_uniform(ntc_noise_u32(o.seed, NS_BATCH_PIX, cur_it, 0, s), (uint32_t)(D.W * D.H));
        } else {
            std::uniform_int_distribution<int> U(0, D.W * D.H - 1);
            for (auto& b : batch) b = U(rng);
        }
    }

    // Antithetic ES step. Returns the mean minibatch loss across all evaluations (for stats).
    double step(Decoder& D, const Image& target, std::mt19937& rng, const Options& o, double& diff_std) {
        const size_t P = D.mlp.size();
        const int N = o.mlp_pairs;
        eps.resize((size_t)N * P);
        if (o.rng_hash) {
            for (int i = 0; i < N; i++) for (size_t k = 0; k < P; k++) eps[(size_t)i * P + k] = ntc_gauss(o.seed, NS_MLP, cur_it, i, k);
        } else {
            std::normal_distribution<float> Nd(0.0f, 1.0f);
            for (auto& e : eps) e = Nd(rng);
        }
        draw_batch(D, rng, o);

        std::vector<double> dl(N), lsum(N);
        const float* p0 = D.mlp.p.data();
#pragma omp parallel
        {
            std::vector<float> pp(P), pm(P);
#pragma omp for schedule(dynamic)
            for (int i = 0; i < N; i++) {
                const float* e = &eps[(size_t)i * P];
                for (size_t k = 0; k < P; k++) { pp[k] = p0[k] + o.mlp_sigma * e[k]; pm[k] = p0[k] - o.mlp_sigma * e[k]; }
                double lp = D.loss_subset(pp.data(), D.zdec(), target, batch);
                double lm = D.loss_subset(pm.data(), D.zdec(), target, batch);
                dl[i] = lp - lm; lsum[i] = 0.5 * (lp + lm);
            }
        }
        double mean = 0, mean2 = 0, lmean = 0;
        for (int i = 0; i < N; i++) { mean += dl[i]; mean2 += dl[i] * dl[i]; lmean += lsum[i]; }
        mean /= N; mean2 /= N; lmean /= N;
        diff_std = std::sqrt(std::max(0.0, mean2 - mean * mean));

        std::fill(grad.begin(), grad.end(), 0.0f);
        float scale = 1.0f / (2.0f * N * o.mlp_sigma);
        for (int i = 0; i < N; i++) {
            float w = (float)dl[i] * scale;
            const float* e = &eps[(size_t)i * P];
            for (size_t k = 0; k < P; k++) grad[k] += w * e[k];
        }
        adam.step(D.mlp.p, grad, o.mlp_lr);
        return lmean;
    }

    // Central finite differences, one weight at a time, on a shared minibatch:
    //   dL/dw_j ~= [ L(w + h e_j) - L(w - h e_j) ] / (2h)
    // Exact up to O(h^2) and float rounding, no random-direction noise, but it
    // costs 2P minibatch evaluations per step versus 2N for ES. Neither ES nor
    // backprop: numerical differentiation. Meant as a late-training polish
    // once ES noise, not step count, limits the decoder.
    // Returns the unperturbed minibatch loss; diff_std receives the RMS of the
    // per-weight loss differences (printed as "fd-rms"; not comparable to the
    // ES "dstd", which is the spread of O(sigma) random-direction differences).
    // Note that Adam's second-moment estimate carries ES noise power across the
    // switch, which would throttle the first few hundred FD steps; main resets
    // the MLP Adam state when the FD phase begins.
    double step_fd(Decoder& D, const Image& target, std::mt19937& rng, const Options& o, double& diff_std) {
        const size_t P = D.mlp.size();
        const float h = o.mlp_fd_h;
        draw_batch(D, rng, o);

        const float* p0 = D.mlp.p.data();
        double l0 = D.loss_subset(p0, D.zdec(), target, batch);
        std::vector<double> dl(P), hh(P);
#pragma omp parallel
        {
            std::vector<float> pw(p0, p0 + P);   // per-thread working copy
#pragma omp for schedule(dynamic, 16)
            for (int j = 0; j < (int)P; j++) {
                float orig = pw[j];
                // Use the step sizes actually realized in float, not nominal h.
                float wp = orig + h, wm = orig - h;
                pw[j] = wp;
                double lp = D.loss_subset(pw.data(), D.zdec(), target, batch);
                pw[j] = wm;
                double lm = D.loss_subset(pw.data(), D.zdec(), target, batch);
                pw[j] = orig;
                dl[j] = lp - lm;
                hh[j] = (double)wp - (double)wm;
            }
        }
        double ss = 0;
        for (size_t j = 0; j < P; j++) { grad[j] = (float)(dl[j] / hh[j]); ss += dl[j] * dl[j]; }
        diff_std = std::sqrt(ss / P);
        adam.step(D.mlp.p, grad, o.mlp_lr);
        return l0;
    }
};

// ---------------------------------------------------------------- ES: latent
// [DCT] --dct-lambda part A (DCT_RATE_PLAN.md 2.3): the rate pass of one antithetic pair. dbits per (block, channel) = the proxy
// [DCT] bits of snap(clamp(z + sg eps)) - snap(clamp(z - sg eps)) on the perturbed shadow (not the snapped copy), attributed to
// [DCT] the block's 64 texels like the mse difference to a texel's footprint: grad[i] += rate_scale * dbits * scale * eps[i],
// [DCT] rate_scale = L / (16 * 65025 * npix) (the loss L * bits / npix in the internal LSB^2 / 255^2 units, bits in 1/16 bit),
// [DCT] scale = 1 / (2 K sg). hits counts the nonzero differences. Not inlined: LatentTrainer::step keeps its code shape (the
// [DCT] /fp:fast /arch:AVX2 build's numerics depend on it) and calls this only when the flag is on. = k_dct_rate_pair + k_dct_rate_gather.
#if defined(_MSC_VER)   // [DCT]
#define NTC_NOINLINE __declspec(noinline)   // [DCT]
#else   // [DCT]
#define NTC_NOINLINE __attribute__((noinline))   // [DCT]
#endif   // [DCT]
static NTC_NOINLINE void dct_rate_pass(Decoder& D, const std::vector<float>& eps, std::vector<float>& grad, std::vector<int32_t>& dbits, int K, float sg, size_t& hits) {   // [DCT]
    const DctLevel& Q = D.dct; const Latent& L0 = D.lat.lv[0];   // [DCT]
    const float rate_scale = (float)((double)Q.lambda / (16.0 * 65025.0 * (double)D.W * D.H));   // [DCT] = dct_desc_of's rate_scale
    const size_t nbc = Q.nblk() * (size_t)Q.C;   // [DCT]
    dbits.resize(nbc);   // [DCT]
    const float* z0 = D.lat.z.data() + L0.off; const float* e0 = eps.data() + L0.off;   // [DCT] the shadow, not zdec()
    const long long nb = (long long)Q.nblk();   // [DCT]
#pragma omp parallel for schedule(static)   // [DCT]
    for (long long b = 0; b < nb; b++)   // [DCT]
        for (int c = 0; c < Q.C; c++) dbits[(size_t)b * Q.C + c] = dct_rate_block(Q, L0, (size_t)b, c, z0, e0, sg, 1.0f) - dct_rate_block(Q, L0, (size_t)b, c, z0, e0, sg, -1.0f);   // [DCT]
    for (size_t i = 0; i < nbc; i++) if (dbits[i] != 0) hits++;   // [DCT]
    const float scale = 1.0f / (2.0f * K * sg);   // [DCT]
    for (int ty = 0; ty < L0.H; ty++)   // [DCT]
        for (int tx = 0; tx < L0.W; tx++) {   // [DCT]
            const size_t b = (size_t)(ty / 8) * Q.BW + (size_t)(tx / 8), base = L0.off + ((size_t)ty * L0.W + tx) * L0.C;   // [DCT]
            for (int c = 0; c < L0.C; c++) grad[base + c] += rate_scale * (float)dbits[b * Q.C + c] * scale * eps[base + c];   // [DCT]
        }   // [DCT]
}   // [DCT]
// All texels of all levels are perturbed simultaneously. For each antithetic
// pair we decode the full image twice and get per-pixel squared errors e+ and
// e-. Each pixel's (e+ - e-) is attributed, once per level, to the (up to) 4
// texels its bilinear tap reads on that level, so texel (x,y) accumulates the
// loss change over exactly its footprint.
//   g[x,y,c] = 1/(2 K sigma) * sum_pairs [ sum_{footprint} (e+ - e-) / (3 sum(w) W H) ] * eps[x,y,c]
struct LatentTrainer {
    Adam adam;
    std::vector<float> grad, eps, zp, zm;
    std::vector<float> ep, em;   // per-pixel errors
    std::vector<float> dtex;     // per-texel footprint loss difference (reused per level)
    std::vector<int32_t> dbits;  // [DCT] --dct-lambda: per (block, channel) proxy bit difference of the current pair (1/16 bit)
    int step_no = 0;             // counts steps: keys the --rng hash perturbation stream (NS_LAT), as lat_step_no does on the GPU

    void init(size_t n) { adam.init(n); grad.assign(n, 0); step_no = 0; }

    // [DCT] hook: the body is step_impl<RATE>; RATE = false is the pre-change function text (the --dct-lambda lines vanish under
    // [DCT] if constexpr, so its /fp:fast /arch:AVX2 codegen and the regression pin are untouched), RATE = true adds the rate pass.
    void step(Decoder& D, const Image& target, std::mt19937& rng, const Options& o) {   // [DCT] hook: was the body below
        if (D.dct.live && D.dct.rate_es) step_impl<true>(D, target, rng, o); else step_impl<false>(D, target, rng, o);   // [DCT]
    }   // [DCT]
    template <bool RATE> void step_impl(Decoder& D, const Image& target, std::mt19937& rng, const Options& o) {   // [DCT] hook: was `void step(...)`
        const LatentSet& LS = D.lat;
        const float* zb = D.zdec();   // --qes: perturb the snapped point; the update below goes to the fp32 shadow
        const size_t n = LS.size();
        const int K = o.lat_pairs;
        eps.resize(n); zp.resize(n); zm.resize(n);
        size_t maxtex = 0;
        for (const Latent& L : LS.lv) maxtex = std::max(maxtex, (size_t)L.W * L.H);
        dtex.resize(maxtex);
        std::fill(grad.begin(), grad.end(), 0.0f);
        std::normal_distribution<float> Nd(0.0f, 1.0f);
        const float inv_px = 1.0f / (3.0f * D.wsum * D.W * D.H);
        const float sg = o.lat_sigma;   // one sigma for every ES-trained level
        const size_t nlev = LS.lv.size();
        size_t rate_hits = 0; (void)rate_hits;   // [DCT] --dct-lambda part A (used by the RATE instantiation only)

        for (int k = 0; k < K; k++) {
            // One epsilon over the whole flat vector, drawn level by level (level 0 first).
            // ES-perturbed levels: all of them, or all but level 0 under --qat (it is searched instead).
            for (size_t l = 0; l < nlev; l++) {
                const Latent& L = LS.lv[l];
                const bool active = !(o.qat > 0 && l == 0);   // --qat: level 0 is discrete, eps = 0
                for (size_t i = L.off; i < L.off + L.size(); i++) {
                    if (active) {
                        eps[i] = o.rng_hash ? ntc_gauss(o.seed, NS_LAT, step_no, k, i) : Nd(rng);
                        zp[i] = zb[i] + sg * eps[i];
                        zm[i] = zb[i] - sg * eps[i];
                    } else {
                        eps[i] = 0.0f; zp[i] = zb[i]; zm[i] = zb[i];
                    }
                }
            }
            D.decode_err(D.mlp.p.data(), zp.data(), target, ep);
            D.decode_err(D.mlp.p.data(), zm.data(), target, em);

            // Scatter each pixel's loss difference into the distinct texels its tap
            // reads, once per level with that level's own tap and border guard.
            for (size_t l = 0; l < nlev; l++) {
                if (o.qat > 0 && l == 0) continue;            // discrete level 0: no scatter, no ES gradient
                const Latent& L = LS.lv[l];
                const size_t ntex = (size_t)L.W * L.H;
                const float scale = 1.0f / (2.0f * K * sg);
                std::fill(dtex.begin(), dtex.begin() + ntex, 0.0f);
                for (int py = 0; py < D.H; py++) {
                    float v = (py + 0.5f) / D.H;
                    for (int px = 0; px < D.W; px++) {
                        float u = (px + 0.5f) / D.W;
                        float d = (ep[(size_t)py * D.W + px] - em[(size_t)py * D.W + px]) * inv_px;
                        BilinearTap t = bilinear_tap(L, u, v);
                        dtex[(size_t)t.y0 * L.W + t.x0] += d;
                        if (t.x1 != t.x0) dtex[(size_t)t.y0 * L.W + t.x1] += d;
                        if (t.y1 != t.y0) {
                            dtex[(size_t)t.y1 * L.W + t.x0] += d;
                            if (t.x1 != t.x0) dtex[(size_t)t.y1 * L.W + t.x1] += d;
                        }
                    }
                }
                for (int ty = 0; ty < L.H; ty++)
                    for (int tx = 0; tx < L.W; tx++) {
                        float w = dtex[(size_t)ty * L.W + tx] * scale;
                        size_t base = L.off + ((size_t)ty * L.W + tx) * L.C;
                        for (int c = 0; c < L.C; c++) grad[base + c] += w * eps[base + c];
                    }
            }
            if constexpr (RATE) dct_rate_pass(D, eps, grad, dbits, K, sg, rate_hits);   // [DCT] hook: --dct-lambda part A, the rate pass of this pair (absent from the RATE = false instantiation)
        }
        if constexpr (RATE) { D.dct.rate_hits = rate_hits; D.dct.rate_evals = D.dct.nblk() * (size_t)D.dct.C * (size_t)K; }   // [DCT]
        adam.step(D.lat.z, grad, o.lat_lr);
        D.qes_refresh();   // --qes: re-snap the decode copy (and clamp the shadow once the ranges are frozen)
        step_no++;   // next step's noise key
    }
};

// ---------------------------------------------------------------- QAT: discrete level 0
// With --qat B every level-0 value is one of 2^B grid points -1 + 2k/(2^B-1), k = 0..2^B-1.
// The grid is fixed (no per-channel min/max), so these three functions are the single
// source of truth for the search, the initial snap, the loader and the bitrate histogram.
// Each channel has its own bit depth B (1..8). The grid values come from tables filled
// once (qat_init_grid), so every on-grid comparison sees the same bit pattern regardless
// of how the compiler evaluates the expression at different call sites.
static float QAT_GRID[9][257];   // [bits][index]
static void qat_init_grid() {
    for (int b = 1; b <= 8; b++) { int levels = (1 << b) - 1; for (int k = 0; k <= levels; k++) QAT_GRID[b][k] = -1.0f + 2.0f * k / (float)levels; }
}
static inline int qat_levels(int bits) { return (1 << bits) - 1; }   // largest grid index
static inline float qat_value(int k, int bits) { return QAT_GRID[bits][k]; }
static inline int qat_index(float v, int bits) {
    const int levels = qat_levels(bits);
    int k = (int)std::lround((v + 1.0f) * 0.5f * levels);
    return std::max(0, std::min(levels, k));
}
static inline float qat_snap(float v, int bits) { return qat_value(qat_index(v, bits), bits); }
// Snap every level-0 value to its channel's grid; returns how many values moved.
static size_t qat_snap_level0(LatentSet& LS, const std::vector<int>& ch_bits) {
    const Latent& L = LS.lv[0]; size_t moved = 0;
    for (size_t i = 0; i < L.size(); i++) { float sv = qat_snap(LS.z[i], ch_bits[i % L.C]); if (sv != LS.z[i]) { LS.z[i] = sv; moved++; } }
    return moved;
}
static int qat_decodes(const std::vector<int>& ch_bits) { int d = 0; for (int b : ch_bits) d += 1 << b; return d; }
static std::string qat_spec(const std::vector<int>& ch_bits) {
    std::string r; for (size_t c = 0; c < ch_bits.size(); c++) r += (c ? "," : "") + std::to_string(ch_bits[c]); return r;
}

// Exact coordinate descent on the discrete level 0. With nearest sampling a pixel's output
// depends on exactly one level-0 texel (Decoder::features copies that
// texel's C values into f[0..C) and nothing after depends on level-0 values), so the
// per-texel objective, the summed cw-weighted squared error over the texel's cell, is
// separable across texels: texels are searched independently and in parallel, and within
// a texel the channels are searched one after another, each over all 2^B grid values with
// the others held fixed. Ties keep the current value, so the loss never increases, and
// texels without pixels (latent finer than the image) are left alone. The MLP inputs of
// the cell's pixels are computed once per texel and only f[c] is patched per candidate,
// which is bit-identical to calling Decoder::pixel with the candidate written into z.
// The loss normalization (1 / (3 wsum W H)) does not affect the argmin and is omitted.
static void qat_search(Decoder& D, const Image& target, const std::vector<int>& ch_bits) {
    const Latent& L = D.lat.lv[0];
    const int nc = D.mlp.nout, nin = D.nin();
    // Pixel range [xb, xe) x [yb, ye) of each texel, from the same nearest tap features() uses.
    std::vector<int> xb(L.W, D.W), xe(L.W, 0), yb(L.H, D.H), ye(L.H, 0);
    for (int px = 0; px < D.W; px++) { int t = bilinear_tap(L, (px + 0.5f) / D.W, 0.5f).x0; xb[t] = std::min(xb[t], px); xe[t] = std::max(xe[t], px + 1); }
    for (int py = 0; py < D.H; py++) { int t = bilinear_tap(L, 0.5f, (py + 0.5f) / D.H).y0; yb[t] = std::min(yb[t], py); ye[t] = std::max(ye[t], py + 1); }
    const float* p = D.mlp.p.data();
    float* z = D.zdec_mut();   // the decode buffer (level 0 starts at offset 0; each thread writes only its own texels)
#pragma omp parallel
    {
        std::vector<float> feat;   // per-thread: nin inputs for each pixel of the current cell
#pragma omp for schedule(dynamic)
        for (int ty = 0; ty < L.H; ty++) {
            for (int tx = 0; tx < L.W; tx++) {
                const int x0 = xb[tx], x1 = xe[tx], y0 = yb[ty], y1 = ye[ty];
                if (x0 >= x1 || y0 >= y1) continue;
                const int npx = (x1 - x0) * (y1 - y0);
                feat.resize((size_t)npx * nin);
                { int j = 0; for (int py = y0; py < y1; py++) for (int px = x0; px < x1; px++, j++) D.features(z, px, py, &feat[(size_t)j * nin]); }
                float* zt = z + ((size_t)ty * L.W + tx) * L.C;
                for (int c = 0; c < L.C; c++) {
                    const int bits = ch_bits[c], levels = qat_levels(bits);
                    auto cell_loss = [&](float val) {
                        double sum = 0; int j = 0;
                        for (int py = y0; py < y1; py++)
                            for (int px = x0; px < x1; px++, j++) {
                                float* f = &feat[(size_t)j * nin]; f[c] = val;
                                float out[MAXOUT];
                                mlp_forward(D.mlp, p, f, out, D.opt->clamp_out);
                                const float* t = &target.rgb[((size_t)py * D.W + px) * nc];
                                for (int q = 0; q < nc; q++) { float d = out[q] - t[q]; sum += D.cw[q] * (d * d); }
                            }
                        return sum;
                    };
                    int best_k = qat_index(zt[c], bits);            // zt[c] is on-grid: this is its index
                    double best = cell_loss(qat_value(best_k, bits));   // the current value first; ties keep it
                    for (int k = 0; k <= levels; k++) {
                        if (k == best_k) continue;
                        double sk = cell_loss(qat_value(k, bits));
                        if (sk < best) { best = sk; best_k = k; }
                    }
                    const float v = qat_value(best_k, bits);
                    zt[c] = v;
                    for (int j = 0; j < npx; j++) feat[(size_t)j * nin + c] = v;   // fixed for the next channel
                }
            }
        }
    }
    // --qes on another level: the search ran on the snapped copy, so carry its level-0 result back to the shadow.
    if (D.qes_live) memcpy(D.lat.z.data(), D.zq.data(), L.size() * sizeof(float));
}

// ---------------------------------------------------------------- I/O helpers
static void save_latent_png(const std::string& path, const Latent& L, const float* z) {
    // Channels laid out side by side, each mapped from [-1.5,1.5] to [0,255].
    int w = L.W * L.C, h = L.H;
    std::vector<unsigned char> buf((size_t)w * h);
    for (int y = 0; y < h; y++)
        for (int c = 0; c < L.C; c++)
            for (int x = 0; x < L.W; x++) {
                float v = z[((size_t)y * L.W + x) * L.C + c];
                float m = std::min(1.0f, std::max(0.0f, v / 3.0f + 0.5f));
                buf[(size_t)y * w + c * L.W + x] = (unsigned char)std::lround(m * 255.0f);
            }
    stbi_write_png(path.c_str(), w, h, 1, buf.data(), w);
}

static void save_model(const std::string& path, const Decoder& D) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return;
    // v6 header: magic, LW0, LH0, LC0, nin, poslen, act, nlayers, nlevels, textures,
    // then one int per level (0 bilinear, 1 nearest), then (W,H,C) for each level
    // after the first, then (v12) per level the --qes bits and C (lo, hi) pairs, then the hidden widths, then
    // poslen bytes of positional spec text, then the flat latent (level 0
    // first, each level (y,x,c) with c fastest), then the MLP parameters
    // (3 * textures outputs).
    // [DCT] v14 (magic 0x4E54433E) only when level 0 is live DCT-coded: v12 through the per-level --qes fields, then
    // [DCT] int dct_N, dct_dc_step, dct_C, int dct_deadzone (0/1; v14, the loaders read v13 as 1), then dct_C ints q[c], then the
    // [DCT] hidden widths and spec as v12, then the payload int16 sym[BH*BW][dct_C][64] (natural order), uint8 code[BH*BW][dct_C],
    // [DCT] the floats of levels >= 1 only, the MLP. (v13, magic 0x4E54433D, is the same layout without the dct_deadzone int.)
    const bool dct13 = D.dct.live;   // [DCT]
    int hdr[10] = { dct13 ? 0x4E54433E : 0x4E54433C, D.lat.lv[0].W, D.lat.lv[0].H, D.lat.lv[0].C, D.mlp.nin,   // [DCT] hook: the magic (v14 since --dct-deadzone)
                    (int)D.pos.spec.size(), 0 /* act: always leaky; the int stays for the v6..v12 layout */, (int)D.mlp.hidden.size(), (int)D.lat.lv.size(), D.mlp.nout / 3 };
    fwrite(hdr, sizeof(hdr), 1, f);
    for (size_t l = 0; l < D.lat.lv.size(); l++) { int nr = D.lat.lv[l].nearest ? 1 : 0; fwrite(&nr, sizeof(int), 1, f); }
    { int db = 0; float fo = 1.0f; fwrite(&db, sizeof(int), 1, f); fwrite(&fo, sizeof(float), 1, f); }   // v7 deblock int / falloff float: layout only (deblocking was removed in v0.8; the loader refuses db == 1)
    fwrite(&D.mlp.leak, sizeof(float), 1, f);   // v8
    fwrite(&D.qat_bits, sizeof(int), 1, f);     // v9: largest --qat bit depth (0 = off)
    for (int c = 0; c < D.lat.lv[0].C; c++) { int b = D.qat_bits > 0 ? D.qat_ch[c] : 0; fwrite(&b, sizeof(int), 1, f); }   // v10: bits per level-0 channel
    {   // v11: decoded (padded) image size, source size before --block padding, and the output nonlinearity (1 = --clamp, 0 = sigmoid)
        int img[5] = { D.W, D.H, D.opt->orig_w, D.opt->orig_h, D.opt->clamp_out ? 1 : 0 };
        fwrite(img, sizeof(img), 1, f);
    }
    for (size_t l = 1; l < D.lat.lv.size(); l++) {
        int d[3] = { D.lat.lv[l].W, D.lat.lv[l].H, D.lat.lv[l].C };
        fwrite(d, sizeof(d), 1, f);
    }
    for (size_t l = 0; l < D.lat.lv.size(); l++) {   // v12: per level, --qes bits (0 = continuous) then C (lo, hi) range floats
        const bool q = D.qes_live && D.qes[l].fitted() && !D.qes[l].from_ntcb;   // [NTCB] hook: a grid reinstalled from a container is written as the continuous level it was (bits 0; its values are on the grid)
        int b = q ? D.qes[l].bits : 0; fwrite(&b, sizeof(int), 1, f);
        std::vector<float> lo(D.lat.lv[l].C, 0.0f), hi(D.lat.lv[l].C, 0.0f);
        if (q) { lo = D.qes[l].lo; hi = D.qes[l].hi; }
        fwrite(lo.data(), sizeof(float), lo.size(), f); fwrite(hi.data(), sizeof(float), hi.size(), f);
    }
    if (dct13) {   // [DCT] v14 header fields
        int di[3] = { D.dct.N, D.dct.dc_step, D.dct.C };   // [DCT]
        fwrite(di, sizeof(di), 1, f);   // [DCT]
        { const int dz = D.dct.deadzone ? 1 : 0; fwrite(&dz, sizeof(int), 1, f); }   // [DCT] v14: the --dct-deadzone flag
        fwrite(D.dct.q.data(), sizeof(int), D.dct.q.size(), f);   // [DCT]
    }   // [DCT]
    fwrite(D.mlp.hidden.data(), sizeof(int), D.mlp.hidden.size(), f);
    fwrite(D.pos.spec.data(), 1, D.pos.spec.size(), f);
    if (dct13) {   // [DCT] v13 payload: symbols, codes, then the floats of levels >= 1
        fwrite(D.dct.sym.data(), sizeof(int16_t), D.dct.sym.size(), f);   // [DCT]
        fwrite(D.dct.code.data(), sizeof(uint8_t), D.dct.code.size(), f);   // [DCT]
        const size_t n0 = D.lat.lv[0].size();   // [DCT]
        fwrite(D.zdec() + n0, sizeof(float), D.lat.z.size() - n0, f);   // [DCT]
    } else   // [DCT]
    fwrite(D.zdec(), sizeof(float), D.lat.z.size(), f);   // --qes levels as their on-grid (snapped) values; others as the shadow
    fwrite(D.mlp.p.data(), sizeof(float), D.mlp.p.size(), f);
    fclose(f);
}

static void latent_stats(const Latent& L, const float* z, float& mean, float& sd, float& mx) {
    double s = 0, s2 = 0; mx = 0;
    for (size_t i = 0; i < L.size(); i++) { float v = z[i]; s += v; s2 += (double)v * v; mx = std::max(mx, std::fabs(v)); }
    mean = (float)(s / L.size());
    sd = (float)std::sqrt(std::max(0.0, s2 / L.size() - (s / L.size()) * (s / L.size())));
}


// ---------------------------------------------------------------- bitrate
// Effective compressed size estimate. The latent is quantized to 8 bits per
// channel with a per-channel min/max scale (2 floats per channel overhead), and
// the MLP weights are counted as fp16. Two latent numbers are reported: raw
// 8 bits/texel/channel, and the zeroth-order entropy of the quantized symbols.
// A third number applies to level 0 (the selector planes) the conditional entropy of
// each symbol given its upper and left neighbours, per channel, with the full symbol
// values as the context: the ideal rate of an adaptive arithmetic coder with an
// order-2 (up, left) context model. Levels >= 1 keep their zeroth-order entropy in that
// total. Only an estimate; nothing is coded.
// The dequantized latent is also returned so the caller can measure the PSNR
// the codec would actually achieve at that bitrate.
struct BitrateStats {
    double bpp_fp32;      // everything stored as fp32
    double bpp_q8;        // 8-bit latent (raw) + fp16 MLP
    double bpp_q8_ent;    // entropy-coded 8-bit latent + fp16 MLP (zeroth order)
    double bpp_q8_ctx;    // the same with level 0 conditioned on its up and left neighbours (other levels zeroth order)
    double bpp_ctx_level0;   // level 0's (up, left)-conditional part alone, in bpp of the image
    double bpp_est = 0;   // [EST] the "est bitrate" (his definition, 2026-09-08): level 0 order-0 (h0), levels >= 1 per-channel DPCM (average of left and up) residuals order-0, fp16 MLP
    double bits_mlp;      // fp16 MLP bits
    double bits_raw_latent = 0, bits_header = 0;   // [NTCB] the raw latent bits and the side-info (min/max header) bits behind bpp_q8; read only by ntcb_write
};

// Conditional entropy H(X | up, left) in bits, summed over the texels of one channel of a
// W x H level whose symbols are idx[y * W + x]; out-of-image neighbours count as symbol 0.
// H(X | U, L) = H(X, U, L) - H(U, L), from hash-counted triples and pairs, with the full
// neighbour values as the context (an adaptive arithmetic coder learns each context's
// distribution online, so for the selector alphabets, at most 256 contexts of 16 symbols
// over millions of texels, the model-learning cost is negligible and this is its ceiling).
static double context_entropy_bits(const std::vector<int>& idx, int W, int H) {
    std::unordered_map<uint64_t, int> triples, pairs;
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            const uint64_t s = (uint64_t)idx[(size_t)y * W + x];
            const uint64_t u = y ? (uint64_t)idx[(size_t)(y - 1) * W + x] : 0;
            const uint64_t l = x ? (uint64_t)idx[(size_t)y * W + x - 1] : 0;
            pairs[(u << 20) | l]++;
            triples[(u << 40) | (l << 20) | s]++;
        }
    const double n = (double)W * H;
    double hxul = 0, hul = 0;
    for (auto& kv : triples) hxul -= kv.second * std::log2(kv.second / n);
    for (auto& kv : pairs) hul -= kv.second * std::log2(kv.second / n);
    return hxul - hul;
}

// With qat_bits > 0, level 0 is already discrete: it is charged qat_bits per value with no
// min/max header, its entropy is that of the 2^qat_bits grid indices, and zq keeps its values
// as they are (re-quantizing them with a min/max scale would be wrong). A live --qes level is
// discrete too: charged its --qes bits per value plus the 64-bit min/max header per channel,
// entropy over its grid indices, zq = its snapped decode values. Before --qes-start such a level
// is still continuous and gets the post-hoc quantization at its --qes depth (so the quoted
// bitrate is the run's final format throughout). Other levels use qbits.
static BitrateStats bitrate_stats(const Decoder& D, int qbits, std::vector<float>& zq) {
    const LatentSet& LS = D.lat; const MLP& m = D.mlp;
    const std::vector<int>& qat_ch = D.qat_ch;
    BitrateStats s;
    const double npix = (double)D.W * D.H;
    s.bits_mlp = m.size() * 16.0;
    s.bpp_fp32 = (LS.size() * 32.0 + m.size() * 32.0) / npix;
    zq.resize(LS.size());
    double ent_bits = 0, header_bits = 0, raw_bits = 0, ctx_bits = 0, est_bits = 0;   // [EST] est_bits
    s.bpp_ctx_level0 = 0;
    for (size_t l = 0; l < LS.lv.size(); l++) {   // each level quantized independently, per channel
        const Latent& L = LS.lv[l];
        const bool fixed = (l == 0 && !qat_ch.empty());   // discrete level: fixed per-channel grids, no scale header
        const bool qes = D.qes_live && D.qes[l].fitted();   // --qes level: its own grid, decode values already snapped
        const QesLevel* Q = qes ? &D.qes[l] : nullptr;
        const int lbits = (!fixed && !qes && D.qes_on && D.qes[l].bits > 0) ? D.qes[l].bits : qbits;   // post-hoc depth of a continuous level
        const float* z = LS.z.data() + L.off;
        const float* zd = D.zdec() + L.off;
        float* q = zq.data() + L.off;
        const size_t nl = L.size();
        const double ntex = (double)L.W * L.H;
        if (l == 0 && D.dct.live) {   // [DCT] hook: a live DCT-coded level 0 is charged by the bit simulator (symbols + scale codes + a 96-bit header per channel), no per-texel bits, no min/max header; the returned quantized latent is the snapped plane
            const DctStats b = dct_analyze(D.dct);   // [DCT]
            raw_bits += b.raw_total - b.header; ent_bits += b.h0_total - b.header; ctx_bits += b.ctx_total - b.header;   // [DCT]
            est_bits += b.h0_total - b.header;   // [EST] level 0: order-0 on the tokens
            s.bpp_ctx_level0 += (b.ctx_total - b.header) / npix;   // [DCT]
            header_bits += b.header;   // [DCT]
            memcpy(q, zd, nl * sizeof(float));   // [DCT]
            continue;   // [DCT]
        }   // [DCT]
        if (!fixed && !qes) raw_bits += nl * (double)lbits;
        for (int c = 0; c < L.C; c++) {
            const int bits = fixed ? qat_ch[c] : (qes ? Q->bits : lbits);
            const int levels = (1 << bits) - 1;
            if (fixed || qes) raw_bits += ntex * (double)bits;
            float lo = 1e30f, hi = -1e30f;
            if (!fixed && !qes) for (size_t i = c; i < nl; i += L.C) { lo = std::min(lo, z[i]); hi = std::max(hi, z[i]); }
            float range = std::max(hi - lo, 1e-6f);
            std::vector<int> hist(levels + 1, 0);
            std::vector<int> idx((size_t)L.W * L.H);
            for (size_t i = c, t = 0; i < nl; i += L.C, t++) {
                int qi;
                if (fixed) { qi = qat_index(z[i], bits); q[i] = z[i]; }
                else if (qes) { qi = qes_index(z[i], Q->lo[c], Q->range[c], levels); q[i] = zd[i]; }
                else {
                    qi = q8_index(z[i], lo, range, levels);   // [NTCB] hook: strict FP (was the same expression inline under /fp:fast; see q8_index)
                    q[i] = q8_value(qi, lo, range, levels);   // [NTCB] hook: strict FP, the value the container readers reconstruct
                }
                hist[qi]++;
                idx[t] = qi;
            }
            double h0 = 0;
            for (int qv = 0; qv <= levels; qv++)
                if (hist[qv]) h0 -= hist[qv] * std::log2(hist[qv] / ntex);
            ent_bits += h0;
            if (l == 0) { const double cb = g_ctx_stats ? context_entropy_bits(idx, L.W, L.H) : h0; ctx_bits += cb; s.bpp_ctx_level0 += cb / npix; }   // progress prints: order-0 stands in
            else ctx_bits += h0;
            if (l == 0) est_bits += h0;   // [EST] a non-DCT level 0 (searched / continuous selectors): order-0
            else {   // [EST] levels >= 1: lossless DPCM, predictor = floor((left + up) / 2) (left alone on the first row, up alone on the first column, 0 at the origin), residual mod 2^bits, order-0 per channel
                std::vector<int> rh(levels + 1, 0);
                for (int y = 0; y < L.H; y++) for (int x = 0; x < L.W; x++) {
                    const int cur = idx[(size_t)y * L.W + x];
                    const int lf = x ? idx[(size_t)y * L.W + x - 1] : -1, up = y ? idx[(size_t)(y - 1) * L.W + x] : -1;
                    const int pred = (lf >= 0 && up >= 0) ? (lf + up) / 2 : (lf >= 0 ? lf : (up >= 0 ? up : 0));
                    rh[(cur - pred) & levels]++;
                }
                double hr = 0;
                for (int qv = 0; qv <= levels; qv++) if (rh[qv]) hr -= rh[qv] * std::log2(rh[qv] / ntex);
                est_bits += hr;
            }
        }
        if (!fixed) header_bits += L.C * 2 * 32.0; // per-channel min/max, per level (--qes levels included)
    }
    s.bpp_q8 = (raw_bits + header_bits + s.bits_mlp) / npix;
    s.bpp_q8_ent = (ent_bits + header_bits + s.bits_mlp) / npix;
    s.bpp_q8_ctx = (ctx_bits + header_bits + s.bits_mlp) / npix;
    s.bpp_est = (est_bits + header_bits + s.bits_mlp) / npix;   // [EST]
    s.bits_raw_latent = raw_bits; s.bits_header = header_bits;   // [NTCB] the terms behind bpp_q8, read by ntcb_write's reconciliation only
    return s;
}

// ---- NTCB: bit-packed container [NTCB]
// [NTCB] --write-ntcb writes <out>/model.ntcb, the fixed-width container of NTCB_PLAN.md section 1 (no entropy coding):
// [NTCB] a header, then one raw section per latent level (mode 0 = the post-hoc --qbits grid, 1 = a --qes grid, 2 = the
// [NTCB] --qat palette, 3 = the DCT symbols and scale codes as DC / (run, |q| - 1, sign) / EOB tokens) and one fp16 MLP
// [NTCB] section. The unpadded content bits must equal bitrate_stats' raw latent bits + fp16 MLP bits exactly (the file:
// [NTCB] line); anything else is a MISMATCH. Off by default: with the flag absent nothing in this region runs except the
// [NTCB] loader's magic test. Every multi-byte field is little-endian, written and read byte by byte; bit fields are packed
// [NTCB] LSB-first. Six-bit sync markers (container overhead, not content) open and close every level body and the MLP body
// [NTCB] and open every channel of a DCT body, so a reader that drifts by one bit refuses instead of decoding garbage. The
// [NTCB] header's hash field covers the whole file (the field itself read as zero). Readers: ntcb_restore below (the --load
// [NTCB] path) and ntc_decode.cpp load_ntcb, a hand-written mirror of this layout: a change here must be made there too and
// [NTCB] re-verified (NTCB_NOTES.md).
static const uint32_t NTCB_MAGIC = 0x6263746Eu;   // [NTCB] the bytes 6E 74 63 62 ("ntcb") read as a little-endian uint32
static const int NTCB_VERSION = 2;                // [NTCB] container version (2 since --dct-deadzone: a mode-3 record carries a dct_flags byte; both readers accept 1)   [DCT]
static const int NTCB_FIXED_HEADER = 52;          // [NTCB] header bytes before the per-texture channel counts
// [NTCB] Sync markers, 6 bits each; pairwise Hamming distance >= 3, none all-zero or all-one.
static const uint32_t NTCB_SYNC_LEVEL = 0x2Bu;    // [NTCB] 101011: start of a level body
static const uint32_t NTCB_SYNC_CHANNEL = 0x16u;  // [NTCB] 010110: start of a channel inside a DCT level body
static const uint32_t NTCB_SYNC_MLP = 0x31u;      // [NTCB] 110001: start of the MLP body
static const uint32_t NTCB_SYNC_END = 0x0Cu;      // [NTCB] 001100: end of a level body or of the MLP body
static const int NTCB_SYNC_BITS = 6;              // [NTCB]
// [NTCB] every marker also encodes where it sits: the same kind at another level or channel carries a different value
// (13 * level + 7 * channel is distinct for every (level 0..2, channel 0..3) pair), so a reader that lands on the right
// kind of marker in the wrong section still refuses. The MLP markers use level = nlevels. ntc_decode mirrors this.
static inline uint32_t ntcb_sync_value(uint32_t base, int level, int channel) { return (base + 13u * (uint32_t)level + 7u * (uint32_t)channel) & 63u; }   // [NTCB]

struct NtcbBitWriter {   // [NTCB] LSB-first bit packer: a 64-bit accumulator filled at the top, bytes emitted from the bottom
    std::vector<uint8_t> out; uint64_t acc = 0; int nacc = 0; double bits = 0, sync = 0;   // bits: everything put so far (unpadded); sync: the marker bits among them
    void put(uint32_t v, int n) {
        if (n < 32) v &= (1u << n) - 1u;
        acc |= (uint64_t)v << nacc; nacc += n; bits += n;
        while (nacc >= 8) { out.push_back((uint8_t)(acc & 0xFFu)); acc >>= 8; nacc -= 8; }
    }
    void marker(uint32_t m) { put(m, NTCB_SYNC_BITS); sync += NTCB_SYNC_BITS; }
    void align() { if (nacc > 0) { out.push_back((uint8_t)(acc & 0xFFu)); acc = 0; nacc = 0; } }   // zero padding to the byte
};
struct NtcbBitReader {   // [NTCB] the writer's mirror, bounds-checked: `ok` drops on the first read past `end` (like ntc_decode's Reader)
    const uint8_t* p; size_t pos, end; uint64_t acc = 0; int nacc = 0; bool ok = true; double sync = 0;
    NtcbBitReader(const uint8_t* p_, size_t begin, size_t end_) : p(p_), pos(begin), end(end_) {}
    uint32_t get(int n) {
        while (nacc < n) { if (pos >= end) { ok = false; return 0; } acc |= (uint64_t)p[pos++] << nacc; nacc += 8; }
        const uint32_t v = (uint32_t)(acc & ((1ull << n) - 1ull)); acc >>= n; nacc -= n; return v;
    }
    bool marker(uint32_t m) { const uint32_t v = get(NTCB_SYNC_BITS); sync += NTCB_SYNC_BITS; return ok && v == m; }   // false: drifted or truncated
    double consumed() const { return (double)(pos * 8) - (double)nacc; }   // bits taken from the stream so far (absolute, from p)
    void align() { acc = 0; nacc = 0; }   // drop the rest of the current byte (at most 7 bits); pos is then the next byte
};
// [NTCB] Little-endian byte helpers: no host-order memory access anywhere in the container code.
static void ntcb_put8(std::vector<uint8_t>& b, uint32_t v) { b.push_back((uint8_t)(v & 0xFFu)); }
static void ntcb_put16(std::vector<uint8_t>& b, uint32_t v) { ntcb_put8(b, v); ntcb_put8(b, v >> 8); }
static void ntcb_put32(std::vector<uint8_t>& b, uint32_t v) { ntcb_put16(b, v & 0xFFFFu); ntcb_put16(b, v >> 16); }
static void ntcb_putf32(std::vector<uint8_t>& b, float f) { uint32_t u; memcpy(&u, &f, 4); ntcb_put32(b, u); }
static void ntcb_patch32(std::vector<uint8_t>& b, size_t at, uint32_t v) { for (int k = 0; k < 4; k++) b[at + k] = (uint8_t)((v >> (8 * k)) & 0xFFu); }
struct NtcbByteReader {   // [NTCB] bounds-checked little-endian field reader
    const uint8_t* p; size_t n, pos = 0; bool ok = true;
    NtcbByteReader(const uint8_t* p_, size_t n_) : p(p_), n(n_) {}
    uint32_t u8() { if (pos + 1 > n) { ok = false; return 0; } return p[pos++]; }
    uint32_t u16() { const uint32_t a = u8(), b = u8(); return a | (b << 8); }
    uint32_t u32() { const uint32_t a = u16(), b = u16(); return a | (b << 16); }
    float f32() { const uint32_t u = u32(); float f; memcpy(&f, &u, 4); return f; }
};
static uint32_t ntcb_fnv1a32(const uint8_t* p, size_t n) { uint32_t h = 2166136261u; for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 16777619u; } return h; }
// [NTCB] The file hash: FNV-1a 32 over every byte with the hash field (offset 28..31) read as zero, so the header is covered too.
static uint32_t ntcb_file_hash(const uint8_t* p, size_t n) {
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < n; i++) { h ^= (i >= 28 && i < 32) ? 0u : p[i]; h *= 16777619u; }
    return h;
}
// [NTCB] Finiteness by bit pattern (std::isfinite can be folded away under -ffast-math on the GCC build).
static inline bool ntcb_finite32(float f) { uint32_t u; memcpy(&u, &f, 4); return (u & 0x7F800000u) != 0x7F800000u; }
static inline bool ntcb_finite16(uint16_t h) { return (h & 0x7C00u) != 0x7C00u; }

// [NTCB] IEEE binary16 conversions: round-to-nearest-even, denormals handled, overflow reported (never folded to infinity silently).
static uint16_t ntcb_f32_to_f16(float f, bool* overflow) {
    uint32_t u; memcpy(&u, &f, 4);
    const uint32_t sign = (u >> 16) & 0x8000u; const int exp = (int)((u >> 23) & 0xFFu); uint32_t man = u & 0x7FFFFFu;
    if (exp == 0xFF) { *overflow = true; return (uint16_t)(sign | 0x7C00u | (man ? 0x200u : 0u)); }   // inf / nan: refused by the callers
    const int e = exp - 127 + 15;
    if (e >= 0x1F) { *overflow = true; return (uint16_t)(sign | 0x7C00u); }
    if (e <= 0) {   // zero or a half denormal: value = man24 * 2^(e - 14) units of 2^-24
        if (e < -10) return (uint16_t)sign;   // below half the smallest denormal: rounds to zero
        man |= 0x800000u;
        const int shift = 14 - e;   // 14..24
        uint32_t half = man >> shift; const uint32_t rem = man & ((1u << shift) - 1u), mid = 1u << (shift - 1);
        if (rem > mid || (rem == mid && (half & 1u))) half++;   // a carry into bit 10 is the smallest normal, which is right
        return (uint16_t)(sign | half);
    }
    uint32_t half = sign | ((uint32_t)e << 10) | (man >> 13);
    const uint32_t rem = man & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (half & 1u))) { half++; if ((half & 0x7C00u) == 0x7C00u) *overflow = true; }
    return (uint16_t)half;
}
static float ntcb_f16_to_f32(uint16_t h) {
    const uint32_t sign = ((uint32_t)h & 0x8000u) << 16; const int e = (h >> 10) & 0x1F; uint32_t man = h & 0x3FFu, u;
    if (e == 0) {
        if (man == 0) u = sign;
        else { int ee = -1; do { ee++; man <<= 1; } while ((man & 0x400u) == 0); u = sign | ((uint32_t)(127 - 15 - ee) << 23) | ((man & 0x3FFu) << 13); }
    } else if (e == 31) u = sign | 0x7F800000u | (man << 13);
    else u = sign | ((uint32_t)(e - 15 + 127) << 23) | (man << 13);
    float f; memcpy(&f, &u, 4); return f;
}
// [NTCB] Every finite half pattern must survive f16 -> f32 -> f16 (run once when the flag is on); a few rounding cases too.
static bool ntcb_selfcheck() {
    size_t bad = 0;
    for (uint32_t h = 0; h < 65536; h++) {
        if (((h >> 10) & 0x1F) == 0x1F) continue;   // inf / nan patterns are refused, not round-tripped
        bool ov = false; const uint16_t back = ntcb_f32_to_f16(ntcb_f16_to_f32((uint16_t)h), &ov);
        if (ov || back != (uint16_t)h) bad++;
    }
    bool ov = false;
    const bool cases = ntcb_f32_to_f16(1.0f, &ov) == 0x3C00u && ntcb_f32_to_f16(-2.0f, &ov) == 0xC000u && ntcb_f32_to_f16(65504.0f, &ov) == 0x7BFFu
        && ntcb_f32_to_f16(5.9604645e-8f, &ov) == 0x0001u && ntcb_f32_to_f16(2.9802322e-8f, &ov) == 0x0000u && ntcb_f32_to_f16(1.00048828f, &ov) == 0x3C00u   // 1 + 2^-11: tie to even
        && ntcb_f32_to_f16(1.00073242f, &ov) == 0x3C01u && !ov;   // 1 + 3 * 2^-12 rounds up
    ov = false; ntcb_f32_to_f16(65520.0f, &ov); const bool ovf = ov;   // rounds to 2^16: overflow
    if (bad || !cases || !ovf) { printf("ntcb     : fp16 self-check FAILED (%zu of 65536 patterns do not round-trip; rounding cases %s, overflow detection %s)\n", bad, cases ? "ok" : "wrong", ovf ? "ok" : "wrong"); return false; }
    return true;
}
// [NTCB] Round every weight through fp16 (decision 2 of NTCB_PLAN.md). False when a weight is non-finite or above 65504.
static bool ntcb_mlp_round_fp16(std::vector<float>& p, size_t& changed, float& maxdw) {
    changed = 0; maxdw = 0.0f;
    for (float& w : p) {
        bool ov = false; const uint16_t h = ntcb_f32_to_f16(w, &ov);
        if (ov || !ntcb_finite32(w)) return false;
        const float r = ntcb_f16_to_f32(h);
        if (r != w) { changed++; maxdw = std::max(maxdw, std::fabs(r - w)); w = r; }
    }
    return true;
}

struct NtcbLevel {   // [NTCB] one level record of the header (NTCB_PLAN.md 1.1)
    int W = 0, H = 0, C = 0, filter = 0, mode = 0, dc_step = 0;   // filter 0 bilinear / 1 nearest; mode 0 q8 / 1 qes / 2 qat / 3 dct
    int deadzone = 1;                         // [DCT] mode 3: bit 0 of the version-2 dct_flags byte (1 = dead-zone AC quantizer, 0 = plain rounding); 1 in a version-1 file
    std::vector<int> bits, q;                 // per channel: bit depth (modes 0..2; 0 for mode 3) and DCT quality (mode 3)
    std::vector<float> lo, hi;                // modes 0 / 1: the per-channel grid range (the simulator's 64-bit header charge)
    std::vector<std::vector<float>> palette;  // mode 2: 2^bits[c] values per channel (QAT_GRID[bits][0..levels])
};
struct NtcbHeader {
    int version = NTCB_VERSION, H = 0, img_w = 0, img_h = 0, src_w = 0, src_h = 0;
    uint32_t file_bytes = 0, hash = 0;
    int latent_cell = 0, dct_block = 0, T = 1, nlevels = 1, act = 0, clamp = 0, nhidden = 1;
    float leak = 0.01f; int nin = 0, nout = 0;
    std::vector<int> cpt;   // channels per texture, T entries (3 each today; nout = their sum)
    std::vector<int> hidden; std::string pos;
    std::vector<NtcbLevel> lv;
};
// [NTCB] Serialize the header; H (offset 6) is patched here, file_bytes (24) and hash (28) by the writer once the payload exists.
static bool ntcb_write_header(const NtcbHeader& h, std::vector<uint8_t>& b) {
    b.clear();
    ntcb_put32(b, NTCB_MAGIC); ntcb_put16(b, (uint32_t)h.version); ntcb_put16(b, 0);
    ntcb_put32(b, (uint32_t)h.img_w); ntcb_put32(b, (uint32_t)h.img_h); ntcb_put32(b, (uint32_t)h.src_w); ntcb_put32(b, (uint32_t)h.src_h);
    ntcb_put32(b, 0); ntcb_put32(b, 0);
    ntcb_put8(b, (uint32_t)h.latent_cell); ntcb_put8(b, (uint32_t)h.dct_block); ntcb_put8(b, (uint32_t)h.T); ntcb_put8(b, (uint32_t)h.nlevels);
    ntcb_put8(b, (uint32_t)h.act); ntcb_put8(b, (uint32_t)h.clamp); ntcb_put8(b, (uint32_t)h.nhidden); ntcb_put8(b, 0);
    ntcb_putf32(b, h.leak); ntcb_put32(b, (uint32_t)h.nin); ntcb_put32(b, (uint32_t)h.nout);
    for (int t = 0; t < h.T; t++) ntcb_put8(b, (uint32_t)h.cpt[t]);
    for (int w : h.hidden) ntcb_put16(b, (uint32_t)w);
    ntcb_put16(b, (uint32_t)h.pos.size()); for (char c : h.pos) ntcb_put8(b, (uint32_t)(uint8_t)c);
    for (const NtcbLevel& L : h.lv) {
        ntcb_put32(b, (uint32_t)L.W); ntcb_put32(b, (uint32_t)L.H); ntcb_put16(b, (uint32_t)L.C); ntcb_put8(b, (uint32_t)L.filter); ntcb_put8(b, (uint32_t)L.mode);
        for (int c = 0; c < L.C; c++) ntcb_put8(b, (uint32_t)L.bits[c]);
        if (L.mode == 3) { ntcb_put8(b, (uint32_t)L.dc_step); ntcb_put8(b, L.deadzone ? 1u : 0u); for (int c = 0; c < L.C; c++) ntcb_put8(b, (uint32_t)L.q[c]); }   // [DCT] hook: the dct_flags byte (version 2) after dc_step
        if (L.mode == 0 || L.mode == 1) for (int c = 0; c < L.C; c++) { ntcb_putf32(b, L.lo[c]); ntcb_putf32(b, L.hi[c]); }
        if (L.mode == 2) for (int c = 0; c < L.C; c++) for (float v : L.palette[c]) ntcb_putf32(b, v);
    }
    if (b.size() > 65535) return false;   // H is a 16-bit field
    b[6] = (uint8_t)(b.size() & 0xFFu); b[7] = (uint8_t)((b.size() >> 8) & 0xFFu);
    return true;
}
// [NTCB] The fewest bytes the sections can occupy given the header (64-bit; the DCT token stream's floor is DC + EOB per block):
// [NTCB] the readers refuse a header that promises more than the file holds before anything is allocated from its fields.
static uint64_t ntcb_min_section_bytes(const NtcbHeader& h) {
    uint64_t total = 0;
    for (const NtcbLevel& L : h.lv) {
        uint64_t bits = 2 * NTCB_SYNC_BITS;
        if (L.mode == 3) {
            const uint64_t nblk = ((uint64_t)L.W / DCT_N) * ((uint64_t)L.H / DCT_N);
            int dc_raw_bits = 0; { const int lvl = 512 / L.dc_step + 1; while ((1 << dc_raw_bits) < lvl) dc_raw_bits++; }
            bits += (uint64_t)L.C * (NTCB_SYNC_BITS + nblk * (uint64_t)(4 + dc_raw_bits + 7));
        } else for (int c = 0; c < L.C; c++) bits += (uint64_t)L.W * (uint64_t)L.H * (uint64_t)L.bits[c];
        total += 8 + (bits + 7) / 8;
    }
    uint64_t nparams = 0; { int prev = h.nin; for (int w : h.hidden) { nparams += (uint64_t)w * prev + w; prev = w; } nparams += (uint64_t)h.nout * prev + h.nout; }
    total += 8 + (16 * nparams + 2 * NTCB_SYNC_BITS + 7) / 8;
    return total;
}
// [NTCB] Parse and validate the header (every range), the file length, the file hash and the minimum section size. `err` names the first problem.
static bool ntcb_read_header(const std::vector<uint8_t>& buf, NtcbHeader& h, std::string& err) {
    const size_t n = buf.size();
    if (n < 8) { err = "not an ntcb file (shorter than 8 bytes)"; return false; }
    NtcbByteReader r(buf.data(), n);
    if (r.u32() != NTCB_MAGIC) { err = "not an ntcb file (bad magic)"; return false; }
    h.version = (int)r.u16(); h.H = (int)r.u16();
    if (h.version != 1 && h.version != NTCB_VERSION) { err = "ntcb version " + std::to_string(h.version) + " (this build reads versions 1 and " + std::to_string(NTCB_VERSION) + ")"; return false; }   // [DCT] hook: version 1 (no dct_flags byte, dead zone implied) and 2
    if (h.H < NTCB_FIXED_HEADER || (size_t)h.H > n) { err = "header length " + std::to_string(h.H) + " exceeds the file size " + std::to_string(n) + " (truncated or corrupt)"; return false; }
    r.n = (size_t)h.H;   // the fixed fields and the records must lie inside the header
    h.img_w = (int)r.u32(); h.img_h = (int)r.u32(); h.src_w = (int)r.u32(); h.src_h = (int)r.u32();
    h.file_bytes = r.u32(); h.hash = r.u32();
    if (h.file_bytes != n) { err = "file_bytes " + std::to_string(h.file_bytes) + " != actual size " + std::to_string(n) + " (truncated or extended)"; return false; }
    if (ntcb_file_hash(buf.data(), n) != h.hash) { err = "file hash mismatch (corrupt file)"; return false; }
    h.latent_cell = (int)r.u8(); h.dct_block = (int)r.u8(); h.T = (int)r.u8(); h.nlevels = (int)r.u8(); h.act = (int)r.u8(); h.clamp = (int)r.u8(); h.nhidden = (int)r.u8();
    const int reserved = (int)r.u8();
    h.leak = r.f32(); h.nin = (int)r.u32(); h.nout = (int)r.u32();
    if (!r.ok) { err = "truncated header"; return false; }
    if (h.img_w < 1 || h.img_h < 1 || h.src_w < 1 || h.src_h < 1 || h.src_w > h.img_w || h.src_h > h.img_h || h.img_w > 16384 || h.img_h > 16384) { err = "bad image size fields (decode size at most 16384 x 16384, source size within it)"; return false; }
    if (h.T < 1 || h.T > MAXT || h.nlevels < 1 || h.nlevels > 3 || h.act != 0 || (h.clamp != 0 && h.clamp != 1) || h.nhidden < 1 || h.nhidden > MAXL || reserved != 0)
        { err = "header field out of range (textures 1..4, 1..3 levels, activation 0, clamp 0/1, 1..8 hidden layers, reserved byte 0)"; return false; }
    if (!ntcb_finite32(h.leak) || h.nin < 1 || h.nin > MAXH || h.nout < 1 || h.nout > MAXH) { err = "bad leak, nin or nout"; return false; }
    h.cpt.resize(h.T); int chan = 0;
    for (int t = 0; t < h.T; t++) { h.cpt[t] = (int)r.u8(); chan += h.cpt[t]; if (r.ok && h.cpt[t] != 3) { err = "texture " + std::to_string(t) + " has " + std::to_string(h.cpt[t]) + " channels; this build decodes 3 per texture"; return false; } }
    if (!r.ok) { err = "truncated header (channel counts)"; return false; }
    if (chan != h.nout) { err = "nout " + std::to_string(h.nout) + " != the sum of the per-texture channel counts " + std::to_string(chan); return false; }
    h.hidden.resize(h.nhidden);
    for (int l = 0; l < h.nhidden; l++) { h.hidden[l] = (int)r.u16(); if (r.ok && (h.hidden[l] < 1 || h.hidden[l] > MAXH)) { err = "bad hidden width"; return false; } }
    const int poslen = (int)r.u16();
    if (!r.ok || poslen >= 4096) { err = "bad positional spec length"; return false; }
    h.pos.resize(poslen); for (int i = 0; i < poslen; i++) h.pos[i] = (char)r.u8();
    if (!r.ok) { err = "truncated header (positional spec)"; return false; }
    h.lv.assign(h.nlevels, NtcbLevel());
    int chsum = 0;
    for (int l = 0; l < h.nlevels; l++) {
        NtcbLevel& L = h.lv[l]; const std::string lv = "level " + std::to_string(l) + ": ";
        L.W = (int)r.u32(); L.H = (int)r.u32(); L.C = (int)r.u16(); L.filter = (int)r.u8(); L.mode = (int)r.u8();
        if (!r.ok) { err = lv + "truncated level record"; return false; }
        if (L.W < 1 || L.H < 1 || L.W > (1 << 20) || L.H > (1 << 20) || L.C < 1 || L.C > MAXH || (L.filter != 0 && L.filter != 1) || L.mode < 0 || L.mode > 3) { err = lv + "bad dims (" + std::to_string(L.W) + "x" + std::to_string(L.H) + "x" + std::to_string(L.C) + "), filter flag (" + std::to_string(L.filter) + ") or mode (" + std::to_string(L.mode) + ")"; return false; }
        if (l > 0 && L.mode > 1) { err = lv + "mode " + std::to_string(L.mode) + " (qat / dct) is only defined on level 0"; return false; }
        L.bits.resize(L.C); for (int c = 0; c < L.C; c++) L.bits[c] = (int)r.u8();
        if (!r.ok) { err = lv + "truncated level record"; return false; }
        for (int c = 0; c < L.C; c++) {
            const int B = L.bits[c];
            const bool okb = L.mode == 3 ? B == 0 : (L.mode == 2 ? (B >= 1 && B <= 8) : (B >= 2 && B <= 12));
            if (!okb) { err = lv + "bit depth " + std::to_string(B) + " out of range for mode " + std::to_string(L.mode); return false; }
            if (L.mode <= 1 && B != L.bits[0]) { err = lv + "per-channel bit depths on a grid level are not supported by this trainer"; return false; }
        }
        if (L.mode == 3) {
            L.dc_step = (int)r.u8();   // [DCT] hook: was `L.dc_step = (int)r.u8(); L.q.resize(L.C); for (...) L.q[c] = (int)r.u8();` on one line
            const int flags = h.version >= 2 ? (int)r.u8() : 1;   // [DCT] version 2: dct_flags after dc_step (bit 0 = dead zone); version 1: dead zone implied
            L.q.resize(L.C); for (int c = 0; c < L.C; c++) L.q[c] = (int)r.u8();   // [DCT] hook: was on the dc_step line
            if (!r.ok) { err = lv + "truncated level record"; return false; }
            if (L.dc_step < 1 || L.dc_step > 64) { err = lv + "dc_step " + std::to_string(L.dc_step) + " out of range (1..64)"; return false; }
            if (flags & ~1) { err = lv + "dct_flags " + std::to_string(flags) + " has reserved bits set (only bit 0, the dead zone, is defined)"; return false; }   // [DCT]
            L.deadzone = flags & 1;   // [DCT]
            for (int c = 0; c < L.C; c++) if (L.q[c] < 1 || L.q[c] > 100) { err = lv + "dct quality " + std::to_string(L.q[c]) + " out of range (1..100)"; return false; }
            if (h.dct_block != DCT_N || L.filter != 1 || L.W != h.img_w || L.H != h.img_h || L.W % DCT_N != 0 || L.H % DCT_N != 0 || L.C > MAX_DCT_CH)
                { err = lv + "a dct level must be a full-resolution nearest level with a size that is a multiple of 8, dct_block 8 and at most 4 channels"; return false; }
        } else if (l == 0 && h.dct_block != 0) { err = "dct_block must be 0 when level 0 is not DCT-coded"; return false; }
        if (L.mode == 2 && L.filter != 1) { err = lv + "a qat level must be nearest"; return false; }
        if (L.mode <= 1) {
            L.lo.resize(L.C); L.hi.resize(L.C);
            for (int c = 0; c < L.C; c++) { L.lo[c] = r.f32(); L.hi[c] = r.f32(); }
            if (!r.ok) { err = lv + "truncated grid ranges"; return false; }
            for (int c = 0; c < L.C; c++) if (!ntcb_finite32(L.lo[c]) || !ntcb_finite32(L.hi[c]) || L.lo[c] > L.hi[c]) { err = lv + "bad grid range"; return false; }
        }
        if (L.mode == 2) {
            L.palette.resize(L.C);
            for (int c = 0; c < L.C; c++) { L.palette[c].resize((size_t)1 << L.bits[c]); for (float& v : L.palette[c]) { v = r.f32(); if (r.ok && !ntcb_finite32(v)) { err = lv + "non-finite palette value"; return false; } } }
            if (!r.ok) { err = lv + "truncated palette"; return false; }
        }
        chsum += L.C;
    }
    if (r.pos != (size_t)h.H) { err = "header length " + std::to_string(h.H) + " disagrees with its fields (" + std::to_string(r.pos) + " bytes)"; return false; }
    { PosEnc pe; if (!pe.parse(h.pos)) { err = "positional spec \"" + h.pos + "\" is not one this build reads"; return false; }
      if (chsum + pe.count() != h.nin) { err = "nin " + std::to_string(h.nin) + " != latent channels " + std::to_string(chsum) + " + positional inputs " + std::to_string(pe.count()); return false; } }
    { const uint64_t need = ntcb_min_section_bytes(h), have = (uint64_t)n - (uint64_t)h.H;
      if (need > have) { err = "the header's sections need at least " + std::to_string(need) + " bytes but the file has " + std::to_string(have) + " after the header (truncated or corrupt)"; return false; } }
    return true;
}
static void ntcb_put_section(std::vector<uint8_t>& file, int kind, int level, const std::vector<uint8_t>& body) {   // [NTCB] 8-byte section header (1.2), coding 0
    ntcb_put8(file, (uint32_t)kind); ntcb_put8(file, 0); ntcb_put8(file, (uint32_t)level); ntcb_put8(file, 0); ntcb_put32(file, (uint32_t)body.size());
    file.insert(file.end(), body.begin(), body.end());
}

struct NtcbReport {   // [NTCB] the terms of the file: line
    size_t file_bytes = 0; int H = 0, nsections = 0;
    double content_bits = 0, pad_bits = 0, sync_bits = 0, sim_total = 0, sim_side = 0;
    bool match = false;
};
static bool ntcb_restore(Decoder& D, const NtcbHeader& h, const std::vector<uint8_t>& buf, std::vector<QesLevel>& q8, std::string& err);   // [NTCB] below; the writer's self-check re-reads through it
// [NTCB] Write the container from the trainer's state. Modes: level 0 dct (D.dct.live) / qat (D.qat_ch) ; any level with a live
// [NTCB] --qes grid (mode 0 again when that grid came from a container) ; otherwise mode 0 at bitrate_stats' post-hoc depth with the
// [NTCB] shadow's per-channel min / max (L1955). `zpost` is bitrate_stats' dequantized latent (the values behind recon_q_final.png):
// [NTCB] every grid value the readers will reconstruct (strict FP) must equal it. Refuses (false, message printed) instead of folding
// [NTCB] anything: a --dct-q run whose switch never happened, |q| > 256, a DC symbol outside its unsigned field, a value off its grid,
// [NTCB] a weight that is not an fp16 value (the rounding hook did not run), and a file whose re-read differs from the written state.
static bool ntcb_write(const std::string& path, const Decoder& D, const BitrateStats& bs, const Options& o, const float* zpost, NtcbReport& R) {
    const LatentSet& LS = D.lat;
    if (o.dct_q > 0 && !D.dct.live) { printf("ntcb     : level 0 is not DCT-coded yet: nothing to write; raise --dct-start or --iters\n"); return false; }
    if (o.block > 255) { printf("ntcb     : --block %d does not fit the container's 8-bit latent_cell field (at most 255); nothing written\n", o.block); return false; }
    NtcbHeader h;
    h.img_w = D.W; h.img_h = D.H; h.src_w = o.orig_w; h.src_h = o.orig_h;
    h.latent_cell = o.block; h.dct_block = D.dct.live ? D.dct.N : 0;
    h.T = D.mlp.nout / 3; h.cpt.assign(h.T, 3); h.nlevels = (int)LS.lv.size(); h.act = 0; h.clamp = o.clamp_out ? 1 : 0;
    h.nhidden = (int)D.mlp.hidden.size(); h.hidden = D.mlp.hidden; h.leak = D.mlp.leak; h.nin = D.mlp.nin; h.nout = D.mlp.nout; h.pos = D.pos.spec;
    h.lv.assign(h.nlevels, NtcbLevel());
    for (int l = 0; l < h.nlevels; l++) {
        const Latent& L = LS.lv[l]; NtcbLevel& N = h.lv[l];
        N.W = L.W; N.H = L.H; N.C = L.C; N.filter = L.nearest ? 1 : 0;
        const bool fixed = (l == 0 && !D.qat_ch.empty());
        const bool qes = D.qes_live && D.qes[l].fitted();
        const int lbits = (!fixed && !qes && D.qes_on && D.qes[l].bits > 0) ? D.qes[l].bits : o.qbits;   // bitrate_stats L1935
        if (l == 0 && D.dct.live) { N.mode = 3; N.bits.assign(L.C, 0); N.dc_step = D.dct.dc_step; N.q = D.dct.q; N.deadzone = D.dct.deadzone ? 1 : 0; }   // [DCT] hook: N.deadzone
        else if (fixed) {
            N.mode = 2; N.bits = D.qat_ch; N.palette.resize(L.C);
            for (int c = 0; c < L.C; c++) { N.palette[c].resize((size_t)1 << N.bits[c]); for (int k = 0; k <= qat_levels(N.bits[c]); k++) N.palette[c][k] = qat_value(k, N.bits[c]); }
        } else if (qes) { N.mode = D.qes[l].from_ntcb ? 0 : 1; N.bits.assign(L.C, D.qes[l].bits); N.lo = D.qes[l].lo; N.hi = D.qes[l].hi; }   // a grid restored from a container keeps its mode-0 label
        else {
            if (lbits < 2 || lbits > 12) { printf("ntcb     : level %d would be stored at %d bits; the container takes 2..12 (--qbits)\n", l, lbits); return false; }
            N.mode = 0; N.bits.assign(L.C, lbits); N.lo.assign(L.C, 1e30f); N.hi.assign(L.C, -1e30f);
            const float* z = LS.z.data() + L.off;
            for (size_t i = 0; i < L.size(); i++) { const int c = (int)(i % L.C); N.lo[c] = std::min(N.lo[c], z[i]); N.hi[c] = std::max(N.hi[c], z[i]); }
        }
    }
    std::vector<uint8_t> file;
    if (!ntcb_write_header(h, file)) { printf("ntcb     : header larger than 65535 bytes; nothing written\n"); return false; }
    const int H = (int)file.size();
    double content = 0, pad = 0, sync = 0; int nsec = 0;
    for (int l = 0; l < h.nlevels; l++) {
        NtcbBitWriter w;
        const Latent& L = LS.lv[l]; const NtcbLevel& N = h.lv[l];
        const float* z = LS.z.data() + L.off; const float* zd = D.zdec() + L.off; const float* zp = zpost + L.off;
        w.marker(ntcb_sync_value(NTCB_SYNC_LEVEL, l, 0));
        if (N.mode == 3) {   // the pass-2 token loop of dct_analyze (L956-978) without the entropy terms; keep the two in step
            const DctLevel& Q = D.dct; const size_t nb = Q.nblk();
            for (int c = 0; c < Q.C; c++) {
                w.marker(ntcb_sync_value(NTCB_SYNC_CHANNEL, l, c));
                for (size_t b = 0; b < nb; b++) w.put(Q.code[b * Q.C + c], 4);
                for (size_t b = 0; b < nb; b++) {
                    const int16_t* sym = Q.symbols(b, c);
                    if (sym[0] < 0 || sym[0] >= (1 << Q.dc_raw_bits)) { printf("ntcb     : DC symbol %d out of range for %d unsigned bits (block %zu, channel %d): writer bug\n", (int)sym[0], Q.dc_raw_bits, b, c); return false; }
                    w.put((uint32_t)sym[0], Q.dc_raw_bits);
                    int run = 0, lnz = 0;
                    for (int p = 1; p < DCT_NN; p++) {
                        const int q = sym[DCT_ZIGZAG[p]];
                        if (q == 0) { run++; continue; }
                        const int mag = std::abs(q);
                        if (mag > 256) { printf("ntcb     : |q| = %d > 256 at block %zu, channel %d (the simulator folds it into the top bin as nmag_over); nothing written\n", mag, b, c); return false; }
                        w.put((uint32_t)run, 7); w.put((uint32_t)(mag - 1), 8); w.put(q < 0 ? 1u : 0u, 1);
                        lnz = p; run = 0;
                    }
                    if (lnz < DCT_NN - 1) w.put((uint32_t)DCT_EOB, 7);
                }
            }
        } else {
            for (int c = 0; c < L.C; c++) {
                const int bits = N.bits[c], levels = (1 << bits) - 1;
                const float range = N.mode == 0 ? std::max(N.hi[c] - N.lo[c], 1e-6f) : 0.0f;   // bitrate_stats L1956
                for (size_t i = c; i < L.size(); i += L.C) {
                    int k;
                    if (N.mode == 2) { k = qat_index(z[i], bits); if (qat_value(k, bits) != z[i]) { printf("ntcb     : level-0 value %g is off its --qat grid (channel %d): writer bug\n", z[i], c); return false; } }
                    else if (N.mode == 1) { k = qes_index(z[i], D.qes[l].lo[c], D.qes[l].range[c], levels); if (D.qes[l].grid[(size_t)c * (levels + 1) + k] != zd[i]) { printf("ntcb     : level %d --qes value %g is not its grid point (channel %d): writer bug\n", l, zd[i], c); return false; } }
                    else {
                        if (D.qes_live && D.qes[l].fitted()) k = qes_index(z[i], D.qes[l].lo[c], D.qes[l].range[c], levels);   // a container's grid, reinstalled: the values sit on it
                        else k = q8_index(z[i], N.lo[c], range, levels);   // bitrate_stats' index, strict
                        if (q8_value(k, N.lo[c], range, levels) != zp[i]) { printf("mode-0 level %d texel %zu: file value != decoded value (fast-math drift): writer refuses\n", l, i / L.C); return false; }
                    }
                    w.put((uint32_t)k, bits);
                }
            }
        }
        w.marker(ntcb_sync_value(NTCB_SYNC_END, l, 0));
        content += w.bits - w.sync; sync += w.sync; w.align(); pad += (double)w.out.size() * 8.0 - w.bits;
        ntcb_put_section(file, 1, l, w.out); nsec++;
    }
    {   // the MLP: fp16 in D.mlp.p order, already fp16 values (decision 2)
        NtcbBitWriter w;
        w.marker(ntcb_sync_value(NTCB_SYNC_MLP, h.nlevels, 0));
        for (size_t i = 0; i < D.mlp.p.size(); i++) {
            bool ov = false; const uint16_t hh = ntcb_f32_to_f16(D.mlp.p[i], &ov);
            if (ov || !ntcb_finite32(D.mlp.p[i])) { printf("ntcb     : weight %zu = %g is not representable in fp16; nothing written\n", i, D.mlp.p[i]); return false; }
            if (ntcb_f16_to_f32(hh) != D.mlp.p[i]) { printf("ntcb     : weight %zu = %.9g is not an fp16 value (the rounding hook did not run): writer bug\n", i, D.mlp.p[i]); return false; }
            w.put(hh, 16);
        }
        w.marker(ntcb_sync_value(NTCB_SYNC_END, h.nlevels, 0));
        content += w.bits - w.sync; sync += w.sync; w.align(); pad += (double)w.out.size() * 8.0 - w.bits;
        ntcb_put_section(file, 2, 0, w.out); nsec++;
    }
    ntcb_patch32(file, 24, (uint32_t)file.size());
    ntcb_patch32(file, 28, 0u);
    ntcb_patch32(file, 28, ntcb_file_hash(file.data(), file.size()));
    {   // self-check: re-read the bytes through the reader path into a copy of the state and compare with what was written
        Decoder D2 = D; NtcbHeader h2; std::vector<QesLevel> q8; std::string e;
        std::fill(D2.lat.z.begin(), D2.lat.z.end(), 0.0f); std::fill(D2.mlp.p.begin(), D2.mlp.p.end(), 0.0f);
        std::fill(D2.dct.sym.begin(), D2.dct.sym.end(), (int16_t)0); std::fill(D2.dct.code.begin(), D2.dct.code.end(), (uint8_t)0);
        if (!ntcb_read_header(file, h2, e) || !ntcb_restore(D2, h2, file, q8, e)) { printf("ntcb     : self-check: re-read file differs from written state (%s)\n", e.c_str()); return false; }
        bool same = D2.mlp.p.size() == D.mlp.p.size() && memcmp(D2.mlp.p.data(), D.mlp.p.data(), D.mlp.p.size() * sizeof(float)) == 0;
        for (int l = 0; l < h.nlevels && same; l++) {
            const Latent& L = LS.lv[l]; const NtcbLevel& N = h.lv[l], & N2 = h2.lv[l];
            if (N.mode == 3) same = memcmp(D2.dct.sym.data(), D.dct.sym.data(), D.dct.sym.size() * sizeof(int16_t)) == 0 && memcmp(D2.dct.code.data(), D.dct.code.data(), D.dct.code.size()) == 0;
            else same = memcmp(D2.lat.z.data() + L.off, zpost + L.off, L.size() * sizeof(float)) == 0;
            if (N.mode <= 1) same = same && memcmp(N2.lo.data(), N.lo.data(), L.C * sizeof(float)) == 0 && memcmp(N2.hi.data(), N.hi.data(), L.C * sizeof(float)) == 0 && N2.bits == N.bits;
            if (N.mode == 2) same = same && N2.palette == N.palette && N2.bits == N.bits;
            if (N.mode == 0) same = same && q8[l].bits == N.bits[0] && q8[l].lo == N.lo && q8[l].hi == N.hi;
            if (!same) printf("ntcb     : self-check: re-read file differs from written state (level %d)\n", l);
        }
        if (!same) { if (D2.mlp.p != D.mlp.p) printf("ntcb     : self-check: re-read file differs from written state (mlp)\n"); return false; }
    }
    FILE* f = fopen(path.c_str(), "wb");
    if (!f || fwrite(file.data(), 1, file.size(), f) != file.size()) { printf("ntcb     : cannot write %s\n", path.c_str()); if (f) fclose(f); return false; }
    fclose(f);
    R.file_bytes = file.size(); R.H = H; R.nsections = nsec; R.content_bits = content; R.pad_bits = pad; R.sync_bits = sync;
    R.sim_side = bs.bits_header; R.sim_total = bs.bits_raw_latent + bs.bits_header + bs.bits_mlp;
    R.match = content == bs.bits_raw_latent + bs.bits_mlp;   // exact, on the unrounded doubles (integers)
    return true;
}
// [NTCB] The --load path: fill D.lat.z (modes 0 / 1 / 2, dequantized through the strict qes_rebuild grid or the stored palette),
// [NTCB] D.dct.sym / code (mode 3; level 0 of the shadow is zeroed, the caller reconstructs it), D.mlp.p (fp16). q8[l] receives
// [NTCB] the grid of every mode-0 level (bits, lo, hi, rebuilt) so the caller can install it as a frozen --qes level. D.lat, D.dct
// [NTCB] and D.mlp are sized by the caller, whose mismatch check has already matched them against the header. Every sync marker
// [NTCB] is checked and named on a mismatch. False with `err`.
static bool ntcb_restore(Decoder& D, const NtcbHeader& h, const std::vector<uint8_t>& buf, std::vector<QesLevel>& q8, std::string& err) {
    const size_t n = buf.size(); size_t pos = (size_t)h.H;
    q8.assign(h.nlevels, QesLevel());
    auto section = [&](int kind, int level, size_t& body, size_t& len) -> bool {
        if (pos + 8 > n) { err = "truncated at the section header of " + std::string(kind == 1 ? "level " + std::to_string(level) : "the mlp"); return false; }
        NtcbByteReader r(buf.data() + pos, 8);
        const int k = (int)r.u8(), coding = (int)r.u8(), lv = (int)r.u8(), res = (int)r.u8(); len = r.u32();
        if (coding != 0) { err = (coding == 1 || coding == 2) ? "coded sections (coding " + std::to_string(coding) + ") need a newer decoder" : "bad section coding " + std::to_string(coding); return false; }
        if (k != kind || lv != level || res != 0) { err = "unexpected section header (kind " + std::to_string(k) + ", level " + std::to_string(lv) + ", reserved " + std::to_string(res) + ") where " + (kind == 1 ? "level " + std::to_string(level) : std::string("the mlp")) + " was expected"; return false; }
        body = pos + 8;
        if (len > n - body) { err = "section of " + (kind == 1 ? "level " + std::to_string(level) : std::string("the mlp")) + " runs past the end of the file"; return false; }
        return true;
    };
    auto sync = [&](NtcbBitReader& br, uint32_t m, const std::string& where) -> bool {
        if (!br.marker(m)) { err = "sync marker mismatch " + where; return false; }
        return true;
    };
    for (int l = 0; l < h.nlevels; l++) {
        const NtcbLevel& N = h.lv[l]; const Latent& L = D.lat.lv[l]; const std::string lv = "level " + std::to_string(l) + ": ", ls = std::to_string(l);
        if (L.W != N.W || L.H != N.H || L.C != N.C) { err = lv + "internal: the trainer's level does not match the header"; return false; }
        size_t body = 0, len = 0;
        if (!section(1, l, body, len)) return false;
        NtcbBitReader br(buf.data(), body, body + len);
        float* z = D.lat.z.data() + L.off;
        if (!sync(br, ntcb_sync_value(NTCB_SYNC_LEVEL, l, 0), "before level " + ls + " body")) return false;
        if (N.mode == 3) {
            DctLevel& Q = D.dct;
            if (!D.dct.on || Q.C != N.C || Q.BW != N.W / DCT_N || Q.BH != N.H / DCT_N || Q.dc_step != N.dc_step || (Q.deadzone ? 1 : 0) != N.deadzone || Q.sym.size() != Q.nblk() * Q.C * DCT_NN) { err = lv + "internal: the trainer's DCT state does not match the header"; return false; }   // [DCT] hook: the deadzone clause
            const size_t nb = Q.nblk();
            for (int c = 0; c < Q.C; c++) {
                if (!sync(br, ntcb_sync_value(NTCB_SYNC_CHANNEL, l, c), "before level " + ls + " channel " + std::to_string(c))) return false;
                for (size_t b = 0; b < nb; b++) Q.code[b * Q.C + c] = (uint8_t)br.get(4);
                for (size_t b = 0; b < nb && br.ok; b++) {
                    int16_t* sym = Q.symbols(b, c);
                    for (int k = 0; k < DCT_NN; k++) sym[k] = 0;
                    sym[0] = (int16_t)br.get(Q.dc_raw_bits);
                    int p = 1;
                    while (p < DCT_NN && br.ok) {
                        const int run = (int)br.get(7);
                        if (!br.ok || run == DCT_EOB) break;
                        if (run > 62) { err = lv + "bad run symbol " + std::to_string(run) + " at block " + std::to_string(b) + " (0..62 or the EOB 64)"; return false; }
                        p += run;
                        if (p >= DCT_NN) { err = lv + "token stream runs past the 63 AC positions at block " + std::to_string(b); return false; }
                        const int mag = (int)br.get(8) + 1, sign = (int)br.get(1);
                        sym[DCT_ZIGZAG[p]] = (int16_t)(sign ? -mag : mag);
                        p++;
                    }
                }
            }
            if (!br.ok) { err = lv + "truncated token stream"; return false; }
            std::fill(z, z + L.size(), 0.0f);
        } else {
            QesLevel Q;
            if (N.mode <= 1) { Q.bits = N.bits[0]; Q.lo = N.lo; Q.hi = N.hi; qes_rebuild(Q, L.C); }
            for (int c = 0; c < L.C; c++) {
                const int bits = N.bits[c], levels = (1 << bits) - 1;
                for (size_t i = c; i < L.size() && br.ok; i += L.C) {
                    const uint32_t k = br.get(bits);
                    z[i] = N.mode == 2 ? N.palette[c][k] : Q.grid[(size_t)c * (levels + 1) + k];
                }
            }
            if (!br.ok) { err = lv + "truncated grid indices"; return false; }
            if (N.mode == 0) { Q.from_ntcb = true; q8[l] = Q; }
        }
        if (!sync(br, ntcb_sync_value(NTCB_SYNC_END, l, 0), "after level " + ls + " body")) return false;
        br.align();
        if (br.pos != body + len) { err = lv + "section length " + std::to_string(len) + " != the " + std::to_string(br.pos - body) + " bytes its content occupies"; return false; }
        pos = body + len;
    }
    {
        size_t body = 0, len = 0;
        if (!section(2, 0, body, len)) return false;
        if (len != (D.mlp.p.size() * 16 + 2 * NTCB_SYNC_BITS + 7) / 8) { err = "mlp section length " + std::to_string(len) + " != the bytes of " + std::to_string(D.mlp.p.size()) + " fp16 weights and two sync markers"; return false; }
        NtcbBitReader br(buf.data(), body, body + len);
        if (!sync(br, ntcb_sync_value(NTCB_SYNC_MLP, h.nlevels, 0), "before the mlp body")) return false;
        for (size_t i = 0; i < D.mlp.p.size(); i++) { const uint16_t hh = (uint16_t)br.get(16); D.mlp.p[i] = ntcb_f16_to_f32(hh); if (br.ok && !ntcb_finite16(hh)) { err = "non-finite mlp weight " + std::to_string(i); return false; } }
        if (!br.ok) { err = "truncated mlp section"; return false; }
        if (!sync(br, ntcb_sync_value(NTCB_SYNC_END, h.nlevels, 0), "after the mlp body")) return false;
        br.align();
        if (br.pos != body + len) { err = "mlp section length " + std::to_string(len) + " != the " + std::to_string(br.pos - body) + " bytes its content occupies"; return false; }
        pos = body + len;
    }
    if (pos != n) { err = "trailing bytes after the mlp section (" + std::to_string(n - pos) + ")"; return false; }
    return true;
}
// ---- end of the NTCB region [NTCB]
// ---------------------------------------------------------------- main
// --eval-filter: how good is "interpolate the inputs, decode once" as a texture filter?
// For each sub-texel offset the image is resampled at pixel centers shifted by
// (ox, oy) texels in three ways: (ref) bilinear filtering of the decoded texel image,
// the reference a hardware filter would produce from a decoded texture; (interp) one
// decoder evaluation per sample with the level-0 (selector) values bilinearly
// interpolated between the four nearest texels, the block latent taken from the block
// that contains the sample, and the position inputs from the shifted UV; (target) the
// source image filtered the same way, for context. Samples whose four level-0 taps lie
// in one level-1 block are "interior"; the others straddle a block edge, where blending
// selectors of one block with the latent of another is the questionable case.
static void eval_filter(const Decoder& D, const Image& target, const Image& recon, const Options& o, int T) {
    const std::string dir = o.outdir + "/filter";
#ifdef _WIN32
    _mkdir(dir.c_str());
#else
    mkdir(dir.c_str(), 0755);
#endif
    const float offs[][2] = { { 0.5f, 0.0f }, { 0.0f, 0.5f }, { 0.5f, 0.5f }, { 0.25f, 0.25f }, { 0.75f, 0.5f } };
    const int W = D.W, H = D.H, nc = D.mlp.nout;
    const Latent& L0 = D.lat.lv[0];
    Latent L0b = L0; L0b.nearest = false;   // level 0 sampled bilinearly (texel centers at (i+0.5)/W0)
    const int bw = D.lat.lv.size() > 1 ? std::max(1, W / D.lat.lv[1].W) : W, bh = D.lat.lv.size() > 1 ? std::max(1, H / D.lat.lv[1].H) : H;
    auto bilin = [&](const Image& img, float u, float v, float* out) {   // bilinear at UV, texel centers at (i+0.5)/W
        float x = u * img.w - 0.5f, y = v * img.h - 0.5f;
        int x0 = (int)std::floor(x), y0 = (int)std::floor(y);
        float fx = x - x0, fy = y - y0;
        int xa = std::max(0, std::min(img.w - 1, x0)), xb = std::max(0, std::min(img.w - 1, x0 + 1));
        int ya = std::max(0, std::min(img.h - 1, y0)), yb = std::max(0, std::min(img.h - 1, y0 + 1));
        for (int c = 0; c < img.nc; c++)
            out[c] = (1 - fx) * (1 - fy) * img.rgb[((size_t)ya * img.w + xa) * img.nc + c] + fx * (1 - fy) * img.rgb[((size_t)ya * img.w + xb) * img.nc + c]
                   + (1 - fx) * fy * img.rgb[((size_t)yb * img.w + xa) * img.nc + c] + fx * fy * img.rgb[((size_t)yb * img.w + xb) * img.nc + c];
    };
    printf("eval-filter: decode with interpolated selector inputs (one evaluation per sample) vs bilinear filtering of decoded texels; %dx%d blocks\n", bw, bh);
    printf("  offset       | interp vs ref: all / interior / edge (dB) | vs target: ref / interp (dB)%s\n", T > 1 ? "  [per-texture interp vs ref]" : "");
    for (size_t k = 0; k < sizeof(offs) / sizeof(offs[0]); k++) {
        const float ox = offs[k][0], oy = offs[k][1];
        Image ref, itp, tgt;
        ref.w = itp.w = tgt.w = W; ref.h = itp.h = tgt.h = H; ref.nc = itp.nc = tgt.nc = nc;
        ref.rgb.resize((size_t)W * H * nc); itp.rgb.resize(ref.rgb.size()); tgt.rgb.resize(ref.rgb.size());
        std::vector<unsigned char> interior((size_t)W * H);
#pragma omp parallel for schedule(static)
        for (int py = 0; py < H; py++)
            for (int px = 0; px < W; px++) {
                const float u = (px + 0.5f + ox) / W, v = (py + 0.5f + oy) / H;
                const size_t pi = (size_t)py * W + px;
                bilin(recon, u, v, &ref.rgb[pi * nc]);
                bilin(target, u, v, &tgt.rgb[pi * nc]);
                float f[MAXH];
                BilinearTap t = bilinear_tap(L0b, u, v);
                sample_latent(L0b, D.zdec(), t, f);
                interior[pi] = (t.x0 / bw == t.x1 / bw) && (t.y0 / bh == t.y1 / bh);
                int kf = L0.C;
                BilinearTap t1;
                for (size_t l = 1; l < D.lat.lv.size(); l++) {
                    const Latent& L = D.lat.lv[l];
                    BilinearTap tl = bilinear_tap(L, u, v);
                    if (l == 1) t1 = tl;
                    sample_latent(L, D.zdec() + L.off, tl, f + kf);
                    kf += L.C;
                }
                BilinearTap t0 = bilinear_tap(L0, u, v);   // level 0's own (nearest) tap for the position kinds
                D.pos.encode(u, v, t0, D.lat.lv.size() > 1 ? &t1 : nullptr, f + kf);
                mlp_forward(D.mlp, D.mlp.p.data(), f, &itp.rgb[pi * nc], D.opt->clamp_out);
            }
        double se_all = 0, se_in = 0, se_ed = 0, se_ref_t = 0, se_itp_t = 0; size_t n_in = 0, n_ed = 0;
        std::vector<double> se_tex(T, 0.0);
        for (size_t pi = 0; pi < (size_t)W * H; pi++) {
            double e = 0;
            for (int c = 0; c < nc; c++) {
                double d = itp.rgb[pi * nc + c] - ref.rgb[pi * nc + c]; e += d * d; se_tex[c / 3] += d * d;
                double dr = ref.rgb[pi * nc + c] - tgt.rgb[pi * nc + c]; se_ref_t += dr * dr;
                double di = itp.rgb[pi * nc + c] - tgt.rgb[pi * nc + c]; se_itp_t += di * di;
            }
            se_all += e;
            if (interior[pi]) { se_in += e; n_in++; } else { se_ed += e; n_ed++; }
        }
        const double npx = (double)W * H;
        auto ps = [](double se, double n) { return n > 0 && se > 0 ? 10.0 * std::log10(n / se) : 99.0; };
        printf("  (%.2f, %.2f) | %6.2f / %6.2f / %6.2f (%5.1f%% edge) | %6.2f / %6.2f", ox, oy,
            ps(se_all, npx * nc), ps(se_in, (double)n_in * nc), ps(se_ed, (double)n_ed * nc), 100.0 * n_ed / npx,
            ps(se_ref_t, npx * nc), ps(se_itp_t, npx * nc));
        if (T > 1) { printf("  ["); for (int t = 0; t < T; t++) printf("%s%.2f", t ? " " : "", ps(se_tex[t], npx * 3)); printf("]"); }
        printf("\n");
        char base[160];
        snprintf(base, sizeof(base), "%s/offset_%02d_x%.2f_y%.2f", dir.c_str(), (int)k, ox, oy);
        for (int t = 0; t < T; t++) {
            std::string sfx = T > 1 ? "_t" + std::to_string(t) : std::string();
            save_png(std::string(base) + "_ref" + sfx + ".png", ref, t);
            save_png(std::string(base) + "_interp" + sfx + ".png", itp, t);
            save_png(std::string(base) + "_target" + sfx + ".png", tgt, t);
        }
    }
    printf("eval-filter: PNGs written to %s/\n", dir.c_str());
}

#ifdef NTC_CUDA
// [DCT] The device description of a live DCT level: the geometry and, for --dct-lambda, the two part flags, rate_scale (the
// [DCT] ES rate weight of DCT_RATE_PLAN.md 2.1) and lam_t16[16] as host-computed float bit patterns (never recomputed there).
static ntc_cuda::DctDesc dct_desc_of(const DctLevel& Q, int W, int H) {   // [DCT]
    ntc_cuda::DctDesc d; memset(&d, 0, sizeof(d));   // [DCT]
    d.live = 1; d.C = Q.C; d.BW = Q.BW; d.BH = Q.BH; d.N = Q.N; d.dc_step = Q.dc_step;   // [DCT]
    d.rate_es = Q.rate_es ? 1 : 0; d.rate_trunc = Q.rate_trunc ? 1 : 0;   // [DCT]
    d.rate_scale = Q.rate_es ? (float)((double)Q.lambda / (16.0 * 65025.0 * (double)W * H)) : 0.0f;   // [DCT] = LatentTrainer::step's rate_scale
    for (int k = 0; k < 16; k++) d.lam_t16[k] = Q.lam_t16[k];   // [DCT]
    d.lo_w16 = Q.lo_w16;   // [DCT] --dct-lambda-lo
    d.deadzone = Q.deadzone ? 1 : 0;   // [DCT] --dct-deadzone
    return d;   // [DCT]
}   // [DCT]
// [DCT] The host replica of the --dct-lambda rate pass for check 4 (= dct_rate_pass with the hash noise of step 0, pair k, regenerated
// [DCT] here as the replica regenerates its mse noise): adds the rate term of every pair to gc, leaves pair K-1's dbits in hdb and the
// [DCT] hit count in hhits for the `dct rate` line. The shadow is D.lat.z (the lr-0 step above left it unchanged).
static void cuda_check_rate_replica(Decoder& D, const Options& o, int K, std::vector<float>& gc, std::vector<int32_t>& hdb, size_t& hhits) {   // [DCT]
    const DctLevel& Q = D.dct; const Latent& L0 = D.lat.lv[0];   // [DCT]
    const float rate_scale = (float)((double)Q.lambda / (16.0 * 65025.0 * (double)D.W * D.H));   // [DCT]
    const float sg = o.lat_sigma, scale = 1.0f / (2.0f * K * sg);   // [DCT]
    std::vector<float> e0(L0.size()); hdb.assign(Q.nblk() * (size_t)Q.C, 0); hhits = 0;   // [DCT]
    for (int k = 0; k < K; k++) {   // [DCT]
        for (size_t i = 0; i < L0.size(); i++) e0[i] = ntc_gauss(o.seed, NS_LAT, 0, k, L0.off + i);   // [DCT]
        const long long nb = (long long)Q.nblk();   // [DCT]
#pragma omp parallel for schedule(static)   // [DCT]
        for (long long b = 0; b < nb; b++)   // [DCT]
            for (int c = 0; c < Q.C; c++) hdb[(size_t)b * Q.C + c] = dct_rate_block(Q, L0, (size_t)b, c, D.lat.z.data() + L0.off, e0.data(), sg, 1.0f) - dct_rate_block(Q, L0, (size_t)b, c, D.lat.z.data() + L0.off, e0.data(), sg, -1.0f);   // [DCT]
        for (size_t i = 0; i < hdb.size(); i++) if (hdb[i] != 0) hhits++;   // [DCT]
        for (int ty = 0; ty < L0.H; ty++)   // [DCT]
            for (int tx = 0; tx < L0.W; tx++) {   // [DCT]
                const size_t b = (size_t)(ty / 8) * Q.BW + (size_t)(tx / 8), t = ((size_t)ty * L0.W + tx) * L0.C;   // [DCT]
                for (int c = 0; c < L0.C; c++) gc[L0.off + t + c] += rate_scale * (float)hdb[b * Q.C + c] * scale * e0[t + c];   // [DCT]
            }   // [DCT]
    }   // [DCT]
}   // [DCT]
// --cuda-check: run the deterministic GPU kernels and the CPU code on identical inputs
// and report the differences. The decode and the selector search must agree to float
// rounding; the ES / FD loss differences are compared per pair / per weight against the
// CPU's loss_subset evaluated with the same (hash-generated) perturbations and batch.
static int cuda_check(Decoder& D, const Image& target, const Options& o, ntc_cuda::Trainer& cu) {
    printf("cuda-check: %s\n", cu.banner().c_str());
    const size_t npix = (size_t)D.W * D.H;
    int fails = 0;
    // 1. full decode
    {
        Image a, b;
        D.decode_full(D.mlp.p.data(), D.zdec(), a);
        b.w = D.W; b.h = D.H; b.nc = D.mlp.nout; b.rgb.resize(npix * b.nc);
        cu.decode_full(nullptr, b.rgb.data());
        double mx = 0; for (size_t i = 0; i < a.rgb.size(); i++) mx = std::max(mx, (double)std::fabs(a.rgb[i] - b.rgb[i]));
        bool ok = mx < 1e-5;
        printf("  decode          : max |gpu - cpu| = %.3e  %s\n", mx, ok ? "PASS" : "FAIL"); fails += !ok;
    }
    // 2. MLP ES losses: 8 pairs, lr 0 (weights unchanged), same batch and perturbations on the CPU
    {
        const int N = 8, it = 1;
        cu.mlp_step(it, o.mlp_sigma, 0.0f, N, o.mlp_batch);
        std::vector<int> bidx; cu.debug_last_batch(bidx);
        std::vector<double> dl; cu.debug_last_dl(dl);
        const size_t P = D.mlp.size();
        std::vector<float> pp(P), pm(P);
        double mxrel = 0, dstd = 0;
        for (int i = 0; i < N; i++) dstd += dl[i] * dl[i];
        dstd = std::sqrt(dstd / N);
        for (int i = 0; i < N; i++) {
            for (size_t k = 0; k < P; k++) { float e = ntc_gauss(o.seed, NS_MLP, it, i, k); pp[k] = D.mlp.p[k] + o.mlp_sigma * e; pm[k] = D.mlp.p[k] - o.mlp_sigma * e; }
            double lp = D.loss_subset(pp.data(), D.zdec(), target, bidx), lm = D.loss_subset(pm.data(), D.zdec(), target, bidx);
            mxrel = std::max(mxrel, std::fabs((lp - lm) - dl[i]) / std::max(dstd, 1e-300));
        }
        bool ok = mxrel < 1e-3;
        printf("  mlp ES dl       : max |gpu - cpu| / rms(dl) = %.3e over %d pairs  %s\n", mxrel, N, ok ? "PASS" : "FAIL"); fails += !ok;
    }
    // 3. FD losses: 24 sampled weights against the CPU's central differences on the same batch
    {
        const int it = 2;
        cu.mlp_step_fd(it, o.mlp_fd_h, 0.0f, o.mlp_batch);
        std::vector<int> bidx; cu.debug_last_batch(bidx);
        std::vector<double> dl; cu.debug_last_dl(dl);
        const size_t P = D.mlp.size();
        std::vector<float> pw(D.mlp.p);
        double rms = 0; for (size_t j = 0; j < P; j++) rms += dl[j] * dl[j]; rms = std::sqrt(rms / P);
        double mxrel = 0;
        for (int q = 0; q < 24; q++) {
            size_t j = (size_t)((q * 7919) % P);
            float orig = pw[j], wp = orig + o.mlp_fd_h, wm = orig - o.mlp_fd_h;
            pw[j] = wp; double lp = D.loss_subset(pw.data(), D.zdec(), target, bidx);
            pw[j] = wm; double lm = D.loss_subset(pw.data(), D.zdec(), target, bidx);
            pw[j] = orig;
            mxrel = std::max(mxrel, std::fabs((lp - lm) - dl[j]) / std::max(rms, 1e-300));
        }
        bool ok = mxrel < 1e-2;
        printf("  mlp FD dl       : max |gpu - cpu| / rms(dl) = %.3e over 24 weights  %s\n", mxrel, ok ? "PASS" : "FAIL"); fails += !ok;
    }
    // 4. latent ES gradient: one step with lr 0 on the GPU, the same estimator on the CPU
    //    with the same hash-generated perturbations (nearest levels, decode_err, raster
    //    gather), compared before Adam relative to the gradient's RMS.
    if (!(o.qat > 0 && D.lat.lv.size() == 1)) {
        const int K = o.lat_pairs;
        cu.lat_step(1, o.lat_sigma, 0.0f, K);   // first call: noise step 0
        std::vector<float> gg; cu.debug_last_zgrad(gg);
        const LatentSet& LS = D.lat;
        const float* zb = D.zdec();   // --qes: the pair is built around the snapped point, as LatentTrainer does
        const size_t n = LS.size(), nlev = LS.lv.size();
        std::vector<float> gc(n, 0.0f), zp(n), zm(n), ep, em;
        const float inv_px = 1.0f / (3.0f * D.wsum * D.W * D.H);
        const float sg = o.lat_sigma;
        for (int k = 0; k < K; k++) {
            std::vector<bool> act(nlev);
            for (size_t l = 0; l < nlev; l++) act[l] = !(o.qat > 0 && l == 0);
            for (size_t l = 0; l < nlev; l++) {
                const Latent& L = LS.lv[l];
                for (size_t i = L.off; i < L.off + L.size(); i++) {
                    float e = act[l] ? ntc_gauss(o.seed, NS_LAT, 0, k, i) : 0.0f;
                    zp[i] = zb[i] + sg * e; zm[i] = zb[i] - sg * e;
                }
            }
            D.decode_err(D.mlp.p.data(), zp.data(), target, ep);
            D.decode_err(D.mlp.p.data(), zm.data(), target, em);
            for (size_t l = 0; l < nlev; l++) {
                if (!act[l]) continue;
                const Latent& L = LS.lv[l];
                const float scale = 1.0f / (2.0f * K * sg);
                std::vector<float> dtex((size_t)L.W * L.H, 0.0f);
                for (int py = 0; py < D.H; py++)
                    for (int px = 0; px < D.W; px++) {
                        BilinearTap t = bilinear_tap(L, (px + 0.5f) / D.W, (py + 0.5f) / D.H);
                        float dd = (ep[(size_t)py * D.W + px] - em[(size_t)py * D.W + px]) * inv_px;
                        dtex[(size_t)t.y0 * L.W + t.x0] += dd;
                        if (t.x1 != t.x0) dtex[(size_t)t.y0 * L.W + t.x1] += dd;
                        if (t.y1 != t.y0) { dtex[(size_t)t.y1 * L.W + t.x0] += dd; if (t.x1 != t.x0) dtex[(size_t)t.y1 * L.W + t.x1] += dd; }
                    }
                for (size_t ti = 0; ti < dtex.size(); ti++) {
                    float w = dtex[ti] * scale; size_t base = L.off + ti * L.C;
                    for (int c = 0; c < L.C; c++) gc[base + c] += w * ntc_gauss(o.seed, NS_LAT, 0, k, base + c);
                }
            }
        }
        std::vector<int32_t> hdb; size_t hhits = 0;   // [DCT] the host dbits of pair K-1 and the host hit count of the replica's rate pass
        const bool rate_es = D.dct.live && D.dct.rate_es;   // [DCT]
        if (rate_es) cuda_check_rate_replica(D, o, K, gc, hdb, hhits);   // [DCT] hook: --dct-lambda part A added to the host estimate after the mse loop (one call; the loop's text and codegen are unchanged)
        double rms = 0, mx = 0; size_t cnt = 0;
        for (size_t i = 0; i < n; i++) { rms += (double)gc[i] * gc[i]; cnt++; }
        rms = std::sqrt(rms / std::max<size_t>(cnt, 1));
        for (size_t i = 0; i < n; i++) mx = std::max(mx, (double)std::fabs(gg[i] - gc[i]));
        bool ok = mx < 1e-3 * rms;
        printf("  latent ES grad  : max |gpu - cpu| / rms(grad) = %.3e over %zu values  %s\n", mx / std::max(rms, 1e-300), n, ok ? "PASS" : "FAIL"); fails += !ok;
        if (rate_es) {   // [DCT] 4b. the --dct-lambda figures of the same step: the device's pair K-1 bit differences and hit count against the host replica's,
            // [DCT]     and its snapped-plane proxy bits and truncation counts (k_dct_snap after the lr-0 Adam step re-snapped the unchanged shadow) against a
            // [DCT]     host re-snap of the downloaded shadow (as check 7; a loaded v13 is not re-snapped at load, so the host state alone would differ). Exact integers.
            // [DCT] This re-snap mutates the host state (D.lat.z, D.zq, D.dct.sym / bits / trunc counts) before checks 5-7 run. Benign today: check 5 is
            // [DCT] excluded under --dct-q, check 6 downloads the device shadow afresh, check 7 host-snaps first (host_snap). A check inserted between 4
            // [DCT] and 7 that reads the host symbols or zq must not inherit this state: re-derive it from the device or move this block after it.
            { std::vector<float> zs(n), ptmp(D.mlp.size()); cu.download_model(zs.data(), ptmp.data()); D.lat.z = zs; D.qes_refresh(); }   // [DCT] the shadow is unchanged by the lr-0 step; the host bits / trunc counts now come from the same snap
            const size_t nbc = D.dct.nblk() * (size_t)D.dct.C;   // [DCT]
            std::vector<int32_t> dbits(nbc), dbd(nbc); int dnt = 0, dnz = 0, dh = 0, de = 0;   // [DCT]
            cu.download_dct_rate(dbits.data(), dbd.data(), &dnt, &dnz, &dh, &de);   // [DCT]
            size_t d1 = 0, d2 = 0;   // [DCT]
            for (size_t i = 0; i < nbc; i++) { if (hdb[i] != dbd[i]) d1++; if (D.dct.bits[i] != dbits[i]) d2++; }   // [DCT]
            const bool ok2 = d1 == 0 && d2 == 0 && (size_t)dnt == D.dct.last_truncated && (size_t)dnz == D.dct.last_nz_before && (size_t)dh == hhits && (size_t)de == nbc * (size_t)K;   // [DCT]
            printf("  dct rate        : pair %d bit differences %zu of %zu differ, hits %zu = %d (of %zu = %d); host re-snap of the downloaded shadow vs k_dct_snap: proxy bits %zu of %zu differ, truncated %zu = %d (of %zu = %d nonzero)  %s\n",   // [DCT]
                K - 1, d1, nbc, hhits, dh, nbc * (size_t)K, de, d2, nbc, D.dct.last_truncated, dnt, D.dct.last_nz_before, dnz, ok2 ? "PASS" : "FAIL"); fails += !ok2;   // [DCT]
        }   // [DCT]
    }
    // 5. selector search from the same state
    if (o.qat > 0) {
        std::vector<float> z0 = D.lat.z;
        qat_search(D, target, o.qat_ch);                 // CPU, in place (on the decode buffer; level 0 copied back to the shadow)
        std::vector<float> zc = D.lat.z;
        D.lat.z = z0; D.qes_refresh(); cu.upload_model(D.lat.z.data(), D.mlp.p.data());
        cu.qat_search();
        std::vector<float> zg(D.lat.z.size()), ptmp(D.mlp.size());
        cu.download_model(zg.data(), ptmp.data());
        const size_t n0 = D.lat.lv[0].size();
        size_t diff = 0; for (size_t i = 0; i < n0; i++) if (zc[i] != zg[i]) diff++;
        Image a, b; D.lat.z = zc; D.qes_refresh(); D.decode_full(D.mlp.p.data(), D.zdec(), a);
        D.lat.z = zg; D.qes_refresh(); D.decode_full(D.mlp.p.data(), D.zdec(), b);
        double mc = mse_of(target, a, o.weights, D.wsum).weighted, mg = mse_of(target, b, o.weights, D.wsum).weighted;
        // Per-texel cell losses of both results, on the CPU: a differing texel is a tie
        // (or a rounding-level near tie) unless the GPU's choice is measurably worse.
        const Latent& L0 = D.lat.lv[0];
        std::vector<double> cc((size_t)L0.W * L0.H, 0.0), cg((size_t)L0.W * L0.H, 0.0);
        for (int py = 0; py < D.H; py++)
            for (int px = 0; px < D.W; px++) {
                BilinearTap t = bilinear_tap(L0, (px + 0.5f) / D.W, (py + 0.5f) / D.H);
                size_t ti = (size_t)t.y0 * L0.W + t.x0, pi = (size_t)py * D.W + px;
                const float* tg = &target.rgb[pi * a.nc];
                for (int c = 0; c < a.nc; c++) {
                    float dc = a.rgb[pi * a.nc + c] - tg[c], dg = b.rgb[pi * b.nc + c] - tg[c];
                    cc[ti] += D.cw[c] * (dc * dc); cg[ti] += D.cw[c] * (dg * dg);
                }
            }
        size_t worse = 0, better = 0, texdiff = 0;
        for (size_t ti = 0; ti < cc.size(); ti++) {
            bool d = false; for (int c = 0; c < L0.C; c++) if (zc[ti * L0.C + c] != zg[ti * L0.C + c]) d = true;
            if (!d) continue;
            texdiff++;
            if (cg[ti] > cc[ti] * (1.0 + 1e-4) + 1e-12) worse++;
            if (cc[ti] > cg[ti] * (1.0 + 1e-4) + 1e-12) better++;
        }
        // Rounding flips near-ties both ways; a real bug would be one-sided and change the loss.
        bool ok = worse <= better + better / 2 + 16 && std::fabs(mg - mc) <= mc * 1e-6;
        printf("  qat search      : %zu of %zu values differ in %zu texels: gpu worse %zu, better %zu, ties %zu; loss after search cpu %.6e gpu %.6e  %s\n",
            diff, n0, texdiff, worse, better, texdiff - worse - better, mc, mg, ok ? "PASS" : "FAIL"); fails += !ok;
    }
    // 6. --qes snapped copy: the device's decode buffer against the host snap of the downloaded shadow,
    //    exact (shared grid table, strict-FP host index), now and after one latent step with lr > 0 so
    //    the k_snap-after-Adam path (and the frozen clamp) is exercised. Without a live --qes level the
    //    device buffer aliases the shadow and the comparison is a plain copy check.
    {
        size_t bad = 0, bad2 = 0; const size_t n = D.lat.size();
        auto compare = [&]() {
            std::vector<float> zs(n), ptmp(D.mlp.size()), zqd(n);
            cu.download_model(zs.data(), ptmp.data()); cu.download_zq(zqd.data());
            D.lat.z = zs; D.qes_refresh();
            const float* zh = D.zdec();
            size_t d = 0; for (size_t i = 0; i < n; i++) if (zh[i] != zqd[i]) d++;
            return d;
        };
        bad = compare();
        if (!(o.qat > 0 && D.lat.lv.size() == 1)) { cu.lat_step(2, o.lat_sigma, o.lat_lr, o.lat_pairs); bad2 = compare(); }
        bool ok = bad == 0 && bad2 == 0;
        printf("  qes snap        : device snapped copy vs host snap of the downloaded shadow: %zu of %zu values differ, %zu after a latent step with lr %g%s  %s\n",
            bad, n, bad2, o.lat_lr, D.qes_live ? "" : " (no live --qes level: the decode buffer aliases the shadow)", ok ? "PASS" : "FAIL"); fails += !ok;
    }
    // 7. [DCT] the DCT-coded level 0: the device's symbols, decode plane and scale codes against the host snap of the
    // [DCT]    downloaded shadow, exact. Pre-step: the host's symbols are uploaded (set_dct) so the device plane comes from
    // [DCT]    k_dct_recon while the host's comes from the full snap: "plane 0 of N differ" proves the two IDCT + clamp paths
    // [DCT]    agree (the symbols and codes are equal by construction there). Post-step (lat_step with the run's lr): the device
    // [DCT]    plane and symbols come from k_dct_snap, so the forward DCT and the quantizer are compared as well.
    if (D.dct.on) {   // [DCT]
        if (!D.dct.live) printf("  dct snap        : skipped (not live at iteration 0; pass --dct-start 0)  PASS\n");   // [DCT]
        else {   // [DCT]
            const size_t n0 = D.lat.lv[0].size(), ns = D.dct.sym.size(), nc = D.dct.code.size(), n = D.lat.size();   // [DCT]
            std::vector<int16_t> symd(ns); std::vector<uint8_t> coded(nc); std::vector<float> zqd(n), zs(n), ptmp(D.mlp.size());   // [DCT]
            auto ulp_of = [](float a, float b) { int32_t ia, ib; memcpy(&ia, &a, 4); memcpy(&ib, &b, 4); const long long d = (long long)ia - ib; return (int)std::min<long long>(d < 0 ? -d : d, 1 << 30); };   // [DCT]
            auto host_snap = [&]() { cu.download_model(zs.data(), ptmp.data()); D.lat.z = zs; D.qes_refresh(); };   // [DCT]
            const bool lam = D.dct.lambda > 0.0f; std::vector<int32_t> bitsd(lam ? nc : 0); size_t dbits_all = 0;   // [DCT] --dct-lambda: the device's proxy bits per (block, channel) against the host's, summed over the compares below
            auto compare = [&](size_t& ds, size_t& dp, int& ulp, size_t& dc) {   // [DCT]
                cu.download_zq(zqd.data()); cu.download_dct(symd.data(), coded.data());   // [DCT]
                ds = dp = dc = 0; ulp = 0;   // [DCT]
                for (size_t i = 0; i < ns; i++) if (D.dct.sym[i] != symd[i]) ds++;   // [DCT]
                for (size_t i = 0; i < n0; i++) if (D.zq[i] != zqd[i]) { dp++; ulp = std::max(ulp, ulp_of(D.zq[i], zqd[i])); }   // [DCT]
                for (size_t i = 0; i < nc; i++) if (D.dct.code[i] != coded[i]) dc++;   // [DCT]
                if (lam) { cu.download_dct_rate(bitsd.data(), nullptr, nullptr, nullptr, nullptr, nullptr); for (size_t i = 0; i < nc; i++) if (D.dct.bits[i] != bitsd[i]) dbits_all++; }   // [DCT]
            };   // [DCT]
            host_snap();   // [DCT]
            cu.set_dct(dct_desc_of(D.dct, D.W, D.H), D.dct.code.data(), D.dct.sym.data(), D.dct.step.data());   // [DCT] hook: was the two-line DctDesc fill (dct_desc_of carries the --dct-lambda fields too)
            size_t s1, p1, c1, s2, p2, c2; int u1, u2;   // [DCT]
            compare(s1, p1, u1, c1);   // [DCT]
            cu.lat_step(3, o.lat_sigma, o.lat_lr, o.lat_pairs);   // [DCT]
            host_snap();   // [DCT]
            compare(s2, p2, u2, c2);   // [DCT]
            const bool ok = s1 == 0 && p1 == 0 && c1 == 0 && s2 == 0 && p2 == 0 && c2 == 0 && dbits_all == 0;   // [DCT] hook: && dbits_all == 0 (0 without --dct-lambda)
            const std::string chs = D.dct.C > 1 ? " (" + std::to_string(D.dct.C) + " channels)" : "";   // [DCT] C > 1: the channel count names what the sizes cover; C == 1: the pre-change line, byte for byte
            const std::string bcl = lam ? "; proxy bits " + std::to_string(dbits_all) + " of " + std::to_string(2 * nc) + " differ (--dct-lambda, both compares)" : "";   // [DCT] lambda > 0 only (the lambda 0 line is byte-identical)
            printf("  dct snap        : symbols %zu of %zu differ, plane %zu of %zu values%s differ (max %d ulp), codes %zu of %zu differ (device plane from k_dct_recon on the host's symbols: IDCT + clamp parity); after a latent step with lr %g: %zu / %zu (max %d ulp) / %zu (k_dct_snap: forward DCT + quantizer + IDCT parity)%s  %s\n",   // [DCT] hook: the %s after `values` = chs, the %s before PASS = bcl
                s1, ns, p1, n0, chs.c_str(), u1, c1, nc, o.lat_lr, s2, p2, u2, c2, bcl.c_str(), ok ? "PASS" : "FAIL"); fails += !ok;   // [DCT]
            // [DCT] 7b. the refit route (--dct-refit): the host re-probes, replaces its codes and re-snaps, the new codes and
            // [DCT]     symbols go up through set_dct (k_dct_recon), then a latent step makes k_dct_snap quantize with the new codes.
            size_t m1 = 0, m2 = 0, s3, p3, c3, s4, p4, c4; int u3, u4;   // [DCT]
            dbits_all = 0;   // [DCT] --dct-lambda: the refit compares count their own proxy-bit differences
            D.dct_refit(1, m1, m2);   // [DCT]
            cu.set_dct(dct_desc_of(D.dct, D.W, D.H), D.dct.code.data(), D.dct.sym.data(), D.dct.step.data());   // [DCT] hook: was the two-line DctDesc fill (dct_desc_of carries the --dct-lambda fields too)
            compare(s3, p3, u3, c3);   // [DCT]
            cu.lat_step(4, o.lat_sigma, o.lat_lr, o.lat_pairs);   // [DCT]
            host_snap();   // [DCT]
            compare(s4, p4, u4, c4);   // [DCT]
            const bool ok2 = s3 == 0 && p3 == 0 && c3 == 0 && s4 == 0 && p4 == 0 && c4 == 0 && dbits_all == 0;   // [DCT] hook: && dbits_all == 0
            const std::string bcl2 = lam ? "; proxy bits " + std::to_string(dbits_all) + " of " + std::to_string(2 * nc) + " differ" : "";   // [DCT]
            printf("  dct refit       : host re-probe moved %zu of %zu codes (>= 2: %zu), uploaded through set_dct: symbols %zu differ, plane %zu (max %d ulp), codes %zu (k_dct_recon with the refitted codes); after a latent step: %zu / %zu (max %d ulp) / %zu (k_dct_snap quantizes with the refitted codes)%s  %s\n",   // [DCT] hook: %s before PASS = bcl2
                m1, nc, m2, s3, p3, u3, c3, s4, p4, u4, c4, bcl2.c_str(), ok2 ? "PASS" : "FAIL"); fails += !ok2;   // [DCT]
        }   // [DCT]
    }   // [DCT]
    printf("cuda-check: %s\n", fails ? "FAILED" : "all passed");
    return fails ? 1 : 0;
}
#endif

int main(int argc, char** argv) {
#if defined(DEBUG) || defined(_DEBUG)
    printf("DEBUG build\n");   // an unoptimized build announces itself (CMakeLists.txt gives it the _d suffix)
#endif
    Options o;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&](int k = 1) { if (i + k >= argc) { usage(); exit(1); } return argv[i + k]; };
        if (a == "--out") { o.outdir = next(); i++; }
        else if (a == "--crop") { o.crop = atoi(next()); i++; }
        else if (a == "--block") { o.block = atoi(next()); i++; }
        else if (a == "--latent") { o.LW = atoi(next(1)); o.LH = atoi(next(2)); o.LC = atoi(next(3)); i += 3; }
        else if (a == "--latent2") { o.LW2 = atoi(next(1)); o.LH2 = atoi(next(2)); o.LC2 = atoi(next(3)); i += 3; }
        else if (a == "--latent3") { o.LW3 = atoi(next(1)); o.LH3 = atoi(next(2)); o.LC3 = atoi(next(3)); i += 3; }
        else if (a == "--filter") { o.filter = next(); i++; }
        else if (a == "--mlp") {
            o.hidden.clear();
            std::string list = next(); i++;
            for (size_t s = 0; s < list.size();) {
                size_t e = list.find(',', s);
                if (e == std::string::npos) e = list.size();
                if (e > s) o.hidden.push_back(atoi(list.substr(s, e - s).c_str()));
                s = e + 1;
            }
        }
        else if (a == "--leak") { o.leak = (float)atof(next()); i++; }
        else if (a == "--pos") { o.pos = next(); i++; }
        else if (a == "--clamp") { o.clamp_out = true; }
        else if (a == "--iters") { o.iters = atoi(next()); i++; }
        else if (a == "--save-every") { o.save_every = atoi(next()); i++; }
        else if (a == "--print-every") { o.print_every = atoi(next()); i++; }
        else if (a == "--seed") { o.seed = (unsigned)atoi(next()); i++; }
        else if (a == "--mlp-pairs") { o.mlp_pairs = atoi(next()); i++; }
        else if (a == "--mlp-batch") { o.mlp_batch = atoi(next()); i++; }
        else if (a == "--mlp-sigma") { o.mlp_sigma = (float)atof(next()); i++; }
        else if (a == "--mlp-lr") { o.mlp_lr = (float)atof(next()); i++; }
        else if (a == "--mlp-every") { o.mlp_every = atoi(next()); i++; }
        else if (a == "--mlp-fd") { o.mlp_fd_start = (float)atof(next()); i++; }
        else if (a == "--mlp-fd-h") { o.mlp_fd_h = (float)atof(next()); o.mlp_fd_h_set = true; i++; }
        else if (a == "--lat-pairs") { o.lat_pairs = atoi(next()); i++; }
        else if (a == "--lat-sigma") { o.lat_sigma = (float)atof(next()); i++; }
        else if (a == "--lat-lr") { o.lat_lr = (float)atof(next()); i++; }
        else if (a == "--lat-init") { o.lat_init = (float)atof(next()); i++; }
        else if (a == "--lat-init-image") { o.lat_init_image = true; }   // [INIT]
        else if (a == "--lr-anneal") { o.lr_anneal_start = (float)atof(next(1)); o.lr_anneal_final = (float)atof(next(2)); i += 2; }
        else if (a == "--threads") { o.threads = atoi(next()); i++; }
        else if (a == "--qbits") { o.qbits = atoi(next()); i++; }
        else if (a == "--qat") {
            std::string v = next(); i++;
            o.qat_ch.clear();
            size_t pos = 0;
            while (pos <= v.size()) {
                size_t e = v.find(',', pos); if (e == std::string::npos) e = v.size();
                std::string tok = v.substr(pos, e - pos);
                if (tok.empty() || tok.find_first_not_of("0123456789") != std::string::npos) { o.qat_ch.assign(1, -1); break; }
                o.qat_ch.push_back(atoi(tok.c_str()));
                pos = e + 1;
            }
            o.qat = 0; for (int b : o.qat_ch) o.qat = std::max(o.qat, b);
        }
        else if (a == "--qat-every") { o.qat_every = atoi(next()); i++; }
        else if (a == "--qes") {
            std::string v = next(); i++;
            o.qes_ch.clear();
            size_t pos = 0;
            while (pos <= v.size()) {
                size_t e = v.find(',', pos); if (e == std::string::npos) e = v.size();
                std::string tok = v.substr(pos, e - pos);
                if (tok.empty() || tok.find_first_not_of("0123456789") != std::string::npos) { o.qes_ch.assign(1, -1); break; }
                o.qes_ch.push_back(atoi(tok.c_str()));
                pos = e + 1;
            }
        }
        else if (a == "--qes-start") { o.qes_start = (float)atof(next()); i++; }
        else if (a == "--dct-q") {   // [DCT] Q or Q0,Q1,... per level-0 channel (the --qat parser loop); o.dct_q = the maximum
            std::string v = next(); i++;   // [DCT]
            o.dct_q_ch.clear();   // [DCT]
            size_t pos = 0;   // [DCT]
            while (pos <= v.size()) {   // [DCT]
                size_t e = v.find(',', pos); if (e == std::string::npos) e = v.size();   // [DCT]
                std::string tok = v.substr(pos, e - pos);   // [DCT]
                if (tok.empty() || tok.find_first_not_of("0123456789") != std::string::npos) { o.dct_q_ch.assign(1, -1); break; }   // [DCT]
                o.dct_q_ch.push_back(atoi(tok.c_str()));   // [DCT]
                pos = e + 1;   // [DCT]
            }   // [DCT]
            o.dct_q = 0; for (int q : o.dct_q_ch) o.dct_q = std::max(o.dct_q, q);   // [DCT]
        }   // [DCT]
        else if (a == "--dct-start") { o.dct_start = (float)atof(next()); o.dct_opts_given = true; i++; }              // [DCT]
        else if (a == "--dct-refit") { o.dct_refit = atoi(next()); o.dct_opts_given = true; i++; }                    // [DCT]
        else if (a == "--dct-refit-until") { o.dct_refit_until = (float)atof(next()); o.dct_opts_given = true; i++; } // [DCT]
        else if (a == "--dct-dc-step") { o.dct_dc_step = atoi(next()); o.dct_opts_given = true; i++; }                // [DCT]
        else if (a == "--dct-block") { o.dct_block = atoi(next()); o.dct_opts_given = true; i++; }                    // [DCT]
        else if (a == "--dct-deadzone") { o.dct_deadzone = atoi(next()); o.dct_deadzone_given = true; o.dct_opts_given = true; i++; }   // [DCT]
        else if (a == "--dct-shadow-pull") { o.dct_shadow_pull = (float)atof(next()); o.dct_opts_given = true; i++; }   // [DCT]
        else if (a == "--dct-shadow-reset") { o.dct_shadow_reset = true; o.dct_opts_given = true; }   // [DCT]
        else if (a == "--dct-stats") { o.dct_stats = true; o.dct_opts_given = true; }                                 // [DCT]
        else if (a == "--dct-map") { o.dct_map = next(); o.dct_opts_given = true; i++; }                              // [DCT]
        else if (a == "--dct-lambda") { o.dct_lambda = (float)atof(next()); o.dct_lambda_given = true; o.dct_opts_given = true; i++; }               // [DCT]
        else if (a == "--dct-rate") { o.dct_rate = next(); o.dct_rate_given = true; o.dct_opts_given = true; i++; }        // [DCT]
        else if (a == "--dct-lambda-lo") { o.dct_lambda_lo = (float)atof(next()); o.dct_opts_given = true; i++; }           // [DCT]
        else if (a == "--dct-selftest") { o.dct_selftest = true; }                                                    // [DCT]
        else if (a == "--rng") { std::string m = next(); i++; if (m == "hash") o.rng_hash = true; else if (m == "mt") o.rng_hash = false; else { printf("--rng: mt | hash\n"); return 1; } }
        else if (a == "--eval-filter") { o.eval_filter = true; }
        else if (a == "--cuda") { o.cuda = true; }
        else if (a == "--cuda-check") { o.cuda_check = true; }
        else if (a == "--load") { o.load = next(); i++; }
        else if (a == "--resave") { o.resave = true; }
        else if (a == "--write-ntcb") { o.write_ntcb = true; }   // [NTCB]
        else if (a == "--weights") {
            o.weights.clear(); o.weights_given = true;
            std::string list = next(); i++;
            for (size_t s = 0; s <= list.size();) {
                size_t e = list.find(',', s);
                if (e == std::string::npos) e = list.size();
                std::string tok = list.substr(s, e - s);
                char* end = nullptr;
                double w = tok.empty() ? 0.0 : strtod(tok.c_str(), &end);
                if (tok.empty() || *end != 0 || !std::isfinite(w)) { printf("--weights: bad entry '%s' in '%s'\n", tok.c_str(), list.c_str()); return 1; }
                o.weights.push_back((float)w);
                s = e + 1;
            }
        }
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else if (a[0] == '-') { printf("unknown option %s\n", a.c_str()); usage(); return 1; }
        else o.inputs.push_back(a);
    }
    if (o.dct_selftest) return dct_selftest();   // [DCT] hook: the self-test runs before any image is loaded and exits 0 / 1
    if (o.write_ntcb && !ntcb_selfcheck()) return 1;   // [NTCB] hook: the fp16 round-trip self-check, once, only under the flag
    if (o.write_ntcb && o.block > 255) { printf("--write-ntcb: --block %d does not fit the container's 8-bit latent_cell field (at most 255)\n", o.block); return 1; }   // [NTCB] hook
    const bool default_input = o.inputs.empty();
    if (default_input) o.inputs.push_back("kodim23.png");
    const int T = (int)o.inputs.size();
    if (T > MAXT) { printf("at most %d textures per material\n", MAXT); return 1; }
    if (!o.weights_given) o.weights.assign(T, 1.0f);
    if ((int)o.weights.size() != T) { printf("--weights needs %d entries (one per input image), got %zu\n", T, o.weights.size()); return 1; }
    float wsum = 0.0f;
    for (float w : o.weights) { if (w < 0.0f) { printf("--weights must be >= 0\n"); return 1; } wsum += w; }
    if (wsum <= 0.0f) { printf("--weights must not all be zero\n"); return 1; }
    if (o.hidden.empty() || (int)o.hidden.size() > MAXL) { printf("need 1..%d hidden layers\n", MAXL); return 1; }
    if (o.mlp_fd_h <= 0.0f) { printf("--mlp-fd-h must be > 0\n"); return 1; }
    if (o.block < 0) { printf("--block must be >= 0\n"); return 1; }
    if (o.block == 0 && (o.LW < 1 || o.LH < 1)) { printf("--latent needs W, H >= 1 (0 is only allowed with --block)\n"); return 1; }
    if (o.LW < 0 || o.LH < 0 || o.LC < 1 || o.LC > MAXH) { printf("--latent needs W, H >= 0 and 1..%d channels\n", MAXH); return 1; }
    if (o.LC2 > 0 && (o.LW2 < 0 || o.LH2 < 0 || o.LC2 > MAXH || (o.block == 0 && (o.LW2 < 1 || o.LH2 < 1)))) { printf("--latent2 needs W, H >= 1 (0 only with --block) and 1..%d channels\n", MAXH); return 1; }
    if (o.LC2 == 0 && (o.LW2 > 0 || o.LH2 > 0)) { printf("--latent2 needs 1..%d channels\n", MAXH); return 1; }
    if (o.LC3 > 0 && o.LC2 == 0) { printf("--latent3 needs --latent2\n"); return 1; }
    if (o.LC3 > 0 && (o.LW3 < 0 || o.LH3 < 0 || o.LC3 > MAXH || (o.block == 0 && (o.LW3 < 1 || o.LH3 < 1)))) { printf("--latent3 needs W, H >= 1 (0 only with --block) and 1..%d channels\n", MAXH); return 1; }
    if (o.LC3 == 0 && (o.LW3 > 0 || o.LH3 > 0)) { printf("--latent3 needs 1..%d channels\n", MAXH); return 1; }
    if (o.qat_ch.size() == 1 && o.qat_ch[0] < 0) { printf("--qat: malformed bit list (use B or B1,B2,... with B = 1..8)\n"); return 1; }
    for (int b : o.qat_ch) if (b < 0 || b > 8) { printf("--qat needs 1..8 bits per channel (0 = off)\n"); return 1; }
    if (o.qat_ch.size() > 1 && (int)o.qat_ch.size() != o.LC) { printf("--qat lists %zu bit depths but level 0 has %d channels\n", o.qat_ch.size(), o.LC); return 1; }
    if (o.qat > 0) {
        for (int b : o.qat_ch) if (b == 0) { printf("--qat: every channel needs 1..8 bits when quantization is on\n"); return 1; }
        if (o.qat_ch.size() == 1) { const int b = o.qat_ch[0]; o.qat_ch.assign(o.LC, b); }   // one value applies to every channel (copied first: assign(n, v[0]) aliases the vector, undefined; the Debug CRT asserts on it)
    } else o.qat_ch.clear();
    if (o.qat_every < 1) { printf("--qat-every must be >= 1\n"); return 1; }
    if (o.qes_ch.size() == 1 && o.qes_ch[0] < 0) { printf("--qes: malformed bit list (use B or B0,B1[,B2] with B = 2..12, 0 = continuous)\n"); return 1; }
    for (int b : o.qes_ch) if (b != 0 && (b < 2 || b > 12)) { printf("--qes needs 2..12 bits per level (0 = leave that level continuous)\n"); return 1; }
    if (o.qes_start < 0.0f || o.qes_start > 1.0f) { printf("--qes-start must be in [0, 1]\n"); return 1; }
    if (o.qes_ch.empty() && o.qes_start != 0.5f) printf("note: --qes-start is ignored without --qes\n");
    // [DCT] option ranges (before any image is loaded; the layout checks follow the level resolution below)
    if (o.dct_q_ch.size() == 1 && o.dct_q_ch[0] < 0) { printf("--dct-q: malformed quality list (use Q or Q0,Q1,... with Q = 1..100)\n"); return 1; }   // [DCT]
    for (int q : o.dct_q_ch) if (q < 0 || q > 100) { printf("--dct-q needs 1..100 (0 = off)\n"); return 1; }                       // [DCT] hook: was the single-value range check
    if (o.dct_q == 0) o.dct_q_ch.clear();   // [DCT] flag off (absent, `0`, or an all-zero list such as `0,0` of any length): nothing else reads the list, so the checks below see an empty one
    for (int q : o.dct_q_ch) if (q < 1) { printf("--dct-q: every channel needs 1..100 when the DCT plane is on\n"); return 1; }   // [DCT]
    if (o.dct_q_ch.size() > 1 && (int)o.dct_q_ch.size() != o.LC) { printf("--dct-q lists %zu qualities but level 0 has %d channels\n", o.dct_q_ch.size(), o.LC); return 1; }   // [DCT]
    if (o.dct_q_ch.size() == 1) { const int q = o.dct_q_ch[0]; o.dct_q_ch.assign(o.LC, q); }   // [DCT] one value applies to every channel (copied first: assign(n, v[0]) aliases the vector, undefined; the Debug CRT asserts on it)
    if (!o.dct_lambda_given) {   // [DCT] the lambda recipe (PROGRESS.md section 2; his default since September 8, 2026): --dct-lambda L on the command line overrides it
        o.dct_lambda = 0.0f;
        if (o.dct_q > 0) { float L = 0.5f; for (int q : o.dct_q_ch) { const float S = dct_quality_scale(q); L = std::min(L, 7.5f * S * S); } o.dct_lambda = L; o.dct_lambda_recipe = true; }
    }   // [DCT]
    if (o.dct_dc_step < 1 || o.dct_dc_step > 64) { printf("--dct-dc-step needs 1..64\n"); return 1; }                             // [DCT]
    if (o.dct_block != 8) { printf("--dct-block: only 8 in the MVP\n"); return 1; }                                               // [DCT]
    if (o.dct_deadzone != 0 && o.dct_deadzone != 1) { printf("--dct-deadzone: 0 (plain rounding on every AC) or 1 (the dead-zone quantizer)\n"); return 1; }   // [DCT]
    if (o.dct_start < 0.0f || o.dct_start > 1.0f) { printf("--dct-start must be in [0, 1]\n"); return 1; }                        // [DCT]
    if (o.dct_refit < 0) { printf("--dct-refit needs N >= 0 (0 = fit once and freeze)\n"); return 1; }                              // [DCT]
    if (o.dct_refit_until < 0.0f || o.dct_refit_until > 1.0f) { printf("--dct-refit-until must be in [0, 1]\n"); return 1; }        // [DCT]
    if (o.dct_q > 0 && o.dct_refit > 0 && o.dct_refit_until <= o.dct_start) { printf("note: --dct-refit-until %g is not after --dct-start %g, so no refit can happen; the codes stay frozen from the switch\n", o.dct_refit_until, o.dct_start); o.dct_refit = 0; }   // [DCT]
    if (!dct_map_kind_ok(o.dct_map)) { printf("--dct-map: nz | code | bits | lnz | dconly | all\n"); return 1; }                  // [DCT]
    if (!(o.dct_lambda >= 0.0f) || !std::isfinite(o.dct_lambda)) { printf("--dct-lambda needs a finite L >= 0 (0 = off; leave it out for the recipe)\n"); return 1; }   // [DCT]
    if (o.dct_rate != "es" && o.dct_rate != "trunc" && o.dct_rate != "both") { printf("--dct-rate: es | trunc | both\n"); return 1; }   // [DCT]
    if (o.dct_q > 0 && o.dct_rate_given && o.dct_lambda == 0.0f) printf("note: --dct-rate %s has no effect %s\n", o.dct_rate.c_str(), o.dct_lambda_recipe ? "because the lambda recipe gives 0 at q 100" : "without --dct-lambda L > 0");   // [DCT]
    if (!(o.dct_lambda_lo >= 0.0f && o.dct_lambda_lo <= 1.0f)) { printf("--dct-lambda-lo needs W in [0, 1] (1 = the first-order pair costs its full bits, 0 = never charged)\n"); return 1; }   // [DCT]
    if (o.dct_q == 0 && o.dct_opts_given) printf("note: --dct-start / --dct-refit / --dct-refit-until / --dct-dc-step / --dct-block / --dct-deadzone / --dct-stats / --dct-map / --dct-lambda / --dct-rate are ignored without --dct-q\n");   // [DCT] hook: --dct-deadzone in the list
    if (o.qat == 0 && o.qat_every != 1) printf("note: --qat-every is ignored without --qat\n");
    if (o.mlp_pairs < 1 || o.lat_pairs < 1) { printf("pair counts must be >= 1\n"); return 1; }
    if (o.mlp_fd_h_set && o.mlp_fd_start >= 1.0f) printf("note: --mlp-fd-h is ignored without --mlp-fd\n");
    for (int h : o.hidden) if (h < 1 || h > MAXH) { printf("hidden width must be 1..%d\n", MAXH); return 1; }
#ifdef _OPENMP
    if (o.threads > 0) omp_set_num_threads(o.threads);
    printf("OpenMP threads: %d\n", omp_get_max_threads());
#endif

#ifdef _WIN32
    _mkdir(o.outdir.c_str());
#else
    mkdir(o.outdir.c_str(), 0755);
#endif

    // Output file naming: unchanged for one texture, "_tK" suffix per texture otherwise.
    auto tex_name = [&](const char* stem, int t, int it = -1) {
        char name[96];
        if (it >= 0) snprintf(name, sizeof(name), "%s_%06d", stem, it); else snprintf(name, sizeof(name), "%s", stem);
        std::string sname = o.outdir + "/" + name;
        if (T > 1) sname += "_t" + std::to_string(t);
        return sname + ".png";
    };

    // Load every texture (same crop), require identical sizes, interleave into one target.
    Image target;
    {
        std::vector<Image> tex(T);
        for (int t = 0; t < T; t++) {
            if (!find_and_load(argv[0], o.inputs[t], tex[t], o.crop)) {
                printf("could not load %s%s\n", o.inputs[t].c_str(), default_input ? " (the default input; name an image or run from the repository)" : "");
                return 1;
            }
            if (t > 0 && (tex[t].w != tex[0].w || tex[t].h != tex[0].h)) {
                printf("%s is %dx%d after cropping but %s is %dx%d; all textures of a material must have the same size\n",
                    o.inputs[t].c_str(), tex[t].w, tex[t].h, o.inputs[0].c_str(), tex[0].w, tex[0].h);
                return 1;
            }
        }
        if (o.block > 0) {
            // Pad to a multiple of the block size by repeating the last column / row (what
            // BC and ASTC encoders do). From here on the padded image is the source image:
            // every statistic, bitrate and output refers to it.
            const int W0 = tex[0].w, H0 = tex[0].h;
            o.orig_w = W0; o.orig_h = H0;
            // An auto-sized third level covers 2x2 level-2 blocks, so pad to a multiple of 2N then.
            int mult = (o.LC3 > 0 && (o.LW3 == 0 || o.LH3 == 0)) ? 2 * o.block : o.block;   // [DCT]
            if (o.dct_q > 0) { int a = mult, b = o.dct_block; while (b) { const int t = a % b; a = b; b = t; } mult = mult / a * o.dct_block; }   // [DCT] hook: pad to lcm(latent cell, DCT block) so the 8x8 DCT blocks tile the padded image (e.g. --block 6 -> 24)
            const int Wp = (W0 + mult - 1) / mult * mult, Hp = (H0 + mult - 1) / mult * mult;
            if (Wp != W0 || Hp != H0) {
                for (int t = 0; t < T; t++) pad_image(tex[t], Wp, Hp);
                printf("note: %dx%d padded to %dx%d (a multiple of %dx%d%s) by duplicating the last column / row; bitrates are per padded texel, PSNR is measured on the original %dx%d\n", W0, H0, Wp, Hp, mult, mult,   // [DCT]
                    o.dct_q > 0 ? ", the lcm of the latent cell and the --dct-block 8 DCT block" : "", W0, H0);   // [DCT] hook: the %s note
            }
            if (o.LW == 0) o.LW = Wp;
            if (o.LH == 0) o.LH = Hp;
            if (o.LC2 > 0) { if (o.LW2 == 0) o.LW2 = Wp / o.block; if (o.LH2 == 0) o.LH2 = Hp / o.block; }
            if (o.LC3 > 0) { const int b2 = 2 * o.block; if (o.LW3 == 0) o.LW3 = Wp / b2; if (o.LH3 == 0) o.LH3 = Hp / b2; }
        }
        pack_textures(tex, target);
    }
    for (int t = 0; t < T; t++) save_png_src(tex_name("target", t), target, o.orig_w, o.orig_h, t);   // source extent (was the padded image)

    std::mt19937 rng(o.seed);
    Decoder D;
    D.opt = &o;
    if (!D.pos.parse(o.pos)) { printf("bad --pos spec: %s\n", o.pos.c_str()); return 1; }
    D.W = target.w; D.H = target.h;
    if (o.orig_w == 0) { o.orig_w = D.W; o.orig_h = D.H; }   // no --block padding: the source is the decoded size
    // Per-level sampling filter: "bilinear" or "nearest", comma-separated; a single entry applies to level 0 only.
    bool near0 = false, near1 = false, near2 = false;
    {
        std::vector<std::string> fl;
        for (size_t a = 0; a <= o.filter.size();) {
            size_t e = o.filter.find(',', a); if (e == std::string::npos) e = o.filter.size();
            fl.push_back(o.filter.substr(a, e - a)); a = e + 1;
        }
        if (fl.size() > 3) { printf("--filter takes at most three entries (one per latent level)\n"); return 1; }
        for (size_t l = 0; l < fl.size(); l++) {
            bool nr;
            if (fl[l] == "bilinear") nr = false; else if (fl[l] == "nearest") nr = true;
            else { printf("--filter: unknown mode '%s' (bilinear | nearest)\n", fl[l].c_str()); return 1; }
            if (l == 0) near0 = nr; else if (l == 1) near1 = nr; else near2 = nr;
        }
        if (fl.size() > 1 && o.LW2 == 0) printf("note: second --filter entry ignored without --latent2\n");
    }
    if (o.qat > 0 && !near0) { printf("--qat needs nearest sampling on level 0 (--filter nearest): with bilinear a texel is read by pixels of 4 cells, so the per-texel search would not be exact\n"); return 1; }
    D.lat.add(o.LW, o.LH, o.LC, near0);
    D.qat_bits = o.qat;
    D.qat_ch = o.qat_ch;
    if (o.qat > 0) qat_init_grid();
    if (o.LW2 > 0) D.lat.add(o.LW2, o.LH2, o.LC2, near1);
    if (o.LW3 > 0) D.lat.add(o.LW3, o.LH3, o.LC3, near2);
    if (D.pos.uses_level1() && o.LW2 == 0) { printf("--pos uses lv1* features but there is no --latent2\n"); return 1; }
    {   // --qes: resolve the bit list to one entry per level. One value applies to every level
        // without --qat; an explicit list must name every level and may not quantize a --qat level.
        const size_t nlev = D.lat.lv.size();
        if (o.qes_ch.size() == 1) { const int b = o.qes_ch[0]; o.qes_ch.assign(nlev, b); if (o.qat > 0 || o.dct_q > 0) o.qes_ch[0] = 0; }   // [DCT] hook: || o.dct_q > 0 (a single --qes value applies to levels >= 1 under --dct-q, as under --qat)
        else if (o.qes_ch.size() > 1 && o.qes_ch.size() != nlev) { printf("--qes lists %zu levels but the model has %zu\n", o.qes_ch.size(), nlev); return 1; }
        if (o.qat > 0 && !o.qes_ch.empty() && o.qes_ch[0] > 0) { printf("--qes on level 0 conflicts with --qat: a --qat level is discrete already (use --qes 0,B... or drop --qat)\n"); return 1; }
        o.qes = false; for (int b : o.qes_ch) if (b > 0) o.qes = true;
        if (!o.qes && !o.qes_ch.empty()) {
            if (o.qat > 0 && nlev == 1) printf("note: --qes has nothing to apply to: the only level is --qat (discrete already)\n");
            else printf("note: --qes names no level to quantize (every entry is 0)\n");
            o.qes_ch.clear();
        }
        D.qes_on = o.qes;
        D.qes.assign(nlev, QesLevel());
        for (size_t l = 0; l < nlev; l++) D.qes[l].bits = o.qes ? o.qes_ch[l] : 0;
    }
    if (o.dct_q > 0) {   // [DCT] hook: --dct-q layout checks (DCT_MVP_PLAN.md section 1 items 2-4) and the level-0 setup (4.1)
        if (o.qat > 0) { printf("--dct-q replaces --qat on level 0; drop one\n"); return 1; }   // [DCT]
        if (!o.qes_ch.empty() && o.qes_ch[0] > 0) { printf("--qes on level 0 conflicts with --dct-q: use --qes 0,B\n"); return 1; }   // [DCT]
        if (!near0) { printf("--dct-q needs nearest sampling on level 0 (--filter nearest,...)\n"); return 1; }   // [DCT]
        if (o.LW != D.W || o.LH != D.H) { printf("--dct-q needs level 0 at full resolution (--latent 0 0 C with --block, or --latent W H C at the image size); level 0 is %dx%d, the image %dx%d\n", o.LW, o.LH, D.W, D.H); return 1; }   // [DCT] hook: `C` (was `1`) since section 7 of DCT_NOTES.md
        if (o.LC > MAX_DCT_CH) { printf("--dct-q: level 0 may have at most %d channels\n", MAX_DCT_CH); return 1; }   // [DCT] hook: was the `o.LC != 1` refusal (the CUDA envelope refuses the same)
        if (D.W % o.dct_block != 0 || D.H % o.dct_block != 0) { printf("--dct-q needs an image size that is a multiple of %d (%dx%d here): pad with --block\n", o.dct_block, D.W, D.H); return 1; }   // [DCT]
        dct_init_tables();   // [DCT]
        D.dct.on = true; D.dct.N = o.dct_block; D.dct.dc_step = o.dct_dc_step; D.dct.C = o.LC;   // [DCT]
        D.dct.deadzone = o.dct_deadzone != 0;   // [DCT] --dct-deadzone
        D.dct.q = o.dct_q_ch;   // [DCT] hook: was `q.assign(o.LC, o.dct_q)` (one q for every channel); the per-channel list now, so dct_build_steps below builds different tables when they differ
        D.dct.BW = D.W / o.dct_block; D.dct.BH = D.H / o.dct_block;   // [DCT]
        D.dct.refit_every = o.dct_refit; D.dct.refit_until_it = (int)std::ceil((double)o.dct_refit_until * o.iters);   // [DCT]
        D.dct.sym.assign(D.dct.nblk() * D.dct.C * DCT_NN, 0);   // [DCT]
        D.dct.code.assign(D.dct.nblk() * D.dct.C, 0);   // [DCT]
        D.dct.step.assign((size_t)D.dct.C * 16 * DCT_NN, 0);   // [DCT]
        for (int c = 0; c < D.dct.C; c++) dct_build_steps(D.dct.q[c], D.dct.N, D.dct.dc_step, &D.dct.step[(size_t)c * 16 * DCT_NN]);   // [DCT]
        { const int lv = 512 / o.dct_dc_step + 1; int b = 0; while ((1 << b) < lv) b++; D.dct.dc_raw_bits = b; }   // [DCT]
        D.dct.lambda = o.dct_lambda; D.dct.rate_es = o.dct_lambda > 0.0f && o.dct_rate != "trunc"; D.dct.rate_trunc = o.dct_lambda > 0.0f && o.dct_rate != "es";   // [DCT] --dct-lambda (DCT_RATE_PLAN.md section 1)
        D.dct.lo_w16 = std::min(16, std::max(0, (int)lround(o.dct_lambda_lo * 16.0f)));   // [DCT] --dct-lambda-lo W -> round(16 W)
        for (int k = 0; k < 16; k++) D.dct.lam_t16[k] = D.dct.rate_trunc ? (float)((double)o.dct_lambda / (dct_gain_of_code(k) * dct_gain_of_code(k) * 16.0)) : 0.0f;   // [DCT] lambda_t(k) = L / g_k^2 in 1/16-bit units (double, rounded once: the DCT_AK pattern)
        D.dct.bits.assign(D.dct.nblk() * D.dct.C, 0);   // [DCT]
    }   // [DCT]
    {
        std::normal_distribution<float> N(0.0f, o.lat_init);
        for (auto& z : D.lat.z) z = N(rng);
        // --qat: snap level 0 onto its grid. The Gaussian draw is kept (same RNG stream as
        // without --qat, so the MLP init is identical) and rounded to the nearest grid value.
        if (o.qat > 0) qat_snap_level0(D.lat, o.qat_ch);
        if (o.lat_init_image && o.load.empty()) {   // [INIT] the smooth-relief start: luma of the (padded) target, [0,1] -> [-1,1], into channel 0 of level 0 (a full-resolution level 0 only)
            const Latent& L0 = D.lat.lv[0];
            if (L0.W == D.W && L0.H == D.H) {
                for (int y = 0; y < D.H; y++) for (int x = 0; x < D.W; x++) {
                    double l = 0; for (int t = 0; t < T; t++) { const float* px = &target.rgb[((size_t)y * D.W + x) * target.nc + 3 * t]; l += 0.299 * px[0] + 0.587 * px[1] + 0.114 * px[2]; }
                    D.lat.z[L0.off + ((size_t)y * D.W + x) * L0.C] = (float)(2.0 * l / T - 1.0);
                }
                if (o.qat > 0) qat_snap_level0(D.lat, o.qat_ch);
                printf("init     : level 0 channel 0 = image luma (mean over %d texture%s) scaled to [-1, 1] (--lat-init-image)\n", T, T > 1 ? "s" : "");
            } else printf("note: --lat-init-image needs a full-resolution level 0 (level 0 is %dx%d, the image %dx%d); ignored\n", L0.W, L0.H, D.W, D.H);
        }
    }
    D.mlp.nout = 3 * T;   // init() sizes the output layer from nout; must precede it
    if (D.nin() > MAXH) { printf("too many MLP inputs\n"); return 1; }
    if (D.mlp.nout > MAXOUT) { printf("too many MLP outputs\n"); return 1; }   // implied by T <= MAXT; belt and braces for the out[MAXOUT] buffers
    D.mlp.init(D.nin(), o.hidden, rng);
    D.mlp.leak = o.leak;
    D.wsum = wsum;
    D.cw.resize(D.mlp.nout);
    for (int c = 0; c < D.mlp.nout; c++) D.cw[c] = o.weights[c / 3];
    bool qes_from_file = false;   // --qes ranges restored from a v12 file (no fit in this run)
    bool dct_from_file = false;   // [DCT] symbols and scale codes restored from a v13 file (no probe, no re-snap in this run)
    if (!o.load.empty()) {
        FILE* f = fopen(o.load.c_str(), "rb");
        // Accepted layouts: [DCT] v14 (magic 0x4E54433E, v13 plus an int dct_deadzone after the three DCT header ints), v13 (magic 0x4E54433D, v12 plus the DCT header fields and payload; DCT_NOTES.md),
        // v12 (magic 0x4E54433C, v11 plus, per level, an int of --qes bits and C (lo, hi) range floats after the extra-level dims),   [DCT] hook: was the first line of this comment
        // v11 (magic 0x4E54433B, v10 plus five ints: decoded size, source size, clamp flag),
        // v10 (magic 0x4E54433A, v9 plus one int of --qat bits per level-0 channel),
        // v9 (magic 0x4E544339: 10 header ints (magic, level-0 W H C, nin, poslen, act, nlayers, nlevels, textures),
        //     one filter int per level, a deblock int and a falloff float, the leaky-ReLU slope float, an int of
        //     --qat bits (0 = off), extra-level dims, hidden widths, poslen bytes of spec, latent, mlp).
        // The legacy six-int header and v2..v8 were retired in v0.8 (tag pre-v0.8-removal reads them).
        int hdr[8];
        if (!f || fread(hdr, sizeof(int), 6, f) != 6) { printf("cannot read %s\n", o.load.c_str()); return 1; }
        std::vector<int> saved_hidden;
        std::string saved_pos;
        std::vector<Latent> saved_lv;   // W,H,C of each saved level
        bool hdr_ok;
        int saved_T = 1;
        std::vector<int> saved_filter;   // per level, 0 bilinear / 1 nearest
        int saved_deblock = 0; float saved_falloff = 1.0f, saved_leak = 0.01f;   // deblock: layout only, refused when 1
        int saved_qat = 0;   // largest --qat bit depth (0 = continuous)
        std::vector<int> saved_qat_ch;   // v10+: per channel; v9: saved_qat for every channel
        std::vector<int> saved_qes;      // v12+: --qes bits per level (0 = continuous); older files: all 0
        std::vector<std::vector<float>> saved_qes_lo, saved_qes_hi;   // v12+: per level, per channel
        int saved_dct_N = 0, saved_dct_dc = 0, saved_dct_dz = 1; std::vector<int> saved_dct_q;   // [DCT] v13 / v14: DCT block size (0 = not a DCT file), DC step, --dct-deadzone flag (v14; 1 in a v13 file), per-channel q
        const bool is_ntcb = (uint32_t)hdr[0] == NTCB_MAGIC;   // [NTCB] hook: the container's magic; its header fills the same saved_* locals and the fread cascade is skipped
        NtcbHeader ntcb_h; std::vector<uint8_t> ntcb_buf; std::string ntcb_err;   // [NTCB]
        if (is_ntcb) {   // [NTCB] read the whole file, parse and validate the header (ranges, length, payload hash), then fill the saved_* locals
            fseek(f, 0, SEEK_END); const long sz = ftell(f); fseek(f, 0, SEEK_SET);   // [NTCB]
            ntcb_buf.resize(sz > 0 ? (size_t)sz : 0);   // [NTCB]
            hdr_ok = sz > 0 && fread(ntcb_buf.data(), 1, ntcb_buf.size(), f) == ntcb_buf.size() && ntcb_read_header(ntcb_buf, ntcb_h, ntcb_err);   // [NTCB]
            if (!hdr_ok) { printf("%s: corrupt or truncated ntcb file: %s\n", o.load.c_str(), ntcb_err.empty() ? "cannot read it" : ntcb_err.c_str()); fclose(f); return 1; }   // [NTCB]
            hdr[1] = ntcb_h.lv[0].W; hdr[2] = ntcb_h.lv[0].H; hdr[3] = ntcb_h.lv[0].C; hdr[4] = ntcb_h.nin; hdr[5] = (int)ntcb_h.pos.size(); hdr[6] = ntcb_h.act; hdr[7] = ntcb_h.nhidden;   // [NTCB]
            saved_T = ntcb_h.T; saved_leak = ntcb_h.leak; saved_hidden = ntcb_h.hidden; saved_pos = ntcb_h.pos;   // [NTCB]
            const int nlevels = ntcb_h.nlevels;   // [NTCB]
            saved_lv.resize(nlevels); saved_filter.assign(nlevels, 0); saved_qes.assign(nlevels, 0); saved_qes_lo.assign(nlevels, std::vector<float>()); saved_qes_hi.assign(nlevels, std::vector<float>());   // [NTCB]
            saved_qat_ch.assign(hdr[3], 0);   // [NTCB]
            for (int l = 0; l < nlevels; l++) {   // [NTCB]
                const NtcbLevel& N = ntcb_h.lv[l];   // [NTCB]
                saved_lv[l].W = N.W; saved_lv[l].H = N.H; saved_lv[l].C = N.C; saved_filter[l] = N.filter;   // [NTCB]
                if (N.mode == 1) { saved_qes[l] = N.bits[0]; saved_qes_lo[l] = N.lo; saved_qes_hi[l] = N.hi; }   // [NTCB] a --qes grid: restored below as a v12 file's would be
                if (N.mode == 2) { saved_qat_ch = N.bits; for (int b : N.bits) saved_qat = std::max(saved_qat, b); }   // [NTCB] the --qat palette
                if (N.mode == 3) { saved_dct_N = DCT_N; saved_dct_dc = N.dc_step; saved_dct_q = N.q; saved_dct_dz = N.deadzone; }   // [NTCB] the DCT symbols (needs the same --dct-q / --dct-dc-step / --dct-deadzone, as a v14 file)   [DCT] hook: saved_dct_dz
            }   // [NTCB]
            if ((ntcb_h.clamp == 1) != o.clamp_out) printf("note: the model was saved with %s output; this run uses %s\n", ntcb_h.clamp ? "--clamp" : "sigmoid", o.clamp_out ? "--clamp" : "sigmoid");   // [NTCB]
        } else   // [NTCB]
        if (hdr[0] == 0x4E54433E || hdr[0] == 0x4E54433D || hdr[0] == 0x4E54433C || hdr[0] == 0x4E54433B || hdr[0] == 0x4E54433A || hdr[0] == 0x4E544339) {   // [DCT] hook: v13 and v14 magics accepted
            bool v14 = hdr[0] == 0x4E54433E;   // [DCT] v14: v13 plus the dct_deadzone int
            bool v13 = v14 || hdr[0] == 0x4E54433D;   // [DCT] hook: v14 ||
            bool v12 = v13 || hdr[0] == 0x4E54433C;   // [DCT] hook: v13 ||
            bool v11 = v12 || hdr[0] == 0x4E54433B;
            bool v10 = v11 || hdr[0] == 0x4E54433A;
            hdr_ok = fread(hdr + 6, sizeof(int), 2, f) == 2 && hdr[7] >= 1 && hdr[7] <= MAXL;
            int nlevels = 1;
            if (hdr_ok) hdr_ok = fread(&nlevels, sizeof(int), 1, f) == 1 && nlevels >= 1 && nlevels <= 3;
            if (hdr_ok) hdr_ok = fread(&saved_T, sizeof(int), 1, f) == 1 && saved_T >= 1 && saved_T <= MAXT;
            saved_filter.assign(nlevels, 0);
            if (hdr_ok) for (int l = 0; l < nlevels && hdr_ok; l++) hdr_ok = fread(&saved_filter[l], sizeof(int), 1, f) == 1 && (saved_filter[l] == 0 || saved_filter[l] == 1);
            if (hdr_ok) hdr_ok = fread(&saved_deblock, sizeof(int), 1, f) == 1 && fread(&saved_falloff, sizeof(float), 1, f) == 1;   // v7 pair: read for the layout, refused below when set
            if (hdr_ok) hdr_ok = fread(&saved_leak, sizeof(float), 1, f) == 1;
            if (hdr_ok) hdr_ok = fread(&saved_qat, sizeof(int), 1, f) == 1 && saved_qat >= 0 && saved_qat <= 8;
            if (hdr_ok) hdr_ok = hdr[3] >= 1 && hdr[3] <= MAXH;   // level-0 channel count bounds the v10 per-channel list
            saved_qat_ch.assign(hdr_ok ? hdr[3] : 0, saved_qat);
            if (hdr_ok && v10) {
                int mx = 0;
                for (size_t c = 0; c < saved_qat_ch.size() && hdr_ok; c++) { hdr_ok = fread(&saved_qat_ch[c], sizeof(int), 1, f) == 1 && saved_qat_ch[c] >= 0 && saved_qat_ch[c] <= 8; mx = std::max(mx, saved_qat_ch[c]); }
                if (hdr_ok && mx != saved_qat) hdr_ok = false;   // the max field and the per-channel list must agree
            }
            if (hdr_ok && v11) {   // image size, source size, clamp flag: informational for the trainer (the image on the command line is the source)
                int img[5] = { 0, 0, 0, 0, 0 };
                hdr_ok = fread(img, sizeof(int), 5, f) == 5 && img[0] >= 1 && img[1] >= 1 && img[2] >= 1 && img[3] >= 1 && (img[4] == 0 || img[4] == 1);
                if (hdr_ok && (img[4] == 1) != o.clamp_out) printf("note: the model was saved with %s output; this run uses %s\n", img[4] ? "--clamp" : "sigmoid", o.clamp_out ? "--clamp" : "sigmoid");
            }
            if (hdr_ok) {
                saved_lv.resize(nlevels);
                saved_lv[0].W = hdr[1]; saved_lv[0].H = hdr[2]; saved_lv[0].C = hdr[3];
                for (int l = 1; l < nlevels && hdr_ok; l++) {
                    int d[3] = { 0, 0, 0 };
                    hdr_ok = fread(d, sizeof(int), 3, f) == 3;
                    if (hdr_ok) { saved_lv[l].W = d[0]; saved_lv[l].H = d[1]; saved_lv[l].C = d[2]; }
                }
                saved_qes.assign(nlevels, 0); saved_qes_lo.resize(nlevels); saved_qes_hi.resize(nlevels);
                if (v12) for (int l = 0; l < nlevels && hdr_ok; l++) {   // --qes bits, then C lo and C hi floats
                    const int C = saved_lv[l].C;
                    hdr_ok = C >= 1 && C <= MAXH && fread(&saved_qes[l], sizeof(int), 1, f) == 1 && (saved_qes[l] == 0 || (saved_qes[l] >= 2 && saved_qes[l] <= 12));
                    if (hdr_ok) {
                        saved_qes_lo[l].resize(C); saved_qes_hi[l].resize(C);
                        hdr_ok = fread(saved_qes_lo[l].data(), sizeof(float), C, f) == (size_t)C && fread(saved_qes_hi[l].data(), sizeof(float), C, f) == (size_t)C;
                        for (int c = 0; c < C && hdr_ok; c++) hdr_ok = std::isfinite(saved_qes_lo[l][c]) && std::isfinite(saved_qes_hi[l][c]) && saved_qes_lo[l][c] <= saved_qes_hi[l][c];
                    }
                }
                if (hdr_ok && saved_qat > 0 && saved_qes[0] > 0) hdr_ok = false;   // a level cannot be both searched and ES-quantized (ntc_decode refuses this too)
                if (hdr_ok && v13) {   // [DCT] v13 / v14: N, dc_step, C (= level-0 channels), (v14) the dct_deadzone int, then C ints q; level 0 must be neither --qat nor --qes and a multiple of N
                    int di[3] = { 0, 0, 0 };   // [DCT]
                    hdr_ok = fread(di, sizeof(int), 3, f) == 3 && di[0] == DCT_N && di[1] >= 1 && di[1] <= 64 && di[2] == hdr[3] && di[2] >= 1 && di[2] <= MAX_DCT_CH;   // [DCT]
                    if (hdr_ok && v14) hdr_ok = fread(&saved_dct_dz, sizeof(int), 1, f) == 1 && (saved_dct_dz == 0 || saved_dct_dz == 1);   // [DCT] v14: --dct-deadzone (a v13 file leaves the default 1)
                    if (hdr_ok) {   // [DCT]
                        saved_dct_N = di[0]; saved_dct_dc = di[1]; saved_dct_q.resize(di[2]);   // [DCT]
                        hdr_ok = fread(saved_dct_q.data(), sizeof(int), saved_dct_q.size(), f) == saved_dct_q.size();   // [DCT]
                        for (int q : saved_dct_q) if (q < 1 || q > 100) hdr_ok = false;   // [DCT]
                    }   // [DCT]
                    if (hdr_ok && (saved_qat > 0 || saved_qes[0] > 0 || hdr[1] % DCT_N != 0 || hdr[2] % DCT_N != 0 || saved_filter[0] != 1)) hdr_ok = false;   // [DCT]
                }   // [DCT]
            }
            if (hdr_ok) {
                saved_hidden.resize(hdr[7]);
                hdr_ok = fread(saved_hidden.data(), sizeof(int), saved_hidden.size(), f) == saved_hidden.size();
            }
            if (hdr_ok) {
                // hdr[5] is the positional spec length, followed by the spec text.
                hdr_ok = hdr[5] >= 0 && hdr[5] < 4096;
                if (hdr_ok) {
                    saved_pos.resize(hdr[5]);
                    hdr_ok = fread(&saved_pos[0], 1, saved_pos.size(), f) == saved_pos.size();
                }
            }
        } else {
            printf("%s: not a v9..v14 ntc model file (the legacy header and v2..v8 were retired in v0.8)\n", o.load.c_str());   // [DCT] hook: v14 in the message
            fclose(f);
            return 1;
        }
        bool saved_qes_any = false; for (int b : saved_qes) if (b > 0) saved_qes_any = true;
        bool dims_ok = hdr_ok && saved_lv.size() == D.lat.lv.size();
        for (size_t l = 0; dims_ok && l < saved_lv.size(); l++)
            dims_ok = saved_lv[l].W == D.lat.lv[l].W && saved_lv[l].H == D.lat.lv[l].H && saved_lv[l].C == D.lat.lv[l].C
                   && (saved_filter[l] == 1) == D.lat.lv[l].nearest;
        bool leak_ok = saved_leak == o.leak;
        bool qat_ok = (saved_qat == 0 && o.qat == 0) || (saved_qat == 0 && o.qat > 0)   // continuous -> --qat warm start is allowed (snapped below)
                   || (saved_qat > 0 && o.qat > 0 && saved_qat_ch == o.qat_ch);
        // [DCT] a v13 file needs the same --dct-q (per channel) and --dct-dc-step (no fp32 shadow is stored, so it cannot be re-quantized);
        // [DCT] a v9..v12 file loaded with --dct-q is the warm start (its continuous level 0 is snapped at --dct-start)
        if (saved_dct_N > 0 && !o.dct_deadzone_given && saved_dct_dz != o.dct_deadzone) { printf("dct      : --dct-deadzone %d adopted from the file (not given on the command line)\n", saved_dct_dz); o.dct_deadzone = saved_dct_dz; D.dct.deadzone = saved_dct_dz != 0; }   // [DCT] the file's quantizer wins unless the flag was given
        const bool dct_ok = saved_dct_N == 0 || (o.dct_q > 0 && saved_dct_q == D.dct.q && saved_dct_dc == o.dct_dc_step && saved_dct_dz == o.dct_deadzone);   // [DCT] hook: && the --dct-deadzone flag
        if (!hdr_ok) { printf("%s: corrupt or truncated header\n", o.load.c_str()); fclose(f); return 1; }
        if (hdr[6] != 0) { printf("%s was saved with activation %d (1 relu, 2 tanh, 3 sine); only leaky is supported since v0.8\n", o.load.c_str(), hdr[6]); fclose(f); return 1; }
        if (saved_deblock == 1) { printf("%s was saved with --deblock (falloff %g), which was removed in v0.8; it cannot be loaded\n", o.load.c_str(), saved_falloff); fclose(f); return 1; }
        { PosEnc chk; if (!chk.parse(saved_pos)) { printf("%s was saved with --pos %s, which uses a positional kind removed in v0.8; it cannot be loaded by this version\n", o.load.c_str(), saved_pos.c_str()); fclose(f); return 1; } }
        if (!dims_ok || !leak_ok || !qat_ok || !dct_ok || hdr[4] != D.mlp.nin || saved_T != T   // [DCT] hook: || !dct_ok
            || saved_pos != D.pos.spec || saved_hidden != o.hidden) {
            std::string mlp, l2;
            for (size_t k = 0; k < saved_hidden.size(); k++) mlp += (k ? "," : "") + std::to_string(saved_hidden[k]);
            if (saved_lv.size() > 1) l2 = " --latent2 " + std::to_string(saved_lv[1].W) + " " + std::to_string(saved_lv[1].H) + " " + std::to_string(saved_lv[1].C);
            else if (D.lat.lv.size() > 1) l2 = " (no --latent2)";
            if (saved_lv.size() > 2) l2 += " --latent3 " + std::to_string(saved_lv[2].W) + " " + std::to_string(saved_lv[2].H) + " " + std::to_string(saved_lv[2].C);
            else if (D.lat.lv.size() > 2) l2 += " (no --latent3)";
            std::string flt;
            for (size_t l = 0; l < saved_filter.size(); l++) flt += (l ? "," : "") + std::string(saved_filter[l] ? "nearest" : "bilinear");
            std::string dbs;
            if (saved_leak != 0.01f) dbs += " --leak " + std::to_string(saved_leak);
            if (saved_qat > 0) dbs += " --qat " + qat_spec(saved_qat_ch); else if (o.qat > 0) dbs += " (no --qat)";
            if (saved_qes_any) dbs += " --qes " + qes_spec(saved_qes); else if (o.qes) dbs += " (no --qes)";
            if (saved_dct_N > 0) { std::string qs; for (size_t c = 0; c < saved_dct_q.size(); c++) qs += (c ? "," : "") + std::to_string(saved_dct_q[c]); dbs += " --dct-q " + qs + " --dct-dc-step " + std::to_string(saved_dct_dc) + (saved_dct_dz == 0 ? " --dct-deadzone 0" : ""); }   // [DCT] hook: the --dct-deadzone 0 clause
            else if (o.dct_q > 0) dbs += " (no --dct-q)";   // [DCT]
            printf("%s was saved with --latent %d %d %d%s --filter %s%s --mlp %s --pos %s and %d texture(s); pass the same options\n",
                o.load.c_str(), hdr[1], hdr[2], hdr[3], l2.c_str(), flt.c_str(), dbs.c_str(), mlp.c_str(), saved_pos.c_str(), saved_T);
            fclose(f);
            return 1;
        }
        bool ok;   // [DCT]
        std::vector<QesLevel> ntcb_q8;   // [NTCB] per level: the grid of a mode-0 level (bits > 0), installed frozen below
        if (is_ntcb) ok = ntcb_restore(D, ntcb_h, ntcb_buf, ntcb_q8, ntcb_err);   // [NTCB] hook: the container's sections into D.lat.z / D.dct / D.mlp.p
        else   // [NTCB]
        if (saved_dct_N > 0) {   // [DCT] v13 payload: symbols, codes, the floats of levels >= 1 (level 0 is reconstructed below); D.dct is sized (dims_ok, dct_ok)
            const size_t n0 = D.lat.lv[0].size();   // [DCT]
            ok = fread(D.dct.sym.data(), sizeof(int16_t), D.dct.sym.size(), f) == D.dct.sym.size()   // [DCT]
              && fread(D.dct.code.data(), sizeof(uint8_t), D.dct.code.size(), f) == D.dct.code.size()   // [DCT]
              && fread(D.lat.z.data() + n0, sizeof(float), D.lat.z.size() - n0, f) == D.lat.z.size() - n0   // [DCT]
              && fread(D.mlp.p.data(), sizeof(float), D.mlp.p.size(), f) == D.mlp.p.size();   // [DCT]
            for (uint8_t k : D.dct.code) if (k > 15) ok = false;   // [DCT]
            std::fill(D.lat.z.begin(), D.lat.z.begin() + n0, 0.0f);   // [DCT]
        } else   // [DCT]
        ok = fread(D.lat.z.data(), sizeof(float), D.lat.z.size(), f) == D.lat.z.size()   // [DCT]
               && fread(D.mlp.p.data(), sizeof(float), D.mlp.p.size(), f) == D.mlp.p.size();
        fclose(f);
        if (!ok && is_ntcb) { printf("%s: corrupt or truncated ntcb file: %s\n", o.load.c_str(), ntcb_err.c_str()); return 1; }   // [NTCB] hook: the specific message
        if (!ok) { printf("truncated %s\n", o.load.c_str()); return 1; }
        if (o.qat > 0) {   // a --qat file is already on-grid; a continuous file (warm start) is snapped here
            size_t moved = qat_snap_level0(D.lat, o.qat_ch);
            if (moved) printf("note: %zu level-0 values were off the --qat %s grid and were snapped (warm start from a continuous model)\n", moved, qat_spec(o.qat_ch).c_str());
        }
        // --qes: the same bit depths restore the stored ranges (frozen, no refit, so --iters 0 reproduces the
        // run's final numbers); anything else is a warm start from the stored values (ranges fitted at --qes-start).
        if (saved_qes_any && o.qes && saved_qes == o.qes_ch) {
            for (size_t l = 0; l < D.lat.lv.size(); l++) if (o.qes_ch[l] > 0) { D.qes[l].lo = saved_qes_lo[l]; D.qes[l].hi = saved_qes_hi[l]; qes_rebuild(D.qes[l], D.lat.lv[l].C); }
            D.zq = D.lat.z; D.qes_live = true; D.qes_frozen = true; qes_from_file = true;
            size_t moved = D.qes_refresh();
            if (moved) printf("note: %zu values of the loaded --qes levels were off their stored grids and were snapped\n", moved);
            printf("qes      : ranges restored from the file (frozen)\n");
        } else if (saved_qes_any) {
            printf("note: the model was saved with --qes %s; this run uses %s: warm start from the stored values%s\n", qes_spec(saved_qes).c_str(),
                o.qes ? ("--qes " + qes_spec(o.qes_ch)).c_str() : "no --qes", o.qes ? " (ranges fitted at --qes-start)" : "");
        }
        {   // [NTCB] hook: a mode-0 level of the container carries the run's post-hoc grid (lo, hi, B). It is installed as a frozen --qes level so
            // [NTCB] bitrate_stats charges it through the qes branch (the same raw / header bits as the continuous branch, values returned untouched)
            // [NTCB] instead of re-quantizing the dequantized values against hi' = lo + range, which can differ from hi by an ulp. Only with an ntcb file.
            bool any = false;   // [NTCB]
            for (size_t l = 0; l < ntcb_q8.size() && l < D.lat.lv.size(); l++) if (ntcb_q8[l].bits > 0) {   // [NTCB]
                if (o.qes && o.qes_ch[l] != 0) { printf("ntcb     : level %zu: the file's %d-bit post-hoc grid is not installed: this run quantizes the level with --qes %d (warm start from the file's values)\n", l, ntcb_q8[l].bits, o.qes_ch[l]); continue; }   // [NTCB]
                if (l == 0 && (o.qat > 0 || o.dct_q > 0)) { printf("ntcb     : level 0: the file's %d-bit post-hoc grid is not installed: this run makes level 0 discrete with %s (warm start from the file's values)\n", ntcb_q8[l].bits, o.qat > 0 ? "--qat" : "--dct-q"); continue; }   // [NTCB]
                D.qes[l] = ntcb_q8[l]; D.qes[l].from_ntcb = true; any = true;   // [NTCB]
                printf("ntcb     : level %zu: %d-bit grid restored from the file (frozen; its stored values are the run's post-hoc quantization)\n", l, D.qes[l].bits);   // [NTCB]
            }   // [NTCB]
            if (any) {   // [NTCB]
                if (D.zq.empty()) D.zq = D.lat.z;   // [NTCB]
                D.qes_live = true; D.qes_frozen = true;   // [NTCB]
                const size_t moved = D.qes_refresh();   // [NTCB]
                if (moved) printf("note: %zu values of the restored grid levels were off their grids and were snapped\n", moved);   // [NTCB]
                if (o.iters > 0) printf("note: training on from this file treats those levels as --qes levels (shadow clamped to the stored ranges)\n");   // [NTCB]
            }   // [NTCB]
        }   // [NTCB]
        if (saved_dct_N > 0) {   // [DCT] v13: level 0 reconstructed from the symbols (no fp32 shadow in the file); the shadow's level 0 is set equal to it, codes frozen, no probe, no re-snap (the file's symbols are the truth until the first latent step)
            if (D.zq.empty()) D.zq = D.lat.z;   // [DCT]
            dct_recon_level0(D.dct, D.lat.lv[0], D.zq.data());   // [DCT]
            memcpy(D.lat.z.data(), D.zq.data(), D.lat.lv[0].size() * sizeof(float));   // [DCT]
            D.dct.live = true; D.dct.fitted_at = -1; dct_from_file = true;   // [DCT]
            printf("dct      : symbols and scale codes restored from the file (level 0 = IDCT of the symbols, no re-snap at load%s)\n", (o.dct_refit > 0 && o.iters > 0) ? "; the codes refit on the run's schedule" : "; codes frozen");   // [DCT]
        } else if (o.dct_q > 0) printf("note: the model's continuous level 0 is the warm start for --dct-q %s: the scale codes are fitted and the plane snapped at --dct-start (iteration %d)\n", dct_spec(D.dct).c_str(), (int)std::ceil((double)o.dct_start * o.iters));   // [DCT] hook: %s = the per-channel list (was %d)
        if (is_ntcb) printf("ntcb     : the container holds no fp32 shadow: the fp32 `final psnr` of the done: line is decoded from the file's quantized values\n");   // [NTCB] hook
        printf("loaded   : %s\n", o.load.c_str());
    }
    if (D.dct.on && !D.dct.live) D.qes_refresh();   // [DCT] hook: the pre-start clamp of the shadow's level 0 before the first decode (with or without --load)

    // --qes schedule: ranges fitted and frozen at the top of iteration qes_start_it (0 = here, before anything decodes).
    const int qes_start_it = o.qes ? (int)std::ceil((double)o.qes_start * o.iters) : -1;
    auto qes_begin = [&](int it) {   // fit + freeze + snap at the start iteration
        D.qes_fit_all();
        printf("iter %6d  qes: ranges fitted from the fp32 shadow, decode from the snapped copy from here (frozen: the shadow is clamped to them)", it);
        for (size_t l = 0; l < D.lat.lv.size(); l++) if (D.qes[l].fitted()) {
            printf(" | level %zu %d-bit", l, D.qes[l].bits);
            for (int c = 0; c < D.lat.lv[l].C; c++) printf("%s[%.3f, %.3f]", c ? " " : " ", D.qes[l].lo[c], D.qes[l].hi[c]);
        }
        printf("\n"); fflush(stdout);
    };
    if (o.qes && !D.qes_live && qes_start_it == 0) qes_begin(0);
    // [DCT] schedule: the scale codes are fitted from the decoder sensitivity probe at the top of iteration dct_start_it
    // [DCT] (0 = here, after the qes fit so the probe sees the snapped block latent, and before the CUDA init) and frozen;
    // [DCT] from then on every decode reads the DCT-snapped plane. The PSNR before / after the first snap is measured on
    // [DCT] the host (two CPU decodes, once), so the drop is attributable despite the coincident switches at 0.5.
    const int dct_start_it = o.dct_q > 0 ? (int)std::ceil((double)o.dct_start * o.iters) : -1;   // [DCT]
    auto dct_begin = [&](int it) {   // [DCT]
        Image a, b;   // [DCT]
        D.decode_full(D.mlp.p.data(), D.zdec(), a);   // [DCT]
        const double ps0 = psnr_of(mse_of(target, a, o.weights, wsum, o.orig_w, o.orig_h).weighted);   // [DCT]
        std::vector<float> gain;   // [DCT] the probe's block gains (the per-channel mean clause when C > 1)
        D.dct_fit_all(it, &gain);   // [DCT] hook: the gain argument
        D.decode_full(D.mlp.p.data(), D.zdec(), b);   // [DCT]
        const double ps1 = psnr_of(mse_of(target, b, o.weights, wsum, o.orig_w, o.orig_h).weighted);   // [DCT]
        const DctStats S = dct_analyze(D.dct);   // [DCT]
        printf("iter %6d  dct: scale codes fitted from the decoder sensitivity probe (frozen), level 0 decodes from the DCT-snapped plane from here%s; psnr %.2f -> %.2f dB; nz %.2f/blk, eob-only %.1f%%, clamped %.2f%%; histogram k0..k15",   // [DCT] hook: %s = the mean-gain clause (empty when C == 1)
            it, dct_gain_clause(D.dct, gain).c_str(), ps0, ps1, S.nz_mean, 100.0 * S.eob_only_frac, D.lat.lv[0].size() ? 100.0 * D.dct.last_clamped / D.lat.lv[0].size() : 0.0);   // [DCT]
        for (int k = 0; k < 16; k++) printf(" %zu", S.codehist[k]);   // [DCT]
        if (D.dct.rate_trunc) printf("; trunc %zu of %zu pre-truncation nonzero ACs", D.dct.last_truncated, D.dct.last_nz_before); else if (D.dct.lambda > 0.0f) printf("; trunc off");   // [DCT] --dct-lambda part B at the first snap
        printf("\n");   // [DCT]
        dct_print_codes_all(D.dct, S);   // [DCT] hook: was dct_print_codes(D.dct, S) (one `dct codes[c]` line per channel when C > 1)
        printf("dct map legend: --dct-map nz buckets 0 / 1 / 2 / 3-4 / 5-8 / 9-16 / 17-32 / 33-63 nonzero ACs = black / navy / blue / cyan / green / yellow / orange / white (one colour per 8x8 block)\n");   // [DCT]
        fflush(stdout);   // [DCT]
    };   // [DCT]
    if (o.dct_q > 0 && !D.dct.live && dct_start_it == 0) dct_begin(0);   // [DCT] hook (also the fast-sweep mode: --load <v12> --dct-q Q --iters 0)
    // [DCT] --dct-refit: at every multiple of N after the switch (not the switch itself, not the last iteration, none after
    // [DCT] --dct-refit-until) the probe is re-run on the current decoder, the codes replaced and level 0 re-snapped from
    // [DCT] the shadow; PSNR before / after the re-snap from two host decodes (on --cuda the model is downloaded first and
    // [DCT] the new codes / symbols uploaded afterwards through set_dct, which never writes the device's shadow).
    auto dct_refit_due = [&](int it) {   // [DCT]
        return o.dct_q > 0 && D.dct.live && D.dct.refit_every > 0 && it > 0 && it < o.iters && (dct_from_file || it != dct_start_it) && it <= D.dct.refit_until_it && it % D.dct.refit_every == 0;   // [DCT]
    };   // [DCT]
    auto dct_refit_now = [&](int it) {   // [DCT]
        Image a, b;   // [DCT]
        D.decode_full(D.mlp.p.data(), D.zdec(), a);   // [DCT]
        const double ps0 = psnr_of(mse_of(target, a, o.weights, wsum, o.orig_w, o.orig_h).weighted);   // [DCT]
        size_t m1 = 0, m2 = 0, m1c[MAX_DCT_CH] = { 0, 0, 0, 0 };   // [DCT] hook: m1c = the >= 1 movers per channel
        std::vector<float> gain;   // [DCT] the probe's block gains (the per-channel mean clause when C > 1)
        D.dct_refit(it, m1, m2, m1c, &gain);   // [DCT]
        D.decode_full(D.mlp.p.data(), D.zdec(), b);   // [DCT]
        const double ps1 = psnr_of(mse_of(target, b, o.weights, wsum, o.orig_w, o.orig_h).weighted);   // [DCT]
        const DctStats S = dct_analyze(D.dct);   // [DCT]
        const size_t nc = D.dct.code.size();   // [DCT]
        printf("iter %6d  dct: codes refitted (%d); moved >= 1: %.1f%%, >= 2: %.1f%% (of %zu)",   // [DCT] hook: the per-channel clause follows when C > 1
            it, D.dct.refits, nc ? 100.0 * m1 / nc : 0.0, nc ? 100.0 * m2 / nc : 0.0, nc);   // [DCT]
        if (D.dct.C > 1) { printf(" (ch"); for (int c = 0; c < D.dct.C; c++) printf("%s %.1f%%", c ? " /" : "", nc ? 100.0 * m1c[c] * D.dct.C / nc : 0.0); printf(" of %zu each)", nc / (size_t)D.dct.C); }   // [DCT] moved >= 1 as a fraction of that channel's codes (nc / C of them; the denominator is printed so the clause cannot be read against the (of N) total)
        printf("%s", dct_gain_clause(D.dct, gain).c_str());   // [DCT] ` (mean gain ch a / b)` when C > 1, nothing when C == 1
        printf("; psnr %.2f -> %.2f dB; nz %.2f/blk, eob-only %.1f%%, clamped %.2f%%; histogram k0..k15",   // [DCT]
            ps0, ps1, S.nz_mean, 100.0 * S.eob_only_frac, D.lat.lv[0].size() ? 100.0 * D.dct.last_clamped / D.lat.lv[0].size() : 0.0);   // [DCT]
        for (int k = 0; k < 16; k++) printf(" %zu", S.codehist[k]);   // [DCT]
        if (D.dct.rate_trunc) printf("; trunc %zu of %zu pre-truncation nonzero ACs", D.dct.last_truncated, D.dct.last_nz_before); else if (D.dct.lambda > 0.0f) printf("; trunc off");   // [DCT] --dct-lambda part B at the re-snap
        printf("\n");   // [DCT]
        if (D.dct.C > 1) dct_print_codes_all(D.dct, S);   // [DCT] one `dct codes[c]` line per channel (nothing extra when C == 1)
        fflush(stdout);   // [DCT]
    };   // [DCT]
#ifndef NTC_CUDA
    if (o.cuda) { printf("this build has no CUDA backend (configure with -DNTC_CUDA=ON, or the 'cuda' CMake preset)\n"); return 1; }
#else
    ntc_cuda::Trainer cu;
    // Hand the host-fitted --qes tables to the device and rebuild its snapped copy (after every fit / refit / freeze).
    auto qes_sync_device = [&]() {
        if (!o.cuda || !D.qes_live) return;
        ntc_cuda::QesDesc q; memset(&q, 0, sizeof(q));
        std::vector<float> grid;
        q.live = 1; q.frozen = D.qes_frozen ? 1 : 0;
        for (size_t l = 0; l < D.lat.lv.size(); l++) {
            if (!D.qes[l].fitted()) continue;
            q.bits[l] = D.qes[l].bits;
            const int n = D.qes[l].levels() + 1;
            for (int c = 0; c < D.lat.lv[l].C; c++) {
                q.lo[l][c] = D.qes[l].lo[c]; q.hi[l][c] = D.qes[l].hi[c]; q.range[l][c] = D.qes[l].range[c];
                q.grid_off[l][c] = (int)grid.size() + c * n;
            }
            grid.insert(grid.end(), D.qes[l].grid.begin(), D.qes[l].grid.end());
        }
        cu.set_qes(q, grid.data(), grid.size());   // uploads the tables and snaps the device latent
    };
    // [DCT] Hand the host-fitted codes, the symbols and the integer step tables to the device; it rebuilds its level-0
    // [DCT] decode plane from the symbols (k_dct_recon, bit for bit the host's reconstruction) and never touches its shadow.
    auto dct_sync_device = [&]() {   // [DCT]
        if (!o.cuda || !D.dct.live) return;   // [DCT]
        cu.set_dct(dct_desc_of(D.dct, D.W, D.H), D.dct.code.data(), D.dct.sym.data(), D.dct.step.data());   // [DCT] hook: was the two-line DctDesc fill (dct_desc_of adds the --dct-lambda fields: rate flags, rate_scale, lam_t16)
    };   // [DCT]
    if (o.cuda) {
        ntc_cuda::ModelDesc md;
        md.W = D.W; md.H = D.H; md.T = T;
        md.nlev = (int)D.lat.lv.size();
        for (int l = 0; l < md.nlev; l++) { md.lv[l].W = D.lat.lv[l].W; md.lv[l].H = D.lat.lv[l].H; md.lv[l].C = D.lat.lv[l].C; md.lv[l].off = D.lat.lv[l].off; md.lv[l].nearest = D.lat.lv[l].nearest; }
        if (D.pos.feats.size() > (size_t)ntc_cuda::MAX_POS) { printf("--cuda: too many positional features\n"); return 1; }
        md.npos = (int)D.pos.feats.size();
        for (int q = 0; q < md.npos; q++) md.pos[q].kind = (int)D.pos.feats[q].kind;   // PosKind and PosKindDev share one order
        md.nin = D.mlp.nin; md.nout = D.mlp.nout;
        if (o.hidden.size() > (size_t)ntc_cuda::MAX_HIDDEN) { printf("--cuda: too many hidden layers\n"); return 1; }
        md.nhidden = (int)o.hidden.size(); for (int l = 0; l < md.nhidden; l++) md.hidden[l] = o.hidden[l];
        md.leak = D.mlp.leak; md.clamp_out = o.clamp_out;
        for (int c = 0; c < ntc_cuda::MAX_OUT; c++) md.cw[c] = c < D.mlp.nout ? D.cw[c] : 0.0f;
        md.wsum = D.wsum;
        for (int c = 0; c < ntc_cuda::MAX_CH; c++) md.qat_bits[c] = (o.qat > 0 && c < (int)o.qat_ch.size()) ? o.qat_ch[c] : 0;
        md.qat = o.qat;
        md.qat_grid = &QAT_GRID[0][0];
        for (int l = 0; l < ntc_cuda::MAX_LEVELS; l++) md.qes_bits[l] = (o.qes && l < (int)o.qes_ch.size() && o.qes_ch[l] > 0) ? o.qes_ch[l] : (l < (int)D.qes.size() && D.qes[l].fitted() ? D.qes[l].bits : 0);   // [NTCB] hook: the grid of a container's mode-0 level (installed frozen by the loader) reaches the device like a --qes level, also next to --qes depths on other levels; without such a level the value is 0 as before
        md.dct = D.dct.on ? 1 : 0; md.dct_C = D.dct.C; md.dct_N = D.dct.N; md.dct_dc_step = D.dct.dc_step;   // [DCT] hook: ModelDesc fills
        for (int c = 0; c < ntc_cuda::MAX_DCT_CH; c++) md.dct_q[c] = (D.dct.on && c < D.dct.C) ? D.dct.q[c] : 0;   // [DCT]
        md.dct_basis = D.dct.on ? &DCT_BASIS[0][0] : nullptr;   // [DCT] the 64 basis bit patterns, copied to the device
        md.dct_zigzag = D.dct.on ? DCT_ZIGZAG : nullptr;   // [DCT] the zigzag order (the --dct-lambda token walk on the device)
        md.max_pairs = o.mlp_pairs;
        md.max_batch = o.mlp_batch;
        md.seed = o.seed;
        std::string why;
        if (!cu.init(md, D.lat.z.data(), D.mlp.p.data(), target.rgb.data(), why)) { printf("--cuda: %s\n", why.c_str()); return 1; }
        if (o.threads) printf("note: --threads is ignored with --cuda\n");
        qes_sync_device();
        dct_sync_device();   // [DCT] hook
    }
    if (o.cuda && o.cuda_check) return cuda_check(D, target, o, cu);
#endif
    const double bits_latent = D.lat.size() * 32.0;
    const double bits_mlp = D.mlp.size() * 32.0;
    if (T == 1) printf("image    : %dx%d (%s)\n", D.W, D.H, o.inputs[0].c_str());
    else {
        printf("images   : %dx%d x%d textures (", D.W, D.H, T);
        for (int t = 0; t < T; t++) printf("%s%s", t ? ", " : "", o.inputs[t].c_str());
        printf(")\nweights  :");
        for (int t = 0; t < T; t++) printf(" %g", o.weights[t]);
        printf("   (loss = sum_t w_t mse_t / sum_t w_t; mse/psnr/best are on that loss; tex lists are unweighted per-texture psnr)\n");
    }
    printf("latent   : %dx%dx%d = %zu floats, %s\n", o.LW, o.LH, o.LC, D.lat.lv[0].size(), near0 ? "nearest (cells are blocks)" : "bilinear");
    if (D.lat.lv.size() > 1)
        printf("latent2  : %dx%dx%d = %zu floats (all levels %zu), %s\n", o.LW2, o.LH2, o.LC2, D.lat.lv[1].size(), D.lat.size(), near1 ? "nearest" : "bilinear");
    if (D.lat.lv.size() > 2)
        printf("latent3  : %dx%dx%d = %zu floats, %s\n", o.LW3, o.LH3, o.LC3, D.lat.lv[2].size(), near2 ? "nearest" : "bilinear");
    printf("mlp      : %s = %zu params, leaky hidden%s, %s output\n", D.mlp.describe().c_str(), D.mlp.size(),
        o.leak != 0.01f ? (" (leak " + std::to_string(o.leak) + ")").c_str() : "", o.clamp_out ? "clamp" : "sigmoid");
    printf("inputs   : %d latent + %d positional (%s)\n", D.lat.channels(), D.pos.count(), D.pos.spec.c_str());
#ifdef NTC_CUDA
    if (o.cuda) printf("cuda     : %s\n", cu.banner().c_str());
#endif
    // Raw quantized size: every level at --qbits with a 2-float min/max per channel, except a
    // --qat level 0, which is charged its own bit count and needs no scale.
    // --qes levels are charged their own bits plus the same 2-float min/max per channel.
    double bits_q = D.mlp.size() * 16.0;
    for (size_t l = 0; l < D.lat.lv.size(); l++) {
        if (l == 0 && o.dct_q > 0) continue;   // [DCT] hook: a DCT-coded level 0 has no per-texel bits; its rate comes from the bit simulator once live
        const bool fixed = (l == 0 && o.qat > 0);
        const int lb = (o.qes && o.qes_ch[l] > 0) ? o.qes_ch[l] : o.qbits;
        if (fixed) { for (int c = 0; c < D.lat.lv[l].C; c++) bits_q += (double)D.lat.lv[l].W * D.lat.lv[l].H * o.qat_ch[c]; }
        else bits_q += D.lat.lv[l].size() * (double)lb + D.lat.lv[l].C * 64.0;
    }
    std::string qnote;
    if (o.qat > 0) qnote += " (level 0 at its --qat depth";
    if (o.qes) qnote += std::string(o.qat > 0 ? ", " : " (") + "--qes levels at their --qes depth";
    if (!qnote.empty()) qnote += ")";
    const std::string qbl = o.qes ? "--qes " + qes_spec(o.qes_ch) : std::to_string(o.qbits) + "-bit";
    if (o.dct_q > 0)   // [DCT] hook: no raw total with level 0 at 0 bits; the levels >= 1 figure only, labelled
        printf("bitrate  : fp32 %.3f bpp (latent %.3f + mlp %.3f); --dct-q %s level 0 (rate from the bit simulator once live) + %s levels >= 1 + fp16 mlp: levels >= 1 + mlp %.3f bpp raw%s (the progress line charges level 0 at %d bits until the switch)\n",   // [DCT] hook: %s = the per-channel list (was %d)
            (bits_latent + bits_mlp) / (D.W * D.H), bits_latent / (D.W * D.H), bits_mlp / (D.W * D.H), dct_spec(D.dct).c_str(), qbl.c_str(), bits_q / (D.W * D.H), qnote.c_str(), o.qbits);   // [DCT]
    else   // [DCT]
    printf("bitrate  : fp32 %.3f bpp (latent %.3f + mlp %.3f); %s latent + fp16 mlp %.3f bpp raw%s\n",
        (bits_latent + bits_mlp) / (D.W * D.H), bits_latent / (D.W * D.H), bits_mlp / (D.W * D.H),
        qbl.c_str(), bits_q / (D.W * D.H), qnote.c_str());
    if (T > 1)
        printf("bpp/tex  : fp32 %.3f bpp, %s latent + fp16 mlp %.3f bpp raw%s (shared bits / %d textures)\n",   // [DCT]
            (bits_latent + bits_mlp) / ((double)D.W * D.H * T), qbl.c_str(), bits_q / ((double)D.W * D.H * T), o.dct_q > 0 ? " without level 0" : "", T);   // [DCT] hook: the %s
    printf("mlp ES   : %d pairs, batch %d, sigma %g, lr %g, every %d%s\n", o.mlp_pairs, o.mlp_batch, o.mlp_sigma, o.mlp_lr, o.mlp_every, o.rng_hash ? " (hash noise)" : "");
    // Phase start iterations: first it with it/iters >= start (start >= 1 -> never).
    auto phase_iter = [&](float start) { return start >= 1.0f ? o.iters + 1 : (int)std::ceil((double)start * o.iters); };
    const int fd_from = phase_iter(o.mlp_fd_start);
    if (fd_from <= o.iters)
        printf("mlp FD   : from iteration %d, h = %g (%zu weights -> %zu evals per step, ~%.0f decode-equivalents on the minibatch)\n",
            fd_from, o.mlp_fd_h, D.mlp.size(), 2 * D.mlp.size(), 2.0 * D.mlp.size() * o.mlp_batch / (D.W * D.H));
    if (o.qat > 0)
        printf("qat      : level 0 quantized in-loop to %s bits per channel, exhaustive per-texel search every %d iters (%d image decodes)\n",
            qat_spec(o.qat_ch).c_str(), o.qat_every, qat_decodes(o.qat_ch));
    if (o.qes) {
        std::string lv;
        for (size_t l = 0; l < D.lat.lv.size(); l++) if (o.qes_ch[l] > 0) lv += (lv.empty() ? "level " : ", level ") + std::to_string(l) + " at " + std::to_string(o.qes_ch[l]) + " bits";
        printf("qes      : %s: quantization-aware ES (per-channel min/max grid, decode from the snapped copy, ES perturbs the snapped point, update to the fp32 shadow); ", lv.c_str());
        if (qes_from_file) printf("ranges from the loaded file, frozen\n");
        else printf("ranges fitted at iteration %d, then frozen (shadow clamped)\n", qes_start_it);
    }
    if (o.dct_q > 0) {   // [DCT] hook: the banner line
        printf("dct      : level 0 = %dx%d DCT-coded selector plane, q %s (JPEG luma table K.1, libjpeg scaling), DC step %d (uniform, independent of q), AC quantizer %s, per-block 4-bit scale code from the decoder sensitivity probe; ES on the fp32 shadow (clamped to [-1,1] from the start), decode from the snapped plane ",   // [DCT] hook: the AC quantizer clause (--dct-deadzone)
            D.dct.N, D.dct.N, dct_spec(D.dct).c_str(), D.dct.dc_step, D.dct.deadzone ? "dead zone alpha 0.5 with the first-order exemption" : "plain rounding on every AC (--dct-deadzone 0)");   // [DCT] hook: the quantizer name
        if (dct_from_file && D.dct.refit_every > 0 && o.iters > 0) printf("(codes restored from the file, refitted every %d iterations through iteration %d)", D.dct.refit_every, std::min(D.dct.refit_until_it, o.iters - 1));   // [DCT]
        else if (dct_from_file) printf("(codes restored from the file, frozen)");   // [DCT]
        else if (D.dct.refit_every > 0) printf("from iteration %d (codes fitted then, refitted every %d iterations through iteration %d, then frozen)", dct_start_it, D.dct.refit_every, std::min(D.dct.refit_until_it, o.iters - 1));   // [DCT]
        else printf("from iteration %d (codes fitted then, frozen)", dct_start_it);   // [DCT]
        printf("; level 0 has no bit depth; its rate figures are a bit simulator (nothing is entropy coded)");   // [DCT] hook: the pre-change text without its "\n" (the C == 1 banner is byte-identical to it)
        if (D.dct.C > 1) printf("; probe: 20 MLP evaluations per block and channel (%.2f decode-equivalents per probe for %d channels)", 20.0 * D.dct.nblk() * D.dct.C / ((double)D.W * D.H), D.dct.C);   // [DCT] the probe cost (20 = 4 positions x 5 selector values; linear in the channel count), C > 1 only
        if (D.dct.lambda > 0.0f) printf("; rate: lambda %g%s (%s): loss = mse + L * proxy bits / npix, proxy = Exp-Golomb token costs (DCT_RATE_PLAN.md 2.2) of the perturbed shadow's symbols in the latent ES, truncation at snap time with lambda_t(k) = L / g_k^2; the first-order pair (zigzag 1, 2) is charged %d/16 of its token cost (--dct-lambda-lo)", D.dct.lambda, o.dct_lambda_recipe ? " (the recipe: min(0.5, 7.5 S(q)^2) over the channels; --dct-lambda L overrides)" : " (--dct-lambda)", dct_rate_mode(D.dct), D.dct.lo_w16);   // [DCT] --dct-lambda / --dct-lambda-lo
        else printf("; rate: no rate term (%s)", o.dct_lambda_recipe ? "the recipe gives lambda 0 at q 100" : "--dct-lambda 0");   // [DCT]
        printf("\n");   // [DCT]
    }   // [DCT]
    if (!(o.qat > 0 && D.lat.lv.size() == 1))
        printf("latent ES: %d pairs (full image), sigma %g, lr %g%s\n", o.lat_pairs, o.lat_sigma, o.lat_lr, o.qat > 0 ? (D.lat.lv.size() > 2 ? " (levels 1 and 2; level 0 is searched)" : " (level 1 only)") : "");
    if (o.lr_anneal_start < 1.0f)
        printf("anneal   : learning rates x1 -> x%g from iteration %d to the end\n", o.lr_anneal_final, (int)std::ceil((double)o.lr_anneal_start * o.iters));
    fflush(stdout);

    MlpTrainer mt; mt.init(D.mlp.size());
    LatentTrainer lt; lt.init(D.lat.size());

    Image recon, recon_q;
    std::vector<float> zpost;   // the post-hoc quantized latent of bitrate_stats (= the decode copy on --qes levels)
    // Full decode with the current weights and a latent (the model's own decode buffer or e.g.
    // the post-hoc quantized copy); on the GPU the model's latent is the device-resident one.
    auto decode_with = [&](const float* z, Image& img) {
#ifdef NTC_CUDA
        if (o.cuda) {
            img.w = D.W; img.h = D.H; img.nc = D.mlp.nout; img.rgb.resize((size_t)D.W * D.H * img.nc);
            cu.decode_full(z == D.zdec() ? nullptr : z, img.rgb.data());
            return;
        }
#endif
        D.decode_full(D.mlp.p.data(), z, img);
    };
    // Progress-line label of the quantized PSNR: the post-hoc --qbits depth, or the --qes depths once a level trains quantized.
    const std::string qlabel_base = o.qes ? "q" + qes_spec(o.qes_ch) : "q" + std::to_string(o.qbits);   // [DCT] hook: was `const std::string qlabel`
    auto qlabel = [&]() -> std::string {   // [DCT] hook: "dct50" (plus ",q8" for the --qes levels >= 1) once level 0 is DCT-coded
        if (!D.dct.live) return qlabel_base;   // [DCT]
        std::string r = "dct" + dct_spec(D.dct), q;   // [DCT]
        if (o.qes) for (size_t l = 1; l < o.qes_ch.size(); l++) q += (q.empty() ? "" : ",") + std::to_string(o.qes_ch[l]);   // [DCT]
        return q.empty() ? r : r + ",q" + q;   // [DCT]
    };   // [DCT]
    std::vector<int16_t> dct_last_sym; std::vector<float> dct_last_zq;   // [DCT] symbols / level-0 plane at the last progress print (the "chg" churn figures)
    auto t0 = std::chrono::steady_clock::now();
    auto elapsed = [&]() { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
    double last_print_time = 0.0, print_interval = 1.0;   // print_interval grows with the print's own cost (download + decode + statistics) so a large image keeps the GPU busy

    decode_with(D.zdec(), recon);
    Mse m = mse_of(target, recon, o.weights, wsum, o.orig_w, o.orig_h), mq;
    double mse = m.weighted;
    // [NTCB] --write-ntcb, decision 2 of NTCB_PLAN.md: the final weights are rounded to fp16 values (once, before the last decode) so the
    // [NTCB] last progress line, model.bin, recon_q_final.png, the done: line and the container all describe the same decoder. On the GPU the
    // [NTCB] rounded weights are uploaded with the shadow (upload_model re-snaps the copy from the shadow), then the host's --qes grids and
    // [NTCB] DCT symbols are re-installed (set_qes, then set_dct last: it rebuilds level 0 of the copy from the symbols, never from the shadow).
    // [NTCB] The PSNR before and after the rounding is printed: one extra decode (the before); the after is the decode the caller needs
    // [NTCB] anyway, left in `recon` / `m` / `mse`, so every figure printed from here on is post-rounding.
    auto ntcb_round = [&]() -> bool {   // [NTCB]
        size_t changed = 0; float maxdw = 0.0f;   // [NTCB]
        decode_with(D.zdec(), recon);   // [NTCB]
        const double ps0 = psnr_of(mse_of(target, recon, o.weights, wsum, o.orig_w, o.orig_h).weighted);   // [NTCB]
        if (!ntcb_mlp_round_fp16(D.mlp.p, changed, maxdw)) { printf("ntcb     : a weight is non-finite or above 65504 and has no fp16 value; nothing written\n"); return false; }   // [NTCB]
#ifdef NTC_CUDA   // [NTCB]
        if (o.cuda) { cu.upload_model(D.lat.z.data(), D.mlp.p.data()); qes_sync_device(); dct_sync_device(); }   // [NTCB]
#endif   // [NTCB]
        decode_with(D.zdec(), recon);   // [NTCB]
        m = mse_of(target, recon, o.weights, wsum, o.orig_w, o.orig_h); mse = m.weighted;   // [NTCB]
        printf("ntcb     : mlp weights rounded to fp16 (%zu of %zu changed, max |dw| %.3e); psnr %.2f -> %.2f dB; figures from here on are post-rounding\n", changed, D.mlp.p.size(), maxdw, ps0, psnr_of(mse));   // [NTCB]
        return true;   // [NTCB]
    };   // [NTCB]
    printf("iter %6d  mse %.3f  psnr %6.2f dB  (initial)", 0, mse, psnr_of(mse));
    if (T > 1) { printf(" | tex"); for (int t = 0; t < T; t++) printf(" %.2f", psnr_of(m.tex[t])); }
    printf("\n");
    for (int t = 0; t < T; t++) save_png(tex_name("recon", t, 0), recon, t);
    if (o.save_every > 0) {   // the initial latents belong to the periodic snapshot series
    save_latent_png(o.outdir + "/latent_000000.png", D.lat.lv[0], D.lat.level(0));
    if (D.lat.lv.size() > 1) save_latent_png(o.outdir + "/latent2_000000.png", D.lat.lv[1], D.lat.level(1));
    if (D.lat.lv.size() > 2) save_latent_png(o.outdir + "/latent3_000000.png", D.lat.lv[2], D.lat.level(2));
    }
    fflush(stdout);

    double best_psnr = -1;
    double batch_loss = 0, diff_std = 0;
    for (int it = 1; it <= o.iters; it++) {
        // Per-iteration copy of the options with annealed learning rates.
        auto anneal = [&](float start, float fin) {
            float t = (float)it / (float)o.iters;
            if (t <= start || start >= 1.0f) return 1.0f;
            float u = (t - start) / (1.0f - start);
            return 1.0f + (fin - 1.0f) * std::min(1.0f, u);
        };
        Options oi = o;
        float lrm = anneal(o.lr_anneal_start, o.lr_anneal_final);
        oi.mlp_lr *= lrm; oi.lat_lr *= lrm;
        // Late-phase switch (--mlp-fd). Start iteration computed once above.
        bool mlp_fd = it >= fd_from;
        if (it == fd_from) {
            // Fresh optimizer state for a different gradient estimator: the ES phase's
            // second-moment estimate would otherwise throttle FD steps for ~1000 iterations.
            mt.adam.init(D.mlp.size());
#ifdef NTC_CUDA
            if (o.cuda) cu.reset_mlp_adam();
#endif
            printf("iter %6d  MLP switched to finite differences; Adam state reset\n", it); fflush(stdout);
        }
        if (o.qes) {   // --qes schedule (keyed to the iteration counter, so runs stay reproducible)
            bool changed = false;
            if (!D.qes_live && it == qes_start_it) {
#ifdef NTC_CUDA
                if (o.cuda) cu.download_model(D.lat.z.data(), D.mlp.p.data());
#endif
                qes_begin(it); changed = true;
            }
#ifdef NTC_CUDA
            if (changed) qes_sync_device();
#else
            (void)changed;
#endif
        }
        bool dct_switch = false;   // [DCT]
        if (o.dct_q > 0 && !D.dct.live && it == dct_start_it) {   // [DCT] hook: the DCT switch (after the qes switch, so the probe sees the snapped block latent); forces a progress print
#ifdef NTC_CUDA   // [DCT]
            if (o.cuda) { cu.download_model(D.lat.z.data(), D.mlp.p.data()); D.qes_refresh(); }   // [DCT]
#endif   // [DCT]
            dct_begin(it);   // [DCT]
#ifdef NTC_CUDA   // [DCT]
            dct_sync_device();   // [DCT]
#endif   // [DCT]
            dct_switch = true;   // [DCT]
        }   // [DCT]
        if (dct_refit_due(it)) {   // [DCT] hook: the periodic refit (--dct-refit), at the top of the iteration like the switch
#ifdef NTC_CUDA   // [DCT]
            if (o.cuda) { cu.download_model(D.lat.z.data(), D.mlp.p.data()); D.qes_refresh(); }   // [DCT] host snap with the old codes = the device's plane, bit for bit
#endif   // [DCT]
            dct_refit_now(it);   // [DCT]
#ifdef NTC_CUDA   // [DCT]
            dct_sync_device();   // [DCT] upload the new codes and symbols; k_dct_recon rebuilds the device plane, the shadow is untouched
            if (o.dct_shadow_reset) cu.shadow_pull(D.lat.lv[0].off, D.lat.lv[0].size(), 1.0f);   // [DCT] --dct-shadow-reset: device shadow = snapped plane (pull with F = 1), re-snapped
#endif   // [DCT]
            if (o.dct_shadow_reset && !o.cuda) { const size_t o0 = D.lat.lv[0].off, n0 = D.lat.lv[0].size(); memcpy(&D.lat.z[o0], &D.zq[o0], n0 * sizeof(float)); D.qes_refresh(); }   // [DCT] --dct-shadow-reset (host): shadow = snapped plane
        }   // [DCT]
        if (it % o.mlp_every == 0) {
#ifdef NTC_CUDA
            if (o.cuda) {
                const int M = o.mlp_batch;
                if (mlp_fd) cu.mlp_step_fd(it, oi.mlp_fd_h, oi.mlp_lr, M);
                else cu.mlp_step(it, oi.mlp_sigma, oi.mlp_lr, o.mlp_pairs, M);
            } else
#endif
            { mt.cur_it = it; batch_loss = mlp_fd ? mt.step_fd(D, target, rng, oi, diff_std) : mt.step(D, target, rng, oi, diff_std); }
        }
        const char* mlp_stat = mlp_fd ? "fd-rms" : "dstd";
        // --qat with a single level: nothing is ES-perturbed, so skip the (2K decodes) ES step entirely.
#ifdef NTC_CUDA
        if (o.cuda) {
            if (!(o.qat > 0 && D.lat.lv.size() == 1)) cu.lat_step(it, oi.lat_sigma, oi.lat_lr, o.lat_pairs);
            if (D.dct.live && o.dct_shadow_pull > 0.0f) cu.shadow_pull(D.lat.lv[0].off, D.lat.lv[0].size(), o.dct_shadow_pull);   // [DCT] --dct-shadow-pull (device: pull + re-snap)
            if (o.qat > 0 && (it % o.qat_every == 0 || it == o.iters)) cu.qat_search();
        } else
#endif
        {
            if (!(o.qat > 0 && D.lat.lv.size() == 1)) lt.step(D, target, rng, oi);
            if (D.dct.live && o.dct_shadow_pull > 0.0f) {   // [DCT] --dct-shadow-pull: z += F * (zq - z) on level 0, then re-snap (= the device's k_shadow_pull + snap)
                const size_t o0 = D.lat.lv[0].off, n0 = D.lat.lv[0].size(); const float f = o.dct_shadow_pull;
                for (size_t i = 0; i < n0; i++) D.lat.z[o0 + i] += f * (D.zq[o0 + i] - D.lat.z[o0 + i]);
                D.qes_refresh();
            }
            if (o.qat > 0 && (it % o.qat_every == 0 || it == o.iters)) qat_search(D, target, o.qat_ch);
        }

        // Each print downloads the model and decodes the full image, so by default it is
        // paced by wall time rather than by iteration count.
        bool do_print = it == o.iters || dct_switch || (o.print_every > 0 ? it % o.print_every == 0 : elapsed() - last_print_time >= print_interval);   // print_interval: >= 1 s and >= 3x the last print's own cost   // [DCT] hook: || dct_switch
        const double print_t0 = elapsed();
        bool do_save = it == o.iters || (o.save_every > 0 ? it % o.save_every == 0 : it == o.iters / 2);
        if (do_print || do_save) {
#ifdef NTC_CUDA
            if (o.cuda) { cu.download_model(D.lat.z.data(), D.mlp.p.data()); D.qes_refresh(); cu.mlp_stats(batch_loss, diff_std); }
            if (o.cuda && D.dct.live && D.dct.rate_es) { int h = 0, e = 0; cu.download_dct_rate(nullptr, nullptr, nullptr, nullptr, &h, &e); D.dct.rate_hits = (size_t)h; D.dct.rate_evals = (size_t)e; }   // [DCT] --dct-lambda: the hit count of the last device latent step (bits and trunc come from the host re-snap above, = the device's by check 7)
#endif
            if (o.write_ntcb && it == o.iters) { if (!ntcb_round()) return 1; }   // [NTCB] hook: the last iteration's weights become fp16 values; the lambda leaves the final decode in `recon`
            else   // [NTCB]
            decode_with(D.zdec(), recon);
            m = mse_of(target, recon, o.weights, wsum, o.orig_w, o.orig_h);
            mse = m.weighted;
            double ps = psnr_of(mse);
            best_psnr = std::max(best_psnr, ps);
            float lm, lsd, lmx; latent_stats(D.lat.lv[0], D.lat.level(0), lm, lsd, lmx);
            float l2m = 0, l2sd = 0, l2mx = 0;
            if (D.lat.lv.size() > 1) latent_stats(D.lat.lv[1], D.lat.level(1), l2m, l2sd, l2mx);
            // Bitrate and quality when the latent is actually quantized to 8 bits.
            g_ctx_stats = false; BitrateStats bs = bitrate_stats(D, o.qbits, zpost); g_ctx_stats = true;   // no context hashing on progress prints
            decode_with(zpost.data(), recon_q);
            mq = mse_of(target, recon_q, o.weights, wsum, o.orig_w, o.orig_h);
            double ps_q = psnr_of(mq.weighted);
            double sec = elapsed();
            printf("iter %6d  mse %.3f  psnr %6.2f dB  best %6.2f", it, mse, ps, best_psnr);
            if (T > 1) { printf(" | tex"); for (int t = 0; t < T; t++) printf(" %.2f", psnr_of(m.tex[t])); }
            printf(" | %s psnr %6.2f @ %.3f bpp (ent %.3f, est %.3f)", qlabel().c_str(), ps_q, bs.bpp_q8, bs.bpp_q8_ent, bs.bpp_est);   // est replaces ctx on the progress line (ctx is computed for the final block only)   // [DCT] hook: qlabel()
            if (T > 1) { printf(" (%.3f/tex) qtex", bs.bpp_q8 / T); for (int t = 0; t < T; t++) printf(" %.2f", psnr_of(mq.tex[t])); }
            if (D.dct.live) {   // [DCT] hook: the progress-line token group (12.B) and the churn since the last print (13.8)
                const DctStats S = dct_analyze(D.dct);   // [DCT]
                const size_t n0 = D.lat.lv[0].size();   // [DCT]
                size_t chs = 0, chz = 0;   // [DCT]
                if (dct_last_sym.size() == D.dct.sym.size()) { for (size_t i = 0; i < D.dct.sym.size(); i++) if (D.dct.sym[i] != dct_last_sym[i]) chs++; for (size_t i = 0; i < n0; i++) if (D.zq[i] != dct_last_zq[i]) chz++; }   // [DCT]
                else { chs = D.dct.sym.size(); chz = n0; }   // [DCT]
                dct_last_sym = D.dct.sym; dct_last_zq.assign(D.zq.begin(), D.zq.begin() + n0);   // [DCT]
                const double nb = (double)std::max<size_t>(D.dct.nblk(), 1);   // [DCT]
                printf(" | dct nz %.2f", S.nz_mean);   // [DCT] hook: the nz / lnz ... printf split so the --dct-lambda clauses sit next to the figures they qualify
                if (D.dct.rate_trunc) printf(" (pre-trunc %.2f)", (double)D.dct.last_nz_before / (double)std::max<size_t>(D.dct.nblk() * (size_t)D.dct.C, 1));   // [DCT] --dct-lambda part B: the nonzero ACs per (block, channel) before the last snap's truncation (nz is after it)
                printf(" lnz %.1f eob0 %.1f%% clamp %.2f%% bits/blk %.1f/%.1f/%.1f", S.lnz_mean, 100.0 * S.eob_only_frac,   // [DCT] hook: the chg clause moved below (the --dct-lambda proxy sits next to bits/blk)
                    n0 ? 100.0 * D.dct.last_clamped / n0 : 0.0, S.raw_total / nb, S.h0_total / nb, S.ctx_total / nb);   // [DCT]
                if (D.dct.lambda > 0.0f) {   // [DCT] --dct-lambda: proxy bits per block position summed over channels, and its ratio to the h0 AC tokens (run + mag + sign + eob) it prices
                    double pb; dct_proxy_totals(D.dct, pb, nullptr); const double h0ac = S.h0[1] + S.h0[2] + S.h0[3] + S.h0[4];   // [DCT]
                    if (h0ac > 0) printf(" proxy %.1f (%.2fx h0ac)", pb / nb, pb / h0ac); else printf(" proxy %.1f (h0ac 0)", pb / nb);   // [DCT]
                }   // [DCT]
                printf(" chg sym %zu zq %zu", chs, chz);   // [DCT]
                if (D.dct.lambda > 0.0f) {   // [DCT] --dct-lambda: part B's count in the last snap over the pre-truncation nonzero ACs, part A's nonzero-difference fraction in the last latent step; `off` for the part --dct-rate left out
                    if (D.dct.rate_trunc) printf(" trunc %zu/%zu", D.dct.last_truncated, D.dct.last_nz_before); else printf(" trunc off");   // [DCT]
                    if (D.dct.rate_es) printf(" hits %.1f%%", D.dct.rate_evals ? 100.0 * D.dct.rate_hits / D.dct.rate_evals : 0.0); else printf(" hits off");   // [DCT]
                }   // [DCT]
            }   // [DCT]
            printf(" | mlp batch %.5f %s %.2e | lat mean %+.3f sd %.3f max %.2f", batch_loss, mlp_stat, diff_std, lm, lsd, lmx);
            if (D.lat.lv.size() > 1) printf(" | lat2 mean %+.3f sd %.3f max %.2f", l2m, l2sd, l2mx);
            if (o.qat > 0) {   // invariant: level 0 is always on its grid
                size_t off_grid = 0;
                for (size_t i = 0; i < D.lat.lv[0].size(); i++) if (qat_snap(D.lat.z[i], o.qat_ch[i % D.lat.lv[0].C]) != D.lat.z[i]) off_grid++;
                if (off_grid) printf(" | WARNING %zu level-0 values off-grid", off_grid);
            }
            printf(" | %.1fs (%.2f it/s)\n", sec, it / sec);
            if (D.dct.live && o.dct_stats && it != o.iters) dct_print_stats(D.dct, dct_analyze(D.dct), (double)D.W * D.H, D.lat.lv[0], D.lat.level(0), D.zdec());   // [DCT] hook: --dct-stats (the final block is printed once before done:)
            fflush(stdout);
        }
        if (do_print) { last_print_time = elapsed(); print_interval = std::max(1.0, 3.0 * (last_print_time - print_t0)); }   // the once-per-second cadence counts from the END of the print work: on a 6 Mpixel image the download + full decode + statistics take longer than a second, and stamping before them made every iteration print (the GPU idle, one core busy)
        if (do_save) {
            char name[64];
            for (int t = 0; t < T; t++) save_png(tex_name("recon", t, it), recon, t);
            snprintf(name, sizeof(name), "/latent_%06d.png", it);
            save_latent_png(o.outdir + name, D.lat.lv[0], D.lat.level(0));
            if (D.dct.live) {   // [DCT] hook: the per-block map, the snapped plane and the residual PNGs with every snapshot (12.C)
                const DctStats S = dct_analyze(D.dct);   // [DCT]
                Image mp;   // [DCT]
                dct_map_image(D.dct, S, o.dct_map == "all" ? "nz" : o.dct_map, D.W, D.H, mp);   // [DCT]
                snprintf(name, sizeof(name), "/dct_map_%06d.png", it); save_png(o.outdir + name, mp);   // [DCT]
                if (o.dct_map == "all" && it == o.iters)   // [DCT]
                    for (const char* k : { "code", "bits", "lnz", "dconly" }) { dct_map_image(D.dct, S, k, D.W, D.H, mp); snprintf(name, sizeof(name), "/dct_map_%s_%06d.png", k, it); save_png(o.outdir + name, mp); }   // [DCT]
                snprintf(name, sizeof(name), "/latent_q_%06d.png", it); save_latent_png(o.outdir + name, D.lat.lv[0], D.zdec());   // [DCT]
                snprintf(name, sizeof(name), "/dct_resid_%06d.png", it); dct_save_resid_png(o.outdir + name, D.lat.lv[0], D.lat.level(0), D.zdec());   // [DCT]
            }   // [DCT]
            if (D.lat.lv.size() > 1) {
                snprintf(name, sizeof(name), "/latent2_%06d.png", it);
                save_latent_png(o.outdir + name, D.lat.lv[1], D.lat.level(1));
            }
            if (D.lat.lv.size() > 2) {
                snprintf(name, sizeof(name), "/latent3_%06d.png", it);
                save_latent_png(o.outdir + name, D.lat.lv[2], D.lat.level(2));
            }
            save_model(o.outdir + "/model.bin", D);
        }
    }
    if (o.write_ntcb && o.iters == 0 && !ntcb_round()) return 1;   // [NTCB] hook: an evaluation run (--load --iters 0) rounds here, before --resave and the final block
    // --load with --iters 0 writes the loaded model back in the current file layout (a format conversion).
    if (o.resave && o.iters == 0 && !o.load.empty()) save_model(o.outdir + "/model.bin", D);   // --resave: rewrite the loaded model in the current format
    if (o.resave && o.iters == 0 && !o.load.empty() && D.dct.live)   // [DCT] hook: say what --resave wrote
        printf("resave   : wrote %s/model.bin as v14 (level 0 DCT-coded, q %s, DC step %d%s; %s)\n", o.outdir.c_str(), dct_spec(D.dct).c_str(), D.dct.dc_step, D.dct.deadzone ? "" : ", --dct-deadzone 0",   // [DCT] hook: v14, the deadzone clause
            dct_from_file ? "symbols and codes from the loaded file" : "codes fitted and symbols snapped in this run from the loaded continuous level 0");   // [DCT] hook: "loaded file" (v13 or v14)
    int ntcb_rc = 0;   // [NTCB] 1 after a file: MISMATCH (reported after done:)
    {
        BitrateStats bs = bitrate_stats(D, o.qbits, zpost);
        decode_with(zpost.data(), recon_q);
        mq = mse_of(target, recon_q, o.weights, wsum, o.orig_w, o.orig_h);
        double ps_q = psnr_of(mq.weighted);
        if (D.dct.live) {   // [DCT] hook: the final statistics block (12.B), the end-of-run probe re-check (13.9), the final maps (12.C)
            const DctStats S = dct_analyze(D.dct);   // [DCT]
            dct_print_stats(D.dct, S, (double)D.W * D.H, D.lat.lv[0], D.lat.level(0), D.zdec());   // [DCT]
            {   // [DCT]
                std::vector<uint8_t> would; D.dct_probe(would);   // [DCT]
                size_t hist[16] = { 0 }, m1 = 0, m2 = 0;   // [DCT]
                for (size_t i = 0; i < would.size(); i++) { hist[would[i]]++; const int d = std::abs((int)would[i] - (int)D.dct.code[i]); if (d >= 1) m1++; if (d >= 2) m2++; }   // [DCT]
                printf("dct reprobe (end of run, not applied) hist k0..k15");   // [DCT]
                for (int k = 0; k < 16; k++) printf(" %zu", hist[k]);   // [DCT]
                printf(" | codes that would move >= 1: %.1f%%, >= 2: %.1f%% (of %zu)\n", would.empty() ? 0.0 : 100.0 * m1 / would.size(), would.empty() ? 0.0 : 100.0 * m2 / would.size(), would.size());   // [DCT]
            }   // [DCT]
            Image mp;   // [DCT]
            dct_map_image(D.dct, S, o.dct_map == "all" ? "nz" : o.dct_map, D.W, D.H, mp);   // [DCT]
            save_png(o.outdir + "/dct_map.png", mp);   // [DCT]
            if (o.dct_map == "all") for (const char* k : { "code", "bits", "lnz", "dconly" }) { Image m2; dct_map_image(D.dct, S, k, D.W, D.H, m2); save_png(o.outdir + "/dct_map_" + k + ".png", m2); }   // [DCT]
            Image b, sbs; slice_texture(recon_q, 0, b);   // dct_side_by_side.png: recon_q_final (texture 0) | the map of every channel (W x (1 + C) wide) [DCT]
            sbs.w = b.w + mp.w; sbs.h = b.h; sbs.nc = 3; sbs.rgb.resize((size_t)sbs.w * sbs.h * 3);   // [DCT] hook: mp.w = W * C (was D.W, channel 0 only)
            for (int y = 0; y < b.h; y++) {   // [DCT]
                memcpy(&sbs.rgb[((size_t)y * sbs.w) * 3], &b.rgb[((size_t)y * b.w) * 3], (size_t)b.w * 3 * sizeof(float));   // [DCT]
                memcpy(&sbs.rgb[((size_t)y * sbs.w + b.w) * 3], &mp.rgb[((size_t)y * mp.w) * 3], (size_t)mp.w * 3 * sizeof(float));   // [DCT] hook: mp.w (was D.W)
            }   // [DCT]
            save_png(o.outdir + "/dct_side_by_side.png", sbs);   // [DCT]
        }   // [DCT]
        for (int t = 0; t < T; t++) {
            save_png_src(tex_name("recon_q_final", t), recon_q, o.orig_w, o.orig_h, t);   // source extent, = ntcb_decoded.png's size
            // Target | quantized-latent reconstruction, so the picture matches the quoted bitrate.
            Image a, b, sbs;
            slice_texture(target, t, a); slice_texture(recon_q, t, b);
            sbs.w = a.w * 2; sbs.h = a.h; sbs.nc = 3;
            sbs.rgb.resize((size_t)sbs.w * sbs.h * 3);
            for (int y = 0; y < a.h; y++) {
                memcpy(&sbs.rgb[((size_t)y * sbs.w) * 3], &a.rgb[((size_t)y * a.w) * 3], (size_t)a.w * 3 * sizeof(float));
                memcpy(&sbs.rgb[((size_t)y * sbs.w + a.w) * 3], &b.rgb[((size_t)y * a.w) * 3], (size_t)a.w * 3 * sizeof(float));
            }
            save_png(tex_name("side_by_side", t), sbs);
        }
        if (o.write_ntcb) {   // [NTCB] hook: write the container from the same state the PNGs and the done: line describe, and reconcile it with bitrate_stats
            NtcbReport r;   // [NTCB]
            const std::string path = o.outdir + "/model.ntcb";   // [NTCB]
            if (!ntcb_write(path, D, bs, o, zpost.data(), r)) return 1;   // [NTCB]
            const double np = (double)D.W * D.H;   // [NTCB]
            printf("file: %s %zu bytes = %.3f bpp", path.c_str(), r.file_bytes, 8.0 * r.file_bytes / np);   // [NTCB]
            if (T > 1) printf(" (%.3f/tex)", 8.0 * r.file_bytes / np / T);   // [NTCB]
            printf(" (simulator raw %.3f bpp", bs.bpp_q8);   // [NTCB]
            if (T > 1) printf(" (%.3f/tex)", bs.bpp_q8 / T);   // [NTCB]
            printf("); content %.0f bits = simulator %.0f - %.0f side-info bits (stored in the header) [%s]; container header %d + section headers %d + sync %.0f + padding %.0f bits; bpp over the %dx%d decode size, as every bpp in this log; self-check OK\n",   // [NTCB]
                r.content_bits, r.sim_total, r.sim_side, r.match ? "OK" : "MISMATCH", 8 * r.H, 64 * r.nsections, r.sync_bits, r.pad_bits, D.W, D.H);   // [NTCB]
            if (!r.match) { printf("file: MISMATCH: the container's content bits (%.0f) differ from the simulator's raw latent + fp16 mlp bits (%.0f): a writer or simulator bug\n", r.content_bits, r.sim_total - r.sim_side); ntcb_rc = 1; }   // [NTCB]
        }   // [NTCB]
        printf("done: final psnr %.2f dB (best %.2f)", psnr_of(mse), best_psnr);
        if (T > 1) { printf(" tex"); for (int t = 0; t < T; t++) printf(" %.2f", psnr_of(m.tex[t])); }
        printf(" at fp32 %.3f bpp", bs.bpp_fp32);
        if (T > 1) printf(" (%.3f/tex)", bs.bpp_fp32 / T);
        if (D.dct.live) {   // [DCT] hook
            std::string lamcl;   // [DCT] --dct-lambda: ", lambda 80" / ", lambda 80 es-only" / ", lambda 80 trunc-only" inside the simulator bracket; empty at lambda 0
            if (D.dct.lambda > 0.0f) { char lb[64]; snprintf(lb, sizeof lb, ", lambda %g%s lo %d/16", D.dct.lambda, D.dct.rate_es && D.dct.rate_trunc ? "" : (D.dct.rate_es ? " es-only" : " trunc-only"), D.dct.lo_w16); lamcl = lb; }   // [DCT] --dct-lambda-lo
            printf(" | --dct-q %s%s level 0 (bit simulator%s)%s + fp16 mlp: psnr %.2f dB", dct_spec(D.dct).c_str(), D.dct.deadzone ? "" : " dz0", lamcl.c_str(), o.qes ? (" + --qes " + qes_spec(o.qes_ch) + " latent").c_str() : "", ps_q);   // [DCT] hook: the %s after `simulator` = lamcl; the %s after the q list = " dz0" with --dct-deadzone 0
        }   // [DCT]
        else if (o.qes) printf(" | --qes %s latent%s + fp16 mlp: psnr %.2f dB", qes_spec(o.qes_ch).c_str(), D.qes_live ? "" : " (not yet quantized)", ps_q);   // [DCT]
        else printf(" | %d-bit latent + fp16 mlp: psnr %.2f dB", o.qbits, ps_q);
        if (T > 1) { printf(" qtex"); for (int t = 0; t < T; t++) printf(" %.2f", psnr_of(mq.tex[t])); }
        printf(" at %.3f bpp raw, %.3f bpp entropy-coded, %.3f bpp est (order-0 level 0 + DPCM latent), %.3f bpp with an (up, left) context on level 0", bs.bpp_q8, bs.bpp_q8_ent, bs.bpp_est, bs.bpp_q8_ctx);   // [EST] the est figure
        if (T > 1) printf(" (%.3f, %.3f, %.3f /tex)", bs.bpp_q8 / T, bs.bpp_q8_ent / T, bs.bpp_q8_ctx / T);
        printf(" [level 0 alone: %.3f]", bs.bpp_ctx_level0);
        if (D.dct.live) {   // [DCT] hook: the separate dct bracket (raw / h0 / ctx of level 0's symbols incl. the codes; code bpp raw / ctx)
            const DctStats S = dct_analyze(D.dct); const double np = (double)D.W * D.H;   // [DCT]
            printf(" [dct: raw %.3f h0 %.3f ctx %.3f code raw/ctx %.3f/%.3f bpp; nz %.2f/blk eob0 %.1f%%]", (S.raw_total - S.header) / np, (S.h0_total - S.header) / np, (S.ctx_total - S.header) / np,   // [DCT]
                S.code_raw / np, S.code_ctx / np, S.nz_mean, 100.0 * S.eob_only_frac);   // [DCT]
        }   // [DCT]
        printf(" | %.1fs\n", elapsed());
        if (o.eval_filter) {
            decode_with(D.zdec(), recon);   // the texel decode is the reference the filters are applied to
            eval_filter(D, target, recon, o, T);
        }
    }
    if (ntcb_rc) return ntcb_rc;   // [NTCB] hook: a file: MISMATCH is a failure
    return 0;
}
