// CUDA backend for ntc: device-resident ES training of the latent block format.
// See CUDA_PLAN_B.md. The CPU implementation in main.cpp is the reference; every
// device function here mirrors a host function by name and is expected to agree
// with it to float rounding (the host side computes in the same order; the only
// systematic difference is FMA contraction).
//
// Structure: a "texture set" of ntex textures sharing one decoder format. Kernels
// index the texture with blockIdx.y and address MLP weights through a weight-set
// index, so a universal decoder trained over many textures needs no kernel change.
// v1 runs with one texture and one weight set.
//
// DEPENDENCY: keep this file in sync with main.cpp. There is no shared source between
// the two except ntc_noise.h; each device function here (dev_tap, dev_sample,
// dev_pos_encode, dev_features, dev_mlp_forward, dev_activate, the ES / FD / Adam /
// latent-gather / qat-search / snap kernels) is a hand-written copy of the host function of
// the same name and must keep its floating-point evaluation order. main.cpp carries the
// matching note at its top listing what it owns. After any change on either side run
// `ntc <image> --cuda --cuda-check ...` with the options in use (README "Regression
// testing"); all six comparisons must print PASS. New positional kinds, latent levels,
// Options that reach ModelDesc, or model-file fields (v12: per-level --qes bits and ranges)
// are not done until they exist here too (or are rejected with a reason in Trainer::init).
// The standalone CPU decoder ntc_decode.cpp mirrors the decode path only.
//
// Snapped decode buffer rule (--qes, main.cpp Decoder::zdec()): d_z is the fp32 shadow that
// Adam updates; d_zdec is what every decode-type kernel reads (k_decode, k_batch_features,
// k_lat_pair, k_qat_search). Without a --qes level d_zdec aliases d_z and nothing here changes.
// With one, d_zdec is the persistent d_zq rebuilt by k_snap (the mirror of qes_snap_level)
// after every Adam step on the latent, after upload_model and after set_qes; the index is the
// host's expression and the grid values are looked up from the host-computed table, never
// recomputed (the commit-4941c22 rule for the --qat grid). k_qat_search writes level 0 of
// d_zdec and its result is copied back to d_z when the two buffers differ.
// [DCT]
// [DCT] The experimental --dct-q level 0 (main.cpp "DCT" region) follows the same rule: d_zq exists, level 0
// [DCT] of d_zdec is rebuilt by k_dct_snap (= dct_snap_block: clamp the shadow, forward DCT, dead-zone quantizer,
// [DCT] inverse DCT) after every latent step, by k_dct_recon (= dct_recon_block, from the uploaded symbols) at
// [DCT] set_dct, and only clamped (k_dct_clamp0 = dct_clamp_level0, shadow and copy) before the codes are live.
// [DCT] The basis and the integer step tables are copied from the host, the scale codes are host-fitted; every
// [DCT] product and sum is __fmul_rn / __fadd_rn so no multiply-add pair can be contracted. Check 7 of --cuda-check
// [DCT] compares symbols, plane and codes exactly. Off by default (ModelDesc::dct == 0): no buffer, no kernel.
// [DCT] --dct-lambda (DCT_RATE_PLAN.md): dev_dct_c_run / dev_dct_c_mag / dev_dct_block_bits, dev_dct_truncate_block and
// [DCT] dev_dct_quantize_block mirror the host's integer token cost model and snap-time truncation (its three float
// [DCT] operations written with _rn intrinsics); k_dct_rate_pair / k_dct_rate_gather mirror dct_rate_block and the rate
// [DCT] pass of LatentTrainer::step (the perturbed shadow d_z, not the snapped copy). lam_t16 and rate_scale arrive as
// [DCT] host floats in DctDesc; the zigzag order is copied. The `dct rate` line of --cuda-check compares the integers.
#include "ntc_cuda.h"
#include "ntc_noise.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <algorithm>

namespace ntc_cuda {

#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

// ---------------------------------------------------------------- device-side descriptors
struct DevLevel {
    int W, H, C, nearest; size_t off;
    const int *xb, *xe, *yb, *ye;   // nearest levels: pixel range per texel column / row
};
struct DevTex {
    int W, H, nout, T;
    int nlev; DevLevel lv[MAX_LEVELS];
    size_t z_off, tgt_off, pix_off;   // slices into the set's flat arrays
    float cw[MAX_OUT]; float wsum; float inv_px;   // inv_px = 1 / (3 wsum W H), as LatentTrainer
};
struct DevMlp {
    int nin, nout, nl; int w[MAX_HIDDEN]; float leak; int clamp;
};
struct DevPos { int n; int kind[MAX_POS]; };
struct DevQat { int bits[MAX_CH]; };
static const int MAX_QES_GRID = MAX_LEVELS * MAX_QES_CH * 4096;   // largest grid table (12 bits, every channel)

static const int MAX_TEX = 8;
__constant__ DevTex c_tex[MAX_TEX];
__constant__ DevMlp c_mlp;
__constant__ DevPos c_pos;
__constant__ DevQat c_qat;
__constant__ float c_qat_grid[9][257];   // copied from the host QAT_GRID: the single source of on-grid bit patterns
__constant__ QesDesc c_qes;              // --qes ranges (host-fitted); the grid values live in global memory (d_qgrid)


// ---------------------------------------------------------------- mirrors of main.cpp
struct Tap { int x0, x1, y0, y1; float fx, fy; };

// mirror of bilinear_tap
__device__ __forceinline__ Tap dev_tap(const DevLevel& L, float u, float v) {
    Tap t;
    if (L.nearest) {
        float x = u * L.W, y = v * L.H;
        int ix = (int)floorf(x), iy = (int)floorf(y);
        t.fx = x - ix; t.fy = y - iy;
        t.x0 = t.x1 = max(0, min(L.W - 1, ix));
        t.y0 = t.y1 = max(0, min(L.H - 1, iy));
        return t;
    }
    float x = u * L.W - 0.5f, y = v * L.H - 0.5f;
    int x0 = (int)floorf(x), y0 = (int)floorf(y);
    t.fx = x - x0; t.fy = y - y0;
    t.x0 = max(0, min(L.W - 1, x0)); t.x1 = max(0, min(L.W - 1, x0 + 1));
    t.y0 = max(0, min(L.H - 1, y0)); t.y1 = max(0, min(L.H - 1, y0 + 1));
    return t;
}

// mirror of sample_latent
__device__ __forceinline__ void dev_sample(const DevLevel& L, const float* z, const Tap& t, float* out) {
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
    for (int k = 0; k < L.C; k++) out[k] = w00 * a[k] + w10 * b[k] + w01 * c[k] + w11 * d[k];
}

// mirror of activate (leaky ReLU, the only hidden activation)
__device__ __forceinline__ float dev_activate(float x, float leak) { return x > 0 ? x : leak * x; }

// mirror of PosEnc::encode
__device__ __forceinline__ int dev_pos_encode(float u, float v, const Tap& t, const Tap* t1, float* f) {
    int k = 0;
    for (int q = 0; q < c_pos.n; q++) {
        const int kind = c_pos.kind[q];
        switch (kind) {
        case PK_UV: f[k++] = u * 2.0f - 1.0f; f[k++] = v * 2.0f - 1.0f; break;
        case PK_LV1LOCAL: f[k++] = t1->fx * 2.0f - 1.0f; f[k++] = t1->fy * 2.0f - 1.0f; break;
        default: break;
        }
    }
    return k;
}

// mirror of Decoder::features; z points at the texture's flat latent (level 0 at offset 0)
__device__ __forceinline__ void dev_features(const DevTex& tex, const float* z, int px, int py, float* f) {
    float u = (px + 0.5f) / tex.W, v = (py + 0.5f) / tex.H;
    const DevLevel& L0 = tex.lv[0];
    Tap t = dev_tap(L0, u, v);
    dev_sample(L0, z, t, f);
    int k = L0.C;
    Tap t1;
    for (int l = 1; l < tex.nlev; l++) {
        const DevLevel& L = tex.lv[l];
        Tap tl = dev_tap(L, u, v);
        if (l == 1) t1 = tl;
        dev_sample(L, z + L.off, tl, f + k);
        k += L.C;
    }
    dev_pos_encode(u, v, t, tex.nlev > 1 ? &t1 : nullptr, f + k);
}

// mirror of mlp_forward. The input/activation vector `a` is indexed only with
// compile-time indices (the inner loop is fully unrolled and predicated by the runtime
// width), so it stays in registers; the per-layer outputs `b` are written with a
// runtime index and live in L1-backed local memory. Weights come from shared memory;
// every lane reads the same address at the same time (broadcast). MAXW bounds the
// input and hidden widths.
template <int MAXW>
__device__ __forceinline__ void dev_mlp_forward(const float* __restrict__ w, const float* in, float* out) {
    float a[MAXW], b[MAXW];
#pragma unroll
    for (int i = 0; i < MAXW; i++) a[i] = (i < c_mlp.nin) ? in[i] : 0.0f;
    int ncur = c_mlp.nin;
    const float* p = w;
    for (int l = 0; l < c_mlp.nl; l++) {
        const int nh = c_mlp.w[l];
        const float* bias = p + nh * ncur;
        for (int j = 0; j < nh; j++) {
            float s = bias[j];
            const float* wr = p + j * ncur;
#pragma unroll
            for (int i = 0; i < MAXW; i++) if (i < ncur) s += wr[i] * a[i];
            b[j] = dev_activate(s, c_mlp.leak);
        }
        p = bias + nh; ncur = nh;
#pragma unroll
        for (int i = 0; i < MAXW; i++) a[i] = (i < ncur) ? b[i] : 0.0f;
    }
    const float* bias = p + c_mlp.nout * ncur;
    for (int j = 0; j < c_mlp.nout; j++) {
        float s = bias[j];
        const float* wr = p + j * ncur;
#pragma unroll
        for (int i = 0; i < MAXW; i++) if (i < ncur) s += wr[i] * a[i];
        if (c_mlp.clamp) out[j] = fminf(1.0f, fmaxf(0.0f, s + 0.5f));
        else out[j] = 1.0f / (1.0f + expf(-s));
    }
}

// Weighted per-pixel error terms, added one float term at a time into a double, as
// loss_subset / decode_err do (`s += cw[c] * (d * d)` with float products).
__device__ __forceinline__ double dev_pixel_err_d(const DevTex& tex, const float* out, const float* tgt) {
    double s = 0;
    for (int c = 0; c < tex.nout; c++) { float d = out[c] - tgt[c]; s += (double)(tex.cw[c] * (d * d)); }
    return s;
}
__device__ __forceinline__ float dev_pixel_err_f(const DevTex& tex, const float* out, const float* tgt) {
    float e = 0;
    for (int c = 0; c < tex.nout; c++) { float d = out[c] - tgt[c]; e += tex.cw[c] * (d * d); }
    return e;
}

// Fixed-order block reduction of doubles: warp shuffle tree, then warp sums in index order.
template <int BLOCK>
__device__ __forceinline__ double block_sum_double(double v) {
    __shared__ double warp_s[BLOCK / 32];
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) warp_s[wid] = v;
    __syncthreads();
    double r = 0;
    if (threadIdx.x == 0) for (int i = 0; i < BLOCK / 32; i++) r += warp_s[i];
    __syncthreads();
    return r;   // valid in thread 0
}

__device__ __forceinline__ void load_weights_shared(float* ws, const float* src, int P) {
    for (int k = threadIdx.x; k < P; k += blockDim.x) ws[k] = src[k];
    __syncthreads();
}

// ---------------------------------------------------------------- kernels
static const int TPB = 128;   // threads per block for the MLP kernels

// Full decode of one texture (= Decoder::decode_full without deblocking).
template <int MAXW>
__global__ void __launch_bounds__(TPB) k_decode(const float* __restrict__ w, const float* __restrict__ zset, float* __restrict__ img, int P) {
    extern __shared__ float ws[];
    load_weights_shared(ws, w, P);
    const DevTex& tex = c_tex[blockIdx.y];
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= tex.W * tex.H) return;
    int px = p % tex.W, py = p / tex.W;
    float f[MAXW]; dev_features(tex, zset + tex.z_off, px, py, f);
    float o[MAX_OUT]; dev_mlp_forward<MAXW>(ws, f, o);
    float* dst = img + (tex.pix_off + p) * tex.nout;
    for (int c = 0; c < tex.nout; c++) dst[c] = o[c];
}

// The minibatch of one MLP step (= MlpTrainer::draw_batch + the feature part of
// loss_subset): slot s -> (texture, pixel), its MLP input row and its target row.
template <int MAXW>
__global__ void k_batch_features(uint64_t seed, int step, int M, int ntex,
                                 const float* __restrict__ zset, const float* __restrict__ tgt,
                                 int* __restrict__ bidx, float* __restrict__ feat, float* __restrict__ btgt, float* __restrict__ bnrm) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= M) return;
    int ti = 0;
    if (ntex > 1) ti = (int)ntc_noise_uniform(ntc_noise_u32(seed, NS_BATCH_TEX, step, 0, s), ntex);
    const DevTex& tex = c_tex[ti];
    const int idx = (int)ntc_noise_uniform(ntc_noise_u32(seed, NS_BATCH_PIX, step, 0, s), (uint32_t)(tex.W * tex.H));
    bidx[s] = (ti << 28) | idx;
    dev_features(tex, zset + tex.z_off, idx % tex.W, idx / tex.W, feat + (size_t)s * c_mlp.nin);
    const float* t = tgt + (tex.tgt_off + (size_t)idx) * tex.nout;
    float* bt = btgt + (size_t)s * MAX_OUT;
    for (int c = 0; c < tex.nout; c++) bt[c] = t[c];
    bnrm[s] = 1.0f / (3.0f * tex.wsum);   // per-slot normalization (one texture: constant)
}

// Loss of 2N perturbed weight sets over the minibatch (= the loss_subset calls of
// MlpTrainer::step). grid = (chunks, 2N); block (set, chunk) evaluates PPT pixels per
// thread with the set's weights w0 +- sigma * eps in shared memory.
static const int PPT = 2;   // pixels per thread in k_mlp_es_eval
template <int MAXW>
__global__ void __launch_bounds__(TPB) k_mlp_es_eval(uint64_t seed, int step, float sigma, int M, int P,
                                                     const float* __restrict__ w0, const float* __restrict__ feat,
                                                     const float* __restrict__ btgt, const int* __restrict__ bidx,
                                                     double* __restrict__ part, int chunks) {
    extern __shared__ float ws[];
    const int set = blockIdx.y, pair = set >> 1;
    const float sg = (set & 1) ? -sigma : sigma;
    for (int k = threadIdx.x; k < P; k += blockDim.x) ws[k] = w0[k] + sg * ntc_gauss(seed, NS_MLP, step, pair, k);
    __syncthreads();
    double acc = 0;
    for (int r = 0; r < PPT; r++) {
        int s = (blockIdx.x * PPT + r) * blockDim.x + threadIdx.x;
        if (s < M) {
            const DevTex& tex = c_tex[bidx[s] >> 28];
            float o[MAX_OUT]; dev_mlp_forward<MAXW>(ws, feat + (size_t)s * c_mlp.nin, o);
            acc += dev_pixel_err_d(tex, o, btgt + (size_t)s * MAX_OUT);
        }
    }
    double bs = block_sum_double<TPB>(acc);
    if (threadIdx.x == 0) part[(size_t)set * chunks + blockIdx.x] = bs;
}

// Per-pair loss differences (= dl, lsum in MlpTrainer::step). One thread per pair.
__global__ void k_mlp_es_finish(const double* __restrict__ part, int chunks, int N, double norm, double* __restrict__ dl, double* __restrict__ lsum) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    double lp = 0, lm = 0;
    for (int j = 0; j < chunks; j++) { lp += part[(size_t)(2 * i) * chunks + j]; lm += part[(size_t)(2 * i + 1) * chunks + j]; }
    lp /= norm; lm /= norm;
    dl[i] = lp - lm; lsum[i] = 0.5 * (lp + lm);
}

// ES gradient (= the grad accumulation of MlpTrainer::step). One thread per weight.
__global__ void k_mlp_es_grad(uint64_t seed, int step, float scale, int N, int P, const double* __restrict__ dl, float* __restrict__ grad) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= P) return;
    float g = 0;
    for (int i = 0; i < N; i++) { float w = (float)dl[i] * scale; g += w * ntc_gauss(seed, NS_MLP, step, i, k); }
    grad[k] = g;
}

// Central differences (= MlpTrainer::step_fd): one block per weight, both signs.
template <int MAXW>
__global__ void __launch_bounds__(TPB) k_fd_eval(int j0, float h, int M, int P, const float* __restrict__ w0,
                                                 const float* __restrict__ feat, const float* __restrict__ btgt, const int* __restrict__ bidx,
                                                 double norm, double* __restrict__ dl, double* __restrict__ hh) {
    extern __shared__ float ws[];
    load_weights_shared(ws, w0, P);
    const int j = j0 + blockIdx.x;
    const float orig = ws[j], wp = orig + h, wm = orig - h;   // float-realized steps, as on the CPU
    double sums[2];
    for (int sgn = 0; sgn < 2; sgn++) {
        __syncthreads();
        if (threadIdx.x == 0) ws[j] = sgn ? wm : wp;
        __syncthreads();
        double acc = 0;
        for (int s = threadIdx.x; s < M; s += blockDim.x) {
            const DevTex& tex = c_tex[bidx[s] >> 28];
            float o[MAX_OUT]; dev_mlp_forward<MAXW>(ws, feat + (size_t)s * c_mlp.nin, o);
            acc += dev_pixel_err_d(tex, o, btgt + (size_t)s * MAX_OUT);
        }
        sums[sgn] = block_sum_double<TPB>(acc);
    }
    if (threadIdx.x == 0) { dl[j] = sums[0] / norm - sums[1] / norm; hh[j] = (double)wp - (double)wm; }
}

__global__ void k_fd_grad(int P, const double* __restrict__ dl, const double* __restrict__ hh, float* __restrict__ grad) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < P) grad[j] = (float)(dl[j] / hh[j]);
}

// Adam (= Adam::step): theta -= lr * mhat / (sqrt(vhat) + eps)
__global__ void k_adam(float* __restrict__ theta, const float* __restrict__ g, float* __restrict__ m, float* __restrict__ v, int n, float lr, float c1, float c2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
    m[i] = b1 * m[i] + (1 - b1) * g[i];
    v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i];
    float mh = m[i] / c1, vh = v[i] / c2;
    theta[i] -= lr * mh / (sqrtf(vh) + eps);
}

// Latent ES, one antithetic pair (= the perturb + decode_err part of LatentTrainer::step),
// fused per pixel: the pixel's features are built once, then each active nearest level's
// texel values are perturbed by +-sigma * eps directly in the feature vector (bit-identical
// to perturbing z, since sampling is a copy), and the per-pixel loss difference
// (e+ - e-) * inv_px is written to the pixel's own slot.
template <int MAXW>
__global__ void __launch_bounds__(TPB) k_lat_pair(uint64_t seed, int step, int pair, unsigned active_mask, float sg,
                                                  const float* __restrict__ w, const float* __restrict__ zset, const float* __restrict__ tgt,
                                                  float* __restrict__ dpix, int P) {
    extern __shared__ float ws[];
    load_weights_shared(ws, w, P);
    const DevTex& tex = c_tex[blockIdx.y];
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= tex.W * tex.H) return;
    int px = p % tex.W, py = p / tex.W;
    float fp[MAXW]; dev_features(tex, zset + tex.z_off, px, py, fp);
    float fm[MAXW];
#pragma unroll
    for (int i = 0; i < MAXW; i++) fm[i] = fp[i];
    float u = (px + 0.5f) / tex.W, v = (py + 0.5f) / tex.H;
    int slot = 0;
    for (int l = 0; l < tex.nlev; l++) {
        const DevLevel& L = tex.lv[l];
        if (active_mask & (1u << l)) {
            Tap t = dev_tap(L, u, v);
            if (L.nearest) {
                size_t base = tex.z_off + L.off + ((size_t)t.y0 * L.W + t.x0) * L.C;
                for (int c = 0; c < L.C; c++) {
                    float e = ntc_gauss(seed, NS_LAT, step, pair, base + c);
                    fp[slot + c] += sg * e;
                    fm[slot + c] -= sg * e;
                }
            } else {
                // Bilinear sampling is linear, so sample(z + sg*eps) = sample(z) + sg*sample(eps):
                // the four taps' noise (keyed by the clamped texel, so a clamped duplicate tap
                // reuses the same value, exactly as the CPU perturbs each texel once).
                const size_t lb = tex.z_off + L.off;
                const size_t ia = lb + ((size_t)t.y0 * L.W + t.x0) * L.C, ib = lb + ((size_t)t.y0 * L.W + t.x1) * L.C;
                const size_t ic = lb + ((size_t)t.y1 * L.W + t.x0) * L.C, id = lb + ((size_t)t.y1 * L.W + t.x1) * L.C;
                const float w00 = (1 - t.fx) * (1 - t.fy), w10 = t.fx * (1 - t.fy), w01 = (1 - t.fx) * t.fy, w11 = t.fx * t.fy;
                for (int c = 0; c < L.C; c++) {
                    float e = w00 * ntc_gauss(seed, NS_LAT, step, pair, ia + c) + w10 * ntc_gauss(seed, NS_LAT, step, pair, ib + c)
                            + w01 * ntc_gauss(seed, NS_LAT, step, pair, ic + c) + w11 * ntc_gauss(seed, NS_LAT, step, pair, id + c);
                    fp[slot + c] += sg * e;
                    fm[slot + c] -= sg * e;
                }
            }
        }
        slot += L.C;
    }
    const float* tg = tgt + (tex.tgt_off + (size_t)p) * tex.nout;
    float op[MAX_OUT]; dev_mlp_forward<MAXW>(ws, fp, op);
    float ep = dev_pixel_err_f(tex, op, tg);
    float om[MAX_OUT]; dev_mlp_forward<MAXW>(ws, fm, om);
    float em = dev_pixel_err_f(tex, om, tg);
    dpix[tex.pix_off + p] = (ep - em) * tex.inv_px;
}

// Footprint attribution in gather form (= the scatter + grad accumulation of
// LatentTrainer::step for a nearest level): one thread per texel sums its cell's
// per-pixel differences in raster order and adds w * eps to its channels' gradient.
__global__ void k_lat_gather(uint64_t seed, int step, int pair, int level, float scale, const float* __restrict__ dpix, float* __restrict__ grad) {
    const DevTex& tex = c_tex[blockIdx.y];
    const DevLevel& L = tex.lv[level];
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= L.W * L.H) return;
    int tx = t % L.W, ty = t / L.W;
    float d = 0;
    if (L.nearest) {
        for (int py = L.yb[ty]; py < L.ye[ty]; py++)
            for (int px = L.xb[tx]; px < L.xe[tx]; px++)
                d += dpix[tex.pix_off + (size_t)py * tex.W + px];
    } else {
        // Bilinear: a pixel reads texel (tx,ty) iff its tap's clamped x0 or x1 is tx and
        // clamped y0 or y1 is ty, credited once per distinct texel (the CPU scatter's
        // `if (x1 != x0)` guards). Scan a conservative pixel rectangle around the texel in
        // raster order and test each pixel with the same tap the decode uses.
        const float rx = (float)tex.W / L.W, ry = (float)tex.H / L.H;
        const int px0 = max(0, (int)floorf((tx - 1.0f) * rx) - 1), px1 = min(tex.W - 1, (int)ceilf((tx + 2.0f) * rx) + 1);
        const int py0 = max(0, (int)floorf((ty - 1.0f) * ry) - 1), py1 = min(tex.H - 1, (int)ceilf((ty + 2.0f) * ry) + 1);
        for (int py = py0; py <= py1; py++) {
            const float v = (py + 0.5f) / tex.H;
            for (int px = px0; px <= px1; px++) {
                const float u = (px + 0.5f) / tex.W;
                Tap tp = dev_tap(L, u, v);
                const bool hx = (tp.x0 == tx) || (tp.x1 == tx && tp.x1 != tp.x0);
                const bool hy = (tp.y0 == ty) || (tp.y1 == ty && tp.y1 != tp.y0);
                if (hx && hy) d += dpix[tex.pix_off + (size_t)py * tex.W + px];
            }
        }
    }
    float wgt = d * scale;
    size_t base = tex.z_off + L.off + (size_t)t * L.C;
    for (int c = 0; c < L.C; c++) grad[base + c] += wgt * ntc_gauss(seed, NS_LAT, step, pair, base + c);
}

// Exact per-texel search on the discrete level 0 (= qat_search): one thread per texel,
// channels in order, every grid value of the channel, current value first, strict <.
__device__ __forceinline__ int dev_qat_index(float v, int bits) {
    const int levels = (1 << bits) - 1;
    int k = __float2int_rn((v + 1.0f) * 0.5f * levels);
    return max(0, min(levels, k));
}
template <int MAXW>
__global__ void __launch_bounds__(TPB) k_qat_search(const float* __restrict__ w, float* __restrict__ zset, const float* __restrict__ tgt, int P) {
    extern __shared__ float ws[];
    load_weights_shared(ws, w, P);
    const DevTex& tex = c_tex[blockIdx.y];
    const DevLevel& L = tex.lv[0];
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= L.W * L.H) return;
    int tx = t % L.W, ty = t / L.W;
    const int x0 = L.xb[tx], x1 = L.xe[tx], y0 = L.yb[ty], y1 = L.ye[ty];
    if (x0 >= x1 || y0 >= y1) return;
    float* ztex = zset + tex.z_off;
    float* zt = ztex + (size_t)t * L.C;
    for (int c = 0; c < L.C; c++) {
        const int bits = c_qat.bits[c], levels = (1 << bits) - 1;
        // cell loss of candidate value `val` for channel c (double sum in raster order, as the CPU)
        auto cell_loss = [&](float val) {
            double sum = 0;
            for (int py = y0; py < y1; py++)
                for (int px = x0; px < x1; px++) {
                    float f[MAXW]; dev_features(tex, ztex, px, py, f);
                    f[c] = val;
                    float o[MAX_OUT]; dev_mlp_forward<MAXW>(ws, f, o);
                    sum += dev_pixel_err_d(tex, o, tgt + (tex.tgt_off + (size_t)py * tex.W + px) * tex.nout);
                }
            return sum;
        };
        int best_k = dev_qat_index(zt[c], bits);
        double best = cell_loss(c_qat_grid[bits][best_k]);
        for (int k = 0; k <= levels; k++) {
            if (k == best_k) continue;
            double sk = cell_loss(c_qat_grid[bits][k]);
            if (sk < best) { best = sk; best_k = k; }
        }
        zt[c] = c_qat_grid[bits][best_k];   // visible to the next channel's feature builds
    }
}

// --qes snap (= qes_snap_level for every level, non-qes levels copied): zq = grid[index(z)],
// index = clamp(rn(((z - lo) / range) * levels)); when the ranges are frozen the shadow is first
// clamped to [lo, hi] (and written back). The index expression has no multiply-add pair, so
// FMA contraction cannot alter it; the grid value is looked up, never recomputed.
__global__ void k_snap(float* __restrict__ z, float* __restrict__ zq, const float* __restrict__ grid, size_t nz) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nz) return;
    const DevTex& tex = c_tex[0];
    int l = 0;
    for (int q = 1; q < tex.nlev; q++) if (i >= tex.lv[q].off) l = q;
    const DevLevel& L = tex.lv[l];
    const int bits = c_qes.live ? c_qes.bits[l] : 0;
    float v = z[i];
    if (bits == 0) { zq[i] = v; return; }
    const int c = (int)((i - L.off) % L.C);
    const float lo = c_qes.lo[l][c], hi = c_qes.hi[l][c], range = c_qes.range[l][c];
    if (c_qes.frozen) { v = fminf(fmaxf(v, lo), hi); z[i] = v; }
    const int levels = (1 << bits) - 1;
    int k = __float2int_rn(((v - lo) / range) * (float)levels);
    k = max(0, min(levels, k));
    zq[i] = grid[c_qes.grid_off[l][c] + k];
}

// ---------------------------------------------------------------- DCT: transform-coded level 0 [DCT]
// [DCT] Mirrors of main.cpp's DCT region (dct_fwd8, dct_inv8, dct_quant_*, dct_dequant_*, dct_snap_block,
// [DCT] dct_recon_block, dct_clamp_level0). Same evaluation order; every product and sum is an explicit
// [DCT] __fmul_rn / __fadd_rn / __fsub_rn / __fdiv_rn so nvcc's default contraction cannot fuse a multiply-add
// [DCT] pair; rintf (half-even) for the DC as the host's nearbyintf, roundf (half away) for the first-order pair.
__constant__ DctDesc c_dct;                            // [DCT] live = 0 until set_dct
__constant__ float c_dct_basis[64];                    // [DCT] the host's DCT_BASIS bit patterns
__constant__ int c_dct_step[MAX_DCT_CH][16][64];       // [DCT] host-built integer steps per channel and scale code
__constant__ int c_dct_zigzag[64];                     // [DCT] the host's DCT_ZIGZAG (zigzag position -> natural index), for the --dct-lambda token walk

__device__ __forceinline__ float dev_dct_clampf(float v, float lo, float hi) { if (!(v >= lo)) return lo; if (v > hi) return hi; return v; }
__device__ __forceinline__ void dev_dct_fwd8(const float* w, float* C) {
    float T[64];
    for (int x = 0; x < 8; x++)
        for (int v = 0; v < 8; v++) { float s = 0.0f; for (int y = 0; y < 8; y++) s = __fadd_rn(s, __fmul_rn(w[x * 8 + y], c_dct_basis[v * 8 + y])); T[x * 8 + v] = s; }
    for (int u = 0; u < 8; u++)
        for (int v = 0; v < 8; v++) { float s = 0.0f; for (int x = 0; x < 8; x++) s = __fadd_rn(s, __fmul_rn(T[x * 8 + v], c_dct_basis[u * 8 + x])); C[u * 8 + v] = s; }
}
__device__ __forceinline__ void dev_dct_inv8(const float* C, float* w) {
    float T[64];
    for (int x = 0; x < 8; x++)
        for (int v = 0; v < 8; v++) { float s = 0.0f; for (int u = 0; u < 8; u++) s = __fadd_rn(s, __fmul_rn(c_dct_basis[u * 8 + x], C[u * 8 + v])); T[x * 8 + v] = s; }
    for (int x = 0; x < 8; x++)
        for (int y = 0; y < 8; y++) { float s = 0.0f; for (int v = 0; v < 8; v++) s = __fadd_rn(s, __fmul_rn(c_dct_basis[v * 8 + y], T[x * 8 + v])); w[x * 8 + y] = s; }
}
__device__ __forceinline__ bool dev_dct_first_order(int u, int v) { return (u == 1 && v == 0) || (u == 0 && v == 1); }
__device__ __forceinline__ int dev_dct_quant_dc(float d, int S) { return (int)dev_dct_clampf(rintf(__fdiv_rn(d, (float)S)), -1024.0f, 1024.0f); }
__device__ __forceinline__ float dev_dct_dequant_dc(int q, int S) { return __fmul_rn((float)q, (float)S); }
__device__ __forceinline__ int dev_dct_quant_ac(float d, int L, int u, int v) {
    if (!c_dct.deadzone || dev_dct_first_order(u, v)) return (int)dev_dct_clampf(roundf(__fdiv_rn(d, (float)L)), -1024.0f, 1024.0f);   // [DCT] hook: `!c_dct.deadzone ||` (--dct-deadzone 0: plain rounding on every AC, = dct_quant_ac)
    const float s = fabsf(d), tau = __fmul_rn(0.5f, (float)L);
    if (s <= tau) return 0;
    const float qf = __fdiv_rn(__fsub_rn(s, tau), (float)L);
    const int q = (int)dev_dct_clampf(floorf(__fadd_rn(qf, 0.5f)), 0.0f, 1024.0f);
    return d < 0.0f ? -q : q;
}
__device__ __forceinline__ float dev_dct_dequant_ac(int q, int L, int u, int v) {
    if (!c_dct.deadzone || dev_dct_first_order(u, v)) return __fmul_rn((float)q, (float)L);   // [DCT] hook: `!c_dct.deadzone ||` (= dct_dequant_ac)
    if (q == 0) return 0.0f;
    const float tau = __fmul_rn(0.5f, (float)L);
    const float mag = __fadd_rn(tau, __fmul_rn((float)abs(q), (float)L));
    return q < 0 ? -mag : mag;
}
// [DCT] --dct-lambda: the integer token cost model (= dct_c_run / dct_c_mag / dct_block_bits, 1/16 bit), the truncation walk
// [DCT] (= dct_truncate_block; dD and the threshold are the three _rn operations) and the quantize loop with it (= dct_quantize_block).
__device__ __forceinline__ int dev_dct_floor_log2(unsigned v) { int k = 0; while (v >>= 1) k++; return k; }   // [DCT]
__device__ __forceinline__ int dev_dct_c_run(int r) { return 16 * (2 * dev_dct_floor_log2((unsigned)(r + 1)) + 1); }   // [DCT]
__device__ __forceinline__ int dev_dct_c_mag(int m) { return 16 * (2 * dev_dct_floor_log2((unsigned)m) + 1); }   // [DCT]
__device__ __forceinline__ int dev_dct_lo_weight(int tok, int p) { return p <= 2 ? (tok * c_dct.lo_w16 + 8) >> 4 : tok; }   // [DCT] = dct_lo_weight (--dct-lambda-lo)
__device__ int dev_dct_block_bits(const int16_t* sym) {   // [DCT]
    int bits = 0, run = 0, lnz = 0;   // [DCT]
    for (int p = 1; p < 64; p++) {   // [DCT]
        const int q = sym[c_dct_zigzag[p]];   // [DCT]
        if (q == 0) { run++; continue; }   // [DCT]
        bits += dev_dct_lo_weight(dev_dct_c_run(run) + dev_dct_c_mag(min(256, abs(q))) + 16, p);   // [DCT] DCT_C_SIGN = 16; the pair weighted (--dct-lambda-lo)
        lnz = p; run = 0;   // [DCT]
    }   // [DCT]
    if (lnz < 63) bits += 32;   // [DCT] DCT_C_EOB = 32
    return bits;   // [DCT]
}   // [DCT]
__device__ int dev_dct_truncate_block(const float* Cf, const int* st, float lam16, int16_t* sym) {   // [DCT]
    int ntrunc = 0, next = 64;   // [DCT]
    for (int p = 63; p >= 1; p--) {   // [DCT]
        const int k = c_dct_zigzag[p], q = sym[k];   // [DCT]
        if (q == 0) continue;   // [DCT]
        int prev = 0;   // [DCT]
        for (int j = p - 1; j >= 1; j--) if (sym[c_dct_zigzag[j]] != 0) { prev = j; break; }   // [DCT]
        const int r1 = p - prev - 1;   // [DCT]
        const int token = dev_dct_lo_weight(dev_dct_c_run(r1) + dev_dct_c_mag(min(256, abs(q))) + 16, p);   // [DCT] the pair's own token weighted (= dct_truncate_block)
        const int saved = next == 64 ? token - (p == 63 ? 32 : 0)   // [DCT]
                                     : token + dev_dct_c_run(next - p - 1) - dev_dct_c_run(next - p - 1 + r1 + 1);   // [DCT]
        const float v = dev_dct_dequant_ac(q, st[k], k / 8, k % 8), c = Cf[k];   // [DCT]
        const float dD = __fsub_rn(__fmul_rn(c, c), __fmul_rn(__fsub_rn(c, v), __fsub_rn(c, v)));   // [DCT] = c * c - (c - v) * (c - v), contraction off on the host
        if (dD < __fmul_rn(lam16, (float)saved)) { sym[k] = 0; ntrunc++; }   // [DCT]
        else next = p;   // [DCT]
    }   // [DCT]
    return ntrunc;   // [DCT]
}   // [DCT]
__device__ __forceinline__ int dev_dct_quantize_block(const float* Cf, const int* st, int dc_step, float lam16, bool with_bits, int16_t* sym, int* ntrunc, int* nz_before) {   // [DCT]
    for (int k = 0; k < 64; k++) sym[k] = (int16_t)(k == 0 ? dev_dct_quant_dc(Cf[0], dc_step) : dev_dct_quant_ac(Cf[k], st[k], k / 8, k % 8));   // [DCT] was the quantize loop of k_dct_snap
    int nt = 0, nz = 0;   // [DCT]
    if (lam16 > 0.0f) {   // [DCT]
        for (int k = 1; k < 64; k++) if (sym[k] != 0) nz++;   // [DCT]
        nt = dev_dct_truncate_block(Cf, st, lam16, sym);   // [DCT]
    }   // [DCT]
    if (ntrunc) *ntrunc = nt;   // [DCT]
    if (nz_before) *nz_before = nz;   // [DCT]
    return with_bits ? dev_dct_block_bits(sym) : 0;   // [DCT]
}   // [DCT]
// Dequantize + inverse DCT + clamp into zq for one (block, channel) (= dct_recon_block); Cf holds the dequantized coefficients.
__device__ __forceinline__ void dev_dct_recon_block(int i, const int16_t* __restrict__ sym, const uint8_t* __restrict__ code, float* __restrict__ zq) {
    const int b = i / c_dct.C, c = i % c_dct.C, bx = b % c_dct.BW, by = b / c_dct.BW;
    const DevLevel& L0 = c_tex[0].lv[0];
    const int* st = c_dct_step[c][code[i]];
    const int16_t* sy = sym + (size_t)i * 64;
    float Cf[64], w[64];
    for (int k = 0; k < 64; k++) Cf[k] = k == 0 ? dev_dct_dequant_dc(sy[0], c_dct.dc_step) : dev_dct_dequant_ac(sy[k], st[k], k / 8, k % 8);
    dev_dct_inv8(Cf, w);
    for (int y = 0; y < 8; y++)
        for (int x = 0; x < 8; x++) {
            const size_t idx = ((size_t)(by * 8 + y) * L0.W + (bx * 8 + x)) * L0.C + c;
            zq[idx] = dev_dct_clampf(__fsub_rn(__fdiv_rn(w[y * 8 + x], 32.0f), 1.0f), -1.0f, 1.0f);
        }
}
// Not yet live: clamp level 0 of the shadow to [-1,1] and write the clamped value into the decode copy too (= dct_clamp_level0).
__global__ void k_dct_clamp0(float* __restrict__ z, float* __restrict__ zq, size_t n0) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n0) return;
    const float v = dev_dct_clampf(z[i], -1.0f, 1.0f);
    z[i] = v; zq[i] = v;
}
// One thread per (block, channel): gather 64 shadow values (clamped in place), w = (s + 1) * 32, forward DCT, quantize with the
// block's step table, write the symbols, dequantize, inverse DCT, zq = clamp(w' / 32 - 1) (= dct_snap_block).
__global__ void k_dct_snap(float* __restrict__ z, float* __restrict__ zq, int16_t* __restrict__ sym, const uint8_t* __restrict__ code, int32_t* __restrict__ bits, int* __restrict__ cnt) {   // [DCT] hook: bits (proxy bits per (block, channel)) and cnt[2] (ntrunc, nz_before) for --dct-lambda
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= c_dct.BW * c_dct.BH * c_dct.C) return;
    const int b = i / c_dct.C, c = i % c_dct.C, bx = b % c_dct.BW, by = b / c_dct.BW;
    const DevLevel& L0 = c_tex[0].lv[0];
    float w[64], Cf[64];
    for (int y = 0; y < 8; y++)
        for (int x = 0; x < 8; x++) {
            const size_t idx = ((size_t)(by * 8 + y) * L0.W + (bx * 8 + x)) * L0.C + c;
            const float s = dev_dct_clampf(z[idx], -1.0f, 1.0f);
            z[idx] = s;
            w[y * 8 + x] = __fmul_rn(__fadd_rn(s, 1.0f), 32.0f);
        }
    dev_dct_fwd8(w, Cf);
    const int* st = c_dct_step[c][code[i]];
    int16_t* sy = sym + (size_t)i * 64;
    int nt = 0, nz = 0;   // [DCT]
    dev_dct_quantize_block(Cf, st, c_dct.dc_step, c_dct.lam_t16[code[i]], false, sy, &nt, &nz);   // [DCT] hook: was the one-line quantize loop (part B follows inside when lam_t16[code] > 0)
    if (c_dct.rate_es || c_dct.rate_trunc) { bits[i] = dev_dct_block_bits(sy); if (nt) atomicAdd(cnt, nt); if (nz) atomicAdd(cnt + 1, nz); }   // [DCT] integer adds: order-independent
    dev_dct_recon_block(i, sym, code, zq);
}
// Level 0 of the decode copy from the uploaded symbols (set_dct); the shadow is not touched.
__global__ void k_dct_recon(const int16_t* __restrict__ sym, const uint8_t* __restrict__ code, float* __restrict__ zq, int32_t* __restrict__ bits) {   // [DCT] hook: bits (--dct-lambda)
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= c_dct.BW * c_dct.BH * c_dct.C) return;
    if (c_dct.rate_es || c_dct.rate_trunc) bits[i] = dev_dct_block_bits(sym + (size_t)i * 64);   // [DCT] the proxy bits of the uploaded symbols (= dct_recon_block's fill)
    dev_dct_recon_block(i, sym, code, zq);
}
// [DCT] --dct-lambda part A (= dct_rate_block twice per (block, channel) in LatentTrainer::step): one thread per (block, channel) gathers
// [DCT] clamp(z +/- sg * eps) from the shadow with the pair's hash noise (the k_lat_pair key of a nearest level), forward DCT, quantize /
// [DCT] truncate, and writes the proxy bit difference; hits counts the nonzero differences.
__global__ void k_dct_rate_pair(uint64_t seed, int step, int pair, float sg, const float* __restrict__ z, const uint8_t* __restrict__ code, int32_t* __restrict__ dbits, int* __restrict__ hits) {   // [DCT]
    const int i = blockIdx.x * blockDim.x + threadIdx.x;   // [DCT]
    if (i >= c_dct.BW * c_dct.BH * c_dct.C) return;   // [DCT]
    const int b = i / c_dct.C, c = i % c_dct.C, bx = b % c_dct.BW, by = b / c_dct.BW;   // [DCT]
    const DevTex& tex = c_tex[0]; const DevLevel& L0 = tex.lv[0];   // [DCT]
    const int* st = c_dct_step[c][code[i]];   // [DCT]
    const float lam = c_dct.lam_t16[code[i]];   // [DCT]
    float w[64], Cf[64]; int16_t sym[64];   // [DCT]
    int bp = 0, bm = 0;   // [DCT]
    for (int sgn = 0; sgn < 2; sgn++) {   // [DCT] + then -
        for (int y = 0; y < 8; y++)   // [DCT]
            for (int x = 0; x < 8; x++) {   // [DCT]
                const size_t idx = ((size_t)(by * 8 + y) * L0.W + (bx * 8 + x)) * L0.C + c;   // [DCT]
                const float d = __fmul_rn(sg, ntc_gauss(seed, NS_LAT, step, pair, tex.z_off + L0.off + idx));   // [DCT]
                const float s = dev_dct_clampf(sgn == 0 ? __fadd_rn(z[idx], d) : __fsub_rn(z[idx], d), -1.0f, 1.0f);   // [DCT] = the host's z + sign * (sg * eps)
                w[y * 8 + x] = __fmul_rn(__fadd_rn(s, 1.0f), 32.0f);   // [DCT]
            }   // [DCT]
        dev_dct_fwd8(w, Cf);   // [DCT]
        const int bb = dev_dct_quantize_block(Cf, st, c_dct.dc_step, lam, true, sym, nullptr, nullptr);   // [DCT]
        if (sgn == 0) bp = bb; else bm = bb;   // [DCT]
    }   // [DCT]
    dbits[i] = bp - bm;   // [DCT]
    if (bp != bm) atomicAdd(hits, 1);   // [DCT]
}   // [DCT]
// [DCT] The attribution of a (block, channel)'s bit difference to its 64 texels (= the grad loop of the host's rate pass): one thread per
// [DCT] level-0 texel; kept apart from k_lat_gather so that kernel is untouched.
__global__ void k_dct_rate_gather(uint64_t seed, int step, int pair, float scale, const int32_t* __restrict__ dbits, float* __restrict__ grad) {   // [DCT]
    const DevTex& tex = c_tex[0]; const DevLevel& L0 = tex.lv[0];   // [DCT]
    const int t = blockIdx.x * blockDim.x + threadIdx.x;   // [DCT]
    if (t >= L0.W * L0.H) return;   // [DCT]
    const int tx = t % L0.W, ty = t / L0.W;   // [DCT]
    const int b = (ty / 8) * c_dct.BW + tx / 8;   // [DCT]
    const size_t base = tex.z_off + L0.off + (size_t)t * L0.C;   // [DCT]
    for (int c = 0; c < L0.C; c++) grad[base + c] += c_dct.rate_scale * (float)dbits[b * c_dct.C + c] * scale * ntc_gauss(seed, NS_LAT, step, pair, base + c);   // [DCT]
}   // [DCT]
// ---------------------------------------------------------------- end of the DCT region [DCT]

// ---------------------------------------------------------------- host side
struct Trainer::Impl {
    ModelDesc d;
    int ntex = 1;
    int P = 0;                 // MLP weight count
    size_t nz = 0;             // latent floats per texture (one texture in v1)
    int maxw = 0;              // MAXW instantiation chosen for this model
    // device buffers
    float *d_z = nullptr, *d_zgrad = nullptr, *d_zm = nullptr, *d_zv = nullptr;
    float* d_ztmp = nullptr;   // scratch for decode_full with a host-supplied latent
    float* d_zq = nullptr;     // --qes: persistent snapped decode copy (allocated when some level has --qes bits)
    float* d_zdec = nullptr;   // what every decode-type kernel reads: d_zq when it exists, else d_z
    float* d_qgrid = nullptr;  // --qes grid table (host-computed), MAX_QES_GRID floats
    bool qes = false;          // some level has --qes bits
    bool dct = false, dct_live = false;   // [DCT] --dct-q level 0; live once set_dct uploaded the codes and symbols
    int16_t* d_dct_sym = nullptr; uint8_t* d_dct_code = nullptr; size_t dct_n = 0;   // [DCT] [block][channel][64] symbols, [block][channel] codes, dct_n = blocks * channels
    int32_t* d_dct_bits = nullptr; int32_t* d_dct_dbits = nullptr; int* d_dct_cnt = nullptr;   // [DCT] --dct-lambda: proxy bits and last-pair bit differences per (block, channel); cnt[3] = ntrunc, nz_before (last snap), hits (last lat_step)
    int dct_evals = 0;         // [DCT] --dct-lambda: the (block, channel, pair) evaluations of the last lat_step (= dct_n * K when part A ran, host-known: no device copy)
    bool dct_rate_es = false, dct_rate_trunc = false;   // [DCT] host-side copies of the DctDesc flags (read by lat_step / snap_latent_impl)
    float *d_w = nullptr, *d_wgrad = nullptr, *d_wm = nullptr, *d_wv = nullptr;
    float *d_tgt = nullptr, *d_img = nullptr, *d_dpix = nullptr;
    float *d_feat = nullptr, *d_btgt = nullptr, *d_bnrm = nullptr; int* d_bidx = nullptr; int Mcap = 0;   // Mcap = --mlp-batch slots
    double *d_part = nullptr, *d_dl = nullptr, *d_lsum = nullptr, *d_hh = nullptr; size_t part_cap = 0; int pairs_cap = 0;
    int* d_range = nullptr;    // per-level xb/xe/yb/ye
    // Adam step counters (host, as in Adam::t)
    int mlp_t = 0, lat_t = 0, lat_step_no = 0;
    // last MLP step bookkeeping for stats
    int last_N = 0; bool last_fd = false; int last_M = 0;
    std::string dev_name; int sm_major = 0, sm_minor = 0, sms = 0, rt_ver = 0;
    size_t shmem() const { return (size_t)P * sizeof(float); }
};

Trainer::Trainer() : im(new Impl) {}
Trainer::~Trainer() {
    Impl& I = *im;
    if (!I.d_z) { delete im; return; }   // never initialised (a CPU run of a CUDA build): do not touch the driver
    cudaFree(I.d_z); cudaFree(I.d_zgrad); cudaFree(I.d_zm); cudaFree(I.d_zv); cudaFree(I.d_ztmp);
    if (I.d_zq) cudaFree(I.d_zq); if (I.d_qgrid) cudaFree(I.d_qgrid);
    if (I.d_dct_sym) cudaFree(I.d_dct_sym); if (I.d_dct_code) cudaFree(I.d_dct_code);   // [DCT]
    if (I.d_dct_bits) cudaFree(I.d_dct_bits); if (I.d_dct_dbits) cudaFree(I.d_dct_dbits); if (I.d_dct_cnt) cudaFree(I.d_dct_cnt);   // [DCT] (cnt[3]: ntrunc, nz_before, hits)
    cudaFree(I.d_w); cudaFree(I.d_wgrad); cudaFree(I.d_wm); cudaFree(I.d_wv);
    cudaFree(I.d_tgt); cudaFree(I.d_img); cudaFree(I.d_dpix);
    cudaFree(I.d_feat); cudaFree(I.d_btgt); cudaFree(I.d_bnrm); cudaFree(I.d_bidx);
    cudaFree(I.d_part); cudaFree(I.d_dl); cudaFree(I.d_lsum); cudaFree(I.d_hh); cudaFree(I.d_range);
    delete im;
}

static int pick_maxw(int need) {
    if (need <= 32) return 32;
    if (need <= 64) return 64;
    return 0;
}

bool Trainer::init(const ModelDesc& d, const float* z, const float* p, const float* target, std::string& why) {
    Impl& I = *im;
    I.d = d;
    // v1 envelope
    int need = d.nin; for (int l = 0; l < d.nhidden; l++) need = std::max(need, d.hidden[l]);
    I.maxw = pick_maxw(need);
    if (I.maxw == 0) { why = "--cuda v1 supports at most 64 units per layer (input and hidden)"; return false; }
    if (d.nout > MAX_OUT) { why = "too many MLP outputs"; return false; }
    if (d.nlev < 1 || d.nlev > MAX_LEVELS) { why = "bad level count"; return false; }
    if (d.lv[0].C > MAX_CH) { why = "--cuda v1 supports at most 16 level-0 channels"; return false; }
    I.qes = false;
    for (int l = 0; l < d.nlev; l++) if (d.qes_bits[l] > 0) {
        I.qes = true;
        if (d.qes_bits[l] < 2 || d.qes_bits[l] > 12) { why = "--qes bits out of range"; return false; }
        if (d.lv[l].C > MAX_QES_CH) { why = "--cuda supports at most 64 channels on a --qes level"; return false; }
    }
    I.dct = d.dct != 0; I.dct_live = false;   // [DCT]
    if (I.dct) {   // [DCT] envelope: one full-resolution nearest level 0 with at most MAX_DCT_CH channels, a multiple of 8, the host basis table
        if (d.dct_N != 8) { why = "--dct-q: the CUDA backend supports --dct-block 8 only"; return false; }   // [DCT]
        if (d.dct_C < 1 || d.dct_C > MAX_DCT_CH || d.dct_C != d.lv[0].C) { why = "--dct-q: level 0 must have 1..4 channels (= dct_C)"; return false; }   // [DCT]
        if (!d.lv[0].nearest || d.lv[0].W != d.W || d.lv[0].H != d.H || d.W % 8 != 0 || d.H % 8 != 0) { why = "--dct-q: level 0 must be a full-resolution nearest level with a size that is a multiple of 8"; return false; }   // [DCT]
        if (!d.dct_basis) { why = "--dct-q needs the host basis table"; return false; }   // [DCT]
        if (!d.dct_zigzag) { why = "--dct-q needs the host zigzag table"; return false; }   // [DCT]
        if (d.dct_dc_step < 1 || d.dct_dc_step > 64) { why = "--dct-q: DC step out of range"; return false; }   // [DCT]
    }   // [DCT]
    if ((size_t)d.W * d.H >= ((size_t)1 << 28)) { why = "--cuda supports images below 2^28 pixels"; return false; }
    if (d.max_pairs < 1 || d.max_batch < 1) { why = "bad pair / batch counts"; return false; }
    // weight count
    {
        int ncur = d.nin; size_t n = 0;
        for (int l = 0; l < d.nhidden; l++) { n += (size_t)d.hidden[l] * ncur + d.hidden[l]; ncur = d.hidden[l]; }
        n += (size_t)d.nout * ncur + d.nout;
        I.P = (int)n;
    }
    if (I.shmem() > 48 * 1024) { why = "--cuda v1 supports MLPs up to 12288 weights"; return false; }

    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) { why = "no CUDA device"; return false; }
    cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    I.dev_name = prop.name; I.sm_major = prop.major; I.sm_minor = prop.minor; I.sms = prop.multiProcessorCount;
    CUDA_CHECK(cudaRuntimeGetVersion(&I.rt_ver));

    // descriptors
    I.nz = 0; for (int l = 0; l < d.nlev; l++) I.nz += (size_t)d.lv[l].W * d.lv[l].H * d.lv[l].C;
    // per-level texel -> pixel ranges (as qat_search derives them from the tap), one int block per level
    std::vector<int> ranges; std::vector<size_t> roff;
    for (int l = 0; l < d.nlev; l++) {
        const LevelDesc& L = d.lv[l];
        roff.push_back(ranges.size());
        std::vector<int> xb(L.W, d.W), xe(L.W, 0), yb(L.H, d.H), ye(L.H, 0);
        for (int px = 0; px < d.W; px++) { float x = ((px + 0.5f) / d.W) * L.W; int t = std::max(0, std::min(L.W - 1, (int)floorf(x))); xb[t] = std::min(xb[t], px); xe[t] = std::max(xe[t], px + 1); }
        for (int py = 0; py < d.H; py++) { float y = ((py + 0.5f) / d.H) * L.H; int t = std::max(0, std::min(L.H - 1, (int)floorf(y))); yb[t] = std::min(yb[t], py); ye[t] = std::max(ye[t], py + 1); }
        ranges.insert(ranges.end(), xb.begin(), xb.end()); ranges.insert(ranges.end(), xe.begin(), xe.end());
        ranges.insert(ranges.end(), yb.begin(), yb.end()); ranges.insert(ranges.end(), ye.begin(), ye.end());
    }
    CUDA_CHECK(cudaMalloc(&I.d_range, ranges.size() * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(I.d_range, ranges.data(), ranges.size() * sizeof(int), cudaMemcpyHostToDevice));

    DevTex tex[MAX_TEX]; memset(tex, 0, sizeof(tex));
    DevTex& T0 = tex[0];
    T0.W = d.W; T0.H = d.H; T0.nout = d.nout; T0.T = d.T; T0.nlev = d.nlev;
    for (int l = 0; l < d.nlev; l++) {
        DevLevel& L = T0.lv[l];
        L.W = d.lv[l].W; L.H = d.lv[l].H; L.C = d.lv[l].C; L.off = d.lv[l].off; L.nearest = d.lv[l].nearest ? 1 : 0;
        const int* base = I.d_range + roff[l];
        L.xb = base; L.xe = base + L.W; L.yb = base + 2 * L.W; L.ye = base + 2 * L.W + L.H;
    }
    T0.z_off = 0; T0.tgt_off = 0; T0.pix_off = 0;
    for (int c = 0; c < MAX_OUT; c++) T0.cw[c] = c < d.nout ? d.cw[c] : 0.0f;
    T0.wsum = d.wsum;
    T0.inv_px = 1.0f / (3.0f * d.wsum * d.W * d.H);
    CUDA_CHECK(cudaMemcpyToSymbol(c_tex, tex, sizeof(tex)));

    DevMlp mlp; memset(&mlp, 0, sizeof(mlp));
    mlp.nin = d.nin; mlp.nout = d.nout; mlp.nl = d.nhidden; for (int l = 0; l < d.nhidden; l++) mlp.w[l] = d.hidden[l];
    mlp.leak = d.leak; mlp.clamp = d.clamp_out ? 1 : 0;
    CUDA_CHECK(cudaMemcpyToSymbol(c_mlp, &mlp, sizeof(mlp)));

    DevPos pos; memset(&pos, 0, sizeof(pos));
    pos.n = d.npos; for (int q = 0; q < d.npos; q++) pos.kind[q] = d.pos[q].kind;
    CUDA_CHECK(cudaMemcpyToSymbol(c_pos, &pos, sizeof(pos)));

    DevQat qat; memset(&qat, 0, sizeof(qat));
    for (int c = 0; c < MAX_CH; c++) qat.bits[c] = d.qat_bits[c];
    CUDA_CHECK(cudaMemcpyToSymbol(c_qat, &qat, sizeof(qat)));
    if (d.qat > 0) {
        // The grid values must be bit-identical to the CPU's table (the on-grid invariant is
        // checked with exact float compares), so they are copied, never recomputed here.
        if (!d.qat_grid) { why = "--qat needs the host grid table"; return false; }
        CUDA_CHECK(cudaMemcpyToSymbol(c_qat_grid, d.qat_grid, sizeof(float) * 9 * 257));
    }
    {   // --qes: not live until set_qes uploads the host-fitted ranges; k_snap copies the shadow until then
        QesDesc q; memset(&q, 0, sizeof(q));
        for (int l = 0; l < d.nlev; l++) q.bits[l] = d.qes_bits[l];
        CUDA_CHECK(cudaMemcpyToSymbol(c_qes, &q, sizeof(q)));
    }
    if (I.dct) {   // [DCT] the description (not live), the basis bit patterns; the steps, codes and symbols arrive with set_dct
        DctDesc dd; memset(&dd, 0, sizeof(dd));   // [DCT]
        dd.live = 0; dd.C = d.dct_C; dd.N = d.dct_N; dd.BW = d.W / 8; dd.BH = d.H / 8; dd.dc_step = d.dct_dc_step;   // [DCT]
        dd.deadzone = 1;   // [DCT] the quantizer flag arrives with set_dct (nothing quantizes before it; --dct-deadzone)
        CUDA_CHECK(cudaMemcpyToSymbol(c_dct, &dd, sizeof(dd)));   // [DCT]
        CUDA_CHECK(cudaMemcpyToSymbol(c_dct_basis, d.dct_basis, sizeof(float) * 64));   // [DCT]
        CUDA_CHECK(cudaMemcpyToSymbol(c_dct_zigzag, d.dct_zigzag, sizeof(int) * 64));   // [DCT]
        I.dct_n = (size_t)dd.BW * dd.BH * dd.C;   // [DCT]
    }   // [DCT]

    // buffers
    const size_t npix = (size_t)d.W * d.H;
    CUDA_CHECK(cudaMalloc(&I.d_z, I.nz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&I.d_zgrad, I.nz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&I.d_zm, I.nz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&I.d_zv, I.nz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&I.d_ztmp, I.nz * sizeof(float)));
    if (I.qes || I.dct) {   // [DCT] hook: || I.dct (the snapped decode copy exists for a DCT-coded level 0 too)
        CUDA_CHECK(cudaMalloc(&I.d_zq, I.nz * sizeof(float)));
        if (I.qes) {   // [DCT] hook: the grid table only with a --qes level (k_snap never reads it when no level has bits)
        CUDA_CHECK(cudaMalloc(&I.d_qgrid, (size_t)MAX_QES_GRID * sizeof(float)));
        CUDA_CHECK(cudaMemset(I.d_qgrid, 0, (size_t)MAX_QES_GRID * sizeof(float)));
        }   // [DCT]
        if (I.dct) {   // [DCT] symbols and scale codes
            CUDA_CHECK(cudaMalloc(&I.d_dct_sym, I.dct_n * 64 * sizeof(int16_t)));   // [DCT]
            CUDA_CHECK(cudaMalloc(&I.d_dct_code, I.dct_n * sizeof(uint8_t)));   // [DCT]
            CUDA_CHECK(cudaMemset(I.d_dct_sym, 0, I.dct_n * 64 * sizeof(int16_t)));   // [DCT]
            CUDA_CHECK(cudaMemset(I.d_dct_code, 0, I.dct_n * sizeof(uint8_t)));   // [DCT]
            CUDA_CHECK(cudaMalloc(&I.d_dct_bits, I.dct_n * sizeof(int32_t)));   // [DCT] --dct-lambda buffers (written only when a rate flag is set)
            CUDA_CHECK(cudaMalloc(&I.d_dct_dbits, I.dct_n * sizeof(int32_t)));   // [DCT]
            CUDA_CHECK(cudaMalloc(&I.d_dct_cnt, 3 * sizeof(int)));   // [DCT]
            CUDA_CHECK(cudaMemset(I.d_dct_bits, 0, I.dct_n * sizeof(int32_t)));   // [DCT]
            CUDA_CHECK(cudaMemset(I.d_dct_dbits, 0, I.dct_n * sizeof(int32_t)));   // [DCT]
            CUDA_CHECK(cudaMemset(I.d_dct_cnt, 0, 3 * sizeof(int)));   // [DCT]
        }   // [DCT]
        I.d_zdec = I.d_zq;
    } else I.d_zdec = I.d_z;   // no --qes level: the decode buffer is the shadow itself (no copies, no behavior change)
    CUDA_CHECK(cudaMalloc(&I.d_w, I.P * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&I.d_wgrad, I.P * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&I.d_wm, I.P * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&I.d_wv, I.P * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&I.d_tgt, npix * d.nout * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&I.d_img, npix * d.nout * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&I.d_dpix, npix * sizeof(float)));
    I.Mcap = d.max_batch;   // one slot per minibatch pixel (sampling with replacement: M may exceed the pixel count)
    CUDA_CHECK(cudaMalloc(&I.d_feat, (size_t)I.Mcap * d.nin * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&I.d_btgt, (size_t)I.Mcap * MAX_OUT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&I.d_bnrm, (size_t)I.Mcap * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&I.d_bidx, (size_t)I.Mcap * sizeof(int)));
    I.part_cap = (size_t)std::max(2 * 4096, 2 * I.P) * 64;   // grown on demand
    CUDA_CHECK(cudaMalloc(&I.d_part, I.part_cap * sizeof(double)));
    I.pairs_cap = std::max(d.max_pairs, 1);
    CUDA_CHECK(cudaMalloc(&I.d_dl, (size_t)std::max(I.pairs_cap, I.P) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&I.d_lsum, (size_t)I.pairs_cap * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&I.d_hh, (size_t)I.P * sizeof(double)));
    CUDA_CHECK(cudaMemset(I.d_zm, 0, I.nz * sizeof(float)));
    CUDA_CHECK(cudaMemset(I.d_zv, 0, I.nz * sizeof(float)));
    CUDA_CHECK(cudaMemset(I.d_wm, 0, I.P * sizeof(float)));
    CUDA_CHECK(cudaMemset(I.d_wv, 0, I.P * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(I.d_tgt, target, npix * d.nout * sizeof(float), cudaMemcpyHostToDevice));
    upload_model(z, p);
    return true;
}

static inline int blocks_for(size_t n, int tpb) { return (int)((n + tpb - 1) / tpb); }

std::string Trainer::banner() const {
    const Impl& I = *im;
    char buf[256];
    snprintf(buf, sizeof(buf), "%s (sm_%d%d, %d SMs), CUDA runtime %d.%d, MLP kernels MAXW=%d", I.dev_name.c_str(), I.sm_major, I.sm_minor, I.sms, I.rt_ver / 1000, (I.rt_ver % 1000) / 10, I.maxw);
    return buf;
}

static void snap_latent_impl(Trainer::Impl& I) {
    if (!I.qes && !I.dct) return;   // [DCT] hook: && !I.dct
    k_snap<<<blocks_for(I.nz, 256), 256>>>(I.d_z, I.d_zq, I.d_qgrid, I.nz);
    CUDA_CHECK(cudaGetLastError());
    if (I.dct) {   // [DCT] hook: level 0 of the copy was a plain copy above; now the DCT snap (live) or the pre-start clamp of shadow and copy
        if (I.dct_live) {   // [DCT]
            if (I.dct_rate_es || I.dct_rate_trunc) CUDA_CHECK(cudaMemsetAsync(I.d_dct_cnt, 0, 2 * sizeof(int)));   // [DCT] --dct-lambda: the snap's ntrunc / nz_before counters
            k_dct_snap<<<blocks_for(I.dct_n, 128), 128>>>(I.d_z, I.d_zq, I.d_dct_sym, I.d_dct_code, I.d_dct_bits, I.d_dct_cnt);   // [DCT] hook: the two --dct-lambda arguments
        }   // [DCT]
        else { const size_t n0 = (size_t)I.d.lv[0].W * I.d.lv[0].H * I.d.lv[0].C; k_dct_clamp0<<<blocks_for(n0, 256), 256>>>(I.d_z, I.d_zq, n0); }   // [DCT]
        CUDA_CHECK(cudaGetLastError());   // [DCT]
    }   // [DCT]
}   // [DCT]
// [DCT] Upload the description (live), the step tables, the codes and the symbols; rebuild level 0 of the decode copy
// [DCT] from the symbols (k_dct_recon, bit for bit the host's dct_recon_level0). The shadow is not touched.
void Trainer::set_dct(const DctDesc& d, const uint8_t* code, const int16_t* sym, const int* step) {   // [DCT]
    Impl& I = *im;   // [DCT]
    if (!I.dct) { fprintf(stderr, "CUDA backend: set_dct without --dct-q in the model description\n"); exit(1); }   // [DCT]
    if (d.C != I.d.dct_C || (size_t)d.BW * d.BH * d.C != I.dct_n || d.N != 8) { fprintf(stderr, "CUDA backend: set_dct description mismatch\n"); exit(1); }   // [DCT]
    DctDesc dd = d; dd.live = 1;   // [DCT]
    CUDA_CHECK(cudaMemcpyToSymbol(c_dct, &dd, sizeof(dd)));   // [DCT]
    CUDA_CHECK(cudaMemcpyToSymbol(c_dct_step, step, sizeof(int) * 16 * 64 * d.C));   // [DCT]
    CUDA_CHECK(cudaMemcpy(I.d_dct_code, code, I.dct_n * sizeof(uint8_t), cudaMemcpyHostToDevice));   // [DCT]
    CUDA_CHECK(cudaMemcpy(I.d_dct_sym, sym, I.dct_n * 64 * sizeof(int16_t), cudaMemcpyHostToDevice));   // [DCT]
    I.dct_live = true;   // [DCT]
    I.dct_rate_es = d.rate_es != 0; I.dct_rate_trunc = d.rate_trunc != 0;   // [DCT] --dct-lambda flags (host-side copies for lat_step / snap_latent_impl)
    k_dct_recon<<<blocks_for(I.dct_n, 128), 128>>>(I.d_dct_sym, I.d_dct_code, I.d_zq, I.d_dct_bits);   // [DCT] hook: the bits argument
    CUDA_CHECK(cudaGetLastError());   // [DCT]
}   // [DCT]
void Trainer::download_dct(int16_t* sym, uint8_t* code) {   // [DCT]
    Impl& I = *im;   // [DCT]
    if (!I.dct) return;   // [DCT]
    CUDA_CHECK(cudaMemcpy(sym, I.d_dct_sym, I.dct_n * 64 * sizeof(int16_t), cudaMemcpyDeviceToHost));   // [DCT]
    CUDA_CHECK(cudaMemcpy(code, I.d_dct_code, I.dct_n * sizeof(uint8_t), cudaMemcpyDeviceToHost));   // [DCT]
}
void Trainer::download_dct_rate(int32_t* bits, int32_t* dbits_last_pair, int* ntrunc, int* nz_before, int* hits, int* evals) {   // [DCT]
    Impl& I = *im;   // [DCT]
    if (!I.dct) return;   // [DCT]
    if (bits) CUDA_CHECK(cudaMemcpy(bits, I.d_dct_bits, I.dct_n * sizeof(int32_t), cudaMemcpyDeviceToHost));   // [DCT]
    if (dbits_last_pair) CUDA_CHECK(cudaMemcpy(dbits_last_pair, I.d_dct_dbits, I.dct_n * sizeof(int32_t), cudaMemcpyDeviceToHost));   // [DCT]
    int cnt[3]; CUDA_CHECK(cudaMemcpy(cnt, I.d_dct_cnt, 3 * sizeof(int), cudaMemcpyDeviceToHost));   // [DCT]
    if (ntrunc) *ntrunc = cnt[0]; if (nz_before) *nz_before = cnt[1]; if (hits) *hits = cnt[2]; if (evals) *evals = I.dct_evals;   // [DCT] evals is host state (dct_n * K of the last lat_step)
}   // [DCT]

void Trainer::upload_model(const float* z, const float* p) {
    Impl& I = *im;
    CUDA_CHECK(cudaMemcpy(I.d_z, z, I.nz * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(I.d_w, p, I.P * sizeof(float), cudaMemcpyHostToDevice));
    snap_latent_impl(I);
}
void Trainer::set_qes(const QesDesc& q, const float* grid, size_t grid_n) {
    Impl& I = *im;
    if (!I.qes) { fprintf(stderr, "CUDA backend: set_qes without a --qes level in the model description\n"); exit(1); }
    if (grid_n > (size_t)MAX_QES_GRID) { fprintf(stderr, "CUDA backend: --qes grid table too large (%zu floats)\n", grid_n); exit(1); }
    CUDA_CHECK(cudaMemcpyToSymbol(c_qes, &q, sizeof(q)));
    if (grid_n) CUDA_CHECK(cudaMemcpy(I.d_qgrid, grid, grid_n * sizeof(float), cudaMemcpyHostToDevice));
    snap_latent_impl(I);
}
__global__ void k_shadow_pull(float* __restrict__ z, const float* __restrict__ zq, size_t off, size_t n, float f) {   // [DCT] --dct-shadow-pull: z += f * (zq - z) on [off, off + n)
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) z[off + i] += f * (zq[off + i] - z[off + i]);
}
void Trainer::shadow_pull(size_t off, size_t n, float f) {   // [DCT]
    Impl& I = *im;
    if (!I.d_zq || n == 0) return;
    k_shadow_pull<<<(unsigned)((n + 255) / 256), 256>>>(I.d_z, I.d_zq, off, n, f);
    CUDA_CHECK(cudaGetLastError());
    snap_latent_impl(I);
}
void Trainer::snap_latent() { snap_latent_impl(*im); }
void Trainer::download_zq(float* zq) {
    Impl& I = *im;
    CUDA_CHECK(cudaMemcpy(zq, I.d_zdec, I.nz * sizeof(float), cudaMemcpyDeviceToHost));
}
void Trainer::download_model(float* z, float* p) {
    Impl& I = *im;
    CUDA_CHECK(cudaMemcpy(z, I.d_z, I.nz * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(p, I.d_w, I.P * sizeof(float), cudaMemcpyDeviceToHost));
}

// Dispatch helpers over the MAXW instantiations.
#define NTC_DISPATCH_MAXW(I, CALL) \
    switch ((I).maxw) { case 32: { constexpr int MAXW = 32; CALL; break; } default: { constexpr int MAXW = 64; CALL; break; } }

void Trainer::decode_full(const float* z_host, float* img) {
    Impl& I = *im;
    const float* zsrc = I.d_zdec;
    if (z_host) { CUDA_CHECK(cudaMemcpy(I.d_ztmp, z_host, I.nz * sizeof(float), cudaMemcpyHostToDevice)); zsrc = I.d_ztmp; }
    const size_t npix = (size_t)I.d.W * I.d.H;
    dim3 grid(blocks_for(npix, TPB), I.ntex);
    NTC_DISPATCH_MAXW(I, (k_decode<MAXW><<<grid, TPB, I.shmem()>>>(I.d_w, zsrc, I.d_img, I.P)));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(img, I.d_img, npix * I.d.nout * sizeof(float), cudaMemcpyDeviceToHost));
}

static void draw_batch(Trainer::Impl& I, int it, int M) {
    dim3 grid(blocks_for(M, 256));
    NTC_DISPATCH_MAXW(I, (k_batch_features<MAXW><<<grid, 256>>>(I.d.seed, it, M, I.ntex, I.d_zdec, I.d_tgt, I.d_bidx, I.d_feat, I.d_btgt, I.d_bnrm)));
    CUDA_CHECK(cudaGetLastError());
    I.last_M = M;
}

static void ensure_part(Trainer::Impl& I, size_t need) {
    if (need <= I.part_cap) return;
    CUDA_CHECK(cudaFree(I.d_part));
    I.part_cap = need;
    CUDA_CHECK(cudaMalloc(&I.d_part, I.part_cap * sizeof(double)));
}

static void adam_step(float* theta, const float* g, float* m, float* v, int n, float lr, int& t) {
    t++;
    float c1 = 1.0f - powf(0.9f, (float)t), c2 = 1.0f - powf(0.999f, (float)t);
    k_adam<<<blocks_for(n, 256), 256>>>(theta, g, m, v, n, lr, c1, c2);
    CUDA_CHECK(cudaGetLastError());
}

void Trainer::mlp_step(int it, float sigma, float lr, int N, int M) {
    Impl& I = *im;
    if (N > I.pairs_cap || M > I.Mcap) { fprintf(stderr, "CUDA backend: MLP step with %d pairs / %d pixels exceeds the buffers sized at init (%d / %d)\n", N, M, I.pairs_cap, I.Mcap); exit(1); }
    draw_batch(I, it, M);
    const int chunks = (M + TPB * PPT - 1) / (TPB * PPT);
    ensure_part(I, (size_t)2 * N * chunks);
    dim3 grid(chunks, 2 * N);
    NTC_DISPATCH_MAXW(I, (k_mlp_es_eval<MAXW><<<grid, TPB, I.shmem()>>>(I.d.seed, it, sigma, M, I.P, I.d_w, I.d_feat, I.d_btgt, I.d_bidx, I.d_part, chunks)));
    CUDA_CHECK(cudaGetLastError());
    const double norm = 3.0 * I.d.wsum * M;
    k_mlp_es_finish<<<blocks_for(N, 128), 128>>>(I.d_part, chunks, N, norm, I.d_dl, I.d_lsum);
    CUDA_CHECK(cudaGetLastError());
    const float scale = 1.0f / (2.0f * N * sigma);
    k_mlp_es_grad<<<blocks_for(I.P, 256), 256>>>(I.d.seed, it, scale, N, I.P, I.d_dl, I.d_wgrad);
    CUDA_CHECK(cudaGetLastError());
    adam_step(I.d_w, I.d_wgrad, I.d_wm, I.d_wv, I.P, lr, I.mlp_t);
    I.last_N = N; I.last_fd = false;
}

void Trainer::mlp_step_fd(int it, float h, float lr, int M) {
    Impl& I = *im;
    if (M > I.Mcap) { fprintf(stderr, "CUDA backend: FD step with %d pixels exceeds the buffers sized at init (%d)\n", M, I.Mcap); exit(1); }
    draw_batch(I, it, M);
    // unperturbed loss for the stats line: one "pair" with sigma 0 -> lsum[0] = L(w)
    {
        const int chunks = (M + TPB * PPT - 1) / (TPB * PPT);
        ensure_part(I, (size_t)2 * chunks);
        dim3 grid(chunks, 2);
        NTC_DISPATCH_MAXW(I, (k_mlp_es_eval<MAXW><<<grid, TPB, I.shmem()>>>(I.d.seed, it, 0.0f, M, I.P, I.d_w, I.d_feat, I.d_btgt, I.d_bidx, I.d_part, chunks)));
        CUDA_CHECK(cudaGetLastError());
        k_mlp_es_finish<<<1, 128>>>(I.d_part, chunks, 1, 3.0 * I.d.wsum * M, I.d_dl, I.d_lsum);   // dl[0] is overwritten by the FD kernels below; lsum[0] = L(w)
        CUDA_CHECK(cudaGetLastError());
    }
    const double norm = 3.0 * I.d.wsum * M;
    const int slice = 512;   // weights per launch: keeps every kernel well under the WDDM timeout
    for (int j0 = 0; j0 < I.P; j0 += slice) {
        int nb = std::min(slice, I.P - j0);
        NTC_DISPATCH_MAXW(I, (k_fd_eval<MAXW><<<nb, TPB, I.shmem()>>>(j0, h, M, I.P, I.d_w, I.d_feat, I.d_btgt, I.d_bidx, norm, I.d_dl, I.d_hh)));
        CUDA_CHECK(cudaGetLastError());
    }
    k_fd_grad<<<blocks_for(I.P, 256), 256>>>(I.P, I.d_dl, I.d_hh, I.d_wgrad);
    CUDA_CHECK(cudaGetLastError());
    adam_step(I.d_w, I.d_wgrad, I.d_wm, I.d_wv, I.P, lr, I.mlp_t);
    I.last_N = I.P; I.last_fd = true;
}

void Trainer::reset_mlp_adam() {
    Impl& I = *im;
    CUDA_CHECK(cudaMemset(I.d_wm, 0, I.P * sizeof(float)));
    CUDA_CHECK(cudaMemset(I.d_wv, 0, I.P * sizeof(float)));
    I.mlp_t = 0;
}

void Trainer::mlp_stats(double& batch_loss, double& diff_std) {
    Impl& I = *im;
    if (I.last_N == 0) { batch_loss = 0; diff_std = 0; return; }
    if (!I.last_fd) {
        std::vector<double> dl(I.last_N), lsum(I.last_N);
        CUDA_CHECK(cudaMemcpy(dl.data(), I.d_dl, I.last_N * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(lsum.data(), I.d_lsum, I.last_N * sizeof(double), cudaMemcpyDeviceToHost));
        double mean = 0, mean2 = 0, lmean = 0;
        for (int i = 0; i < I.last_N; i++) { mean += dl[i]; mean2 += dl[i] * dl[i]; lmean += lsum[i]; }
        mean /= I.last_N; mean2 /= I.last_N; lmean /= I.last_N;
        diff_std = sqrt(std::max(0.0, mean2 - mean * mean));
        batch_loss = lmean;
    } else {
        std::vector<double> dl(I.P); double l0;
        CUDA_CHECK(cudaMemcpy(dl.data(), I.d_dl, I.P * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&l0, I.d_lsum, sizeof(double), cudaMemcpyDeviceToHost));
        double ss = 0; for (int j = 0; j < I.P; j++) ss += dl[j] * dl[j];
        diff_std = sqrt(ss / I.P);
        batch_loss = l0;
    }
}

void Trainer::lat_step(int it, float sigma, float lr, int K) {
    Impl& I = *im;
    const int nlev = I.d.nlev;
    const bool qat = I.d.qat > 0;
    // ES-perturbed levels: every level, or every level but 0 under --qat (level 0 is searched instead)
    unsigned mask = 0;
    for (int l = 0; l < nlev; l++) if (!(qat && l == 0)) mask |= 1u << l;
    CUDA_CHECK(cudaMemsetAsync(I.d_zgrad, 0, I.nz * sizeof(float)));
    const bool rate = I.dct_live && I.dct_rate_es;   // [DCT] --dct-lambda part A (rate_es is false without the flag: no launch)
    if (rate) CUDA_CHECK(cudaMemsetAsync(I.d_dct_cnt + 2, 0, sizeof(int)));   // [DCT] hits
    I.dct_evals = rate ? (int)(I.dct_n * (size_t)K) : 0;   // [DCT] evals = (block, channel, pair) count, kept on the host (no synchronous copy in the step)
    const size_t npix = (size_t)I.d.W * I.d.H;
    const int step = I.lat_step_no;   // noise key (= LatentTrainer::step_no)
    for (int k = 0; k < K; k++) {
        if (!mask) continue;
        dim3 grid(blocks_for(npix, TPB), I.ntex);
        NTC_DISPATCH_MAXW(I, (k_lat_pair<MAXW><<<grid, TPB, I.shmem()>>>(I.d.seed, step, k, mask, sigma, I.d_w, I.d_zdec, I.d_tgt, I.d_dpix, I.P)));
        CUDA_CHECK(cudaGetLastError());
        for (int l = 0; l < nlev; l++) {
            if (!(mask & (1u << l))) continue;
            const float scale = 1.0f / (2.0f * K * sigma);
            const size_t ntx = (size_t)I.d.lv[l].W * I.d.lv[l].H;
            dim3 g2(blocks_for(ntx, 256), I.ntex);
            k_lat_gather<<<g2, 256>>>(I.d.seed, step, k, l, scale, I.d_dpix, I.d_zgrad);
            CUDA_CHECK(cudaGetLastError());
        }
        if (rate) {   // [DCT] the rate pass of this pair (= the host's, after the mse scatter over the levels)
            k_dct_rate_pair<<<blocks_for(I.dct_n, 128), 128>>>(I.d.seed, step, k, sigma, I.d_z, I.d_dct_code, I.d_dct_dbits, I.d_dct_cnt + 2);   // [DCT]
            CUDA_CHECK(cudaGetLastError());   // [DCT]
            const size_t ntx0 = (size_t)I.d.lv[0].W * I.d.lv[0].H;   // [DCT]
            k_dct_rate_gather<<<blocks_for(ntx0, 256), 256>>>(I.d.seed, step, k, 1.0f / (2.0f * K * sigma), I.d_dct_dbits, I.d_zgrad);   // [DCT]
            CUDA_CHECK(cudaGetLastError());   // [DCT]
        }   // [DCT]
    }
    adam_step(I.d_z, I.d_zgrad, I.d_zm, I.d_zv, (int)I.nz, lr, I.lat_t);
    snap_latent_impl(I);   // --qes: rebuild the snapped decode copy (and clamp the shadow once frozen)
    I.lat_step_no++;
}

void Trainer::qat_search() {
    Impl& I = *im;
    const size_t ntx = (size_t)I.d.lv[0].W * I.d.lv[0].H;
    dim3 grid(blocks_for(ntx, TPB), I.ntex);
    NTC_DISPATCH_MAXW(I, (k_qat_search<MAXW><<<grid, TPB, I.shmem()>>>(I.d_w, I.d_zdec, I.d_tgt, I.P)));
    CUDA_CHECK(cudaGetLastError());
    // The search ran on the decode buffer (so it saw the snapped --qes levels); carry its level-0
    // result back to the shadow when the two buffers differ.
    if (I.d_zdec != I.d_z) {
        const size_t n0 = (size_t)I.d.lv[0].W * I.d.lv[0].H * I.d.lv[0].C;
        CUDA_CHECK(cudaMemcpy(I.d_z, I.d_zdec, n0 * sizeof(float), cudaMemcpyDeviceToDevice));
    }
}

void Trainer::debug_last_batch(std::vector<int>& bidx) const {
    const Impl& I = *im;
    bidx.resize(I.last_M);
    CUDA_CHECK(cudaMemcpy(bidx.data(), I.d_bidx, I.last_M * sizeof(int), cudaMemcpyDeviceToHost));
    for (auto& b : bidx) b &= (1 << 28) - 1;
}
void Trainer::debug_last_zgrad(std::vector<float>& g) const {
    const Impl& I = *im;
    g.resize(I.nz);
    CUDA_CHECK(cudaMemcpy(g.data(), I.d_zgrad, I.nz * sizeof(float), cudaMemcpyDeviceToHost));
}
void Trainer::debug_last_dl(std::vector<double>& dl) const {
    const Impl& I = *im;
    dl.resize(I.last_N);
    CUDA_CHECK(cudaMemcpy(dl.data(), I.d_dl, I.last_N * sizeof(double), cudaMemcpyDeviceToHost));
}

} // namespace ntc_cuda
