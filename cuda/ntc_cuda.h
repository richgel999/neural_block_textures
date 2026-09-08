// Host-facing API of the CUDA backend. Plain C++: no CUDA types, safe to include from
// main.cpp. The backend owns the latent, the MLP weights and both Adam states on the
// device for the whole run; the host Decoder is a mirror refreshed by download_model()
// at print/save time so the unchanged host code (stats, bitrate, PNG and model I/O)
// keeps working. Every kernel is written against a small set of textures sharing one
// decoder format (v1 always runs one texture), see CUDA_PLAN_B.md.
#pragma once
#include <stdint.h>
#include <string>
#include <vector>

namespace ntc_cuda {

static const int MAX_LEVELS = 3;
static const int MAX_POS = 16;
static const int MAX_OUT = 12;      // = MAXOUT in main.cpp (3 channels x 4 textures)
static const int MAX_HIDDEN = 8;    // = MAXL
static const int MAX_CH = 16;       // max level-0 channels with per-channel --qat bits
static const int MAX_QES_CH = 64;   // max channels of a --qes level (the v1 envelope has at most 64 MLP inputs)
static const int MAX_DCT_CH = 4;    // [DCT] max selector channels of a --dct-q level 0 (= MAX_DCT_CH in main.cpp)

// Positional feature kinds, in the same order as PosKind in main.cpp (main.cpp passes the integer).
enum PosKindDev { PK_UV, PK_LV1LOCAL };

struct LevelDesc { int W = 0, H = 0, C = 0; size_t off = 0; bool nearest = false; };
struct PosDesc { int kind = 0; };

// Everything the device needs to know about one model; filled by main.cpp.
struct ModelDesc {
    int W = 0, H = 0;            // image size
    int T = 1;                   // textures (nout = 3T)
    int nlev = 0; LevelDesc lv[MAX_LEVELS];
    int npos = 0; PosDesc pos[MAX_POS];
    int nin = 0, nout = 3;
    int nhidden = 0; int hidden[MAX_HIDDEN];
    float leak = 0.01f;          // leaky ReLU slope (the only hidden activation)
    bool clamp_out = false;
    float cw[MAX_OUT];           // per-output-channel loss weight
    float wsum = 1.0f;
    int qat_bits[MAX_CH];        // per level-0 channel; all 0 when --qat is off
    int qat = 0;                 // largest qat bit depth (0 = off)
    const float* qat_grid = nullptr;   // the host's QAT_GRID[9][257]: copied verbatim so on-grid bit patterns match the CPU exactly
    int qes_bits[MAX_LEVELS] = { 0, 0, 0 };   // --qes bits per level (0 = continuous); any nonzero entry allocates the snapped decode copy
    // [DCT] --dct-q: level 0 is a DCT-coded selector plane (dct != 0 allocates the snapped decode copy and the symbol / code buffers)
    int dct = 0, dct_C = 0, dct_N = 8, dct_dc_step = 4;   // [DCT]
    int dct_q[MAX_DCT_CH] = { 0, 0, 0, 0 };               // [DCT] per-channel quality (informational on the device: the step tables arrive with set_dct)
    const float* dct_basis = nullptr;                     // [DCT] the host's DCT_BASIS[8][8] bit patterns, copied verbatim (never recomputed here)
    const int* dct_zigzag = nullptr;                      // [DCT] the host's DCT_ZIGZAG[64] (zigzag position -> natural index), copied for the --dct-lambda token walk
    int max_pairs = 64;          // largest N the MLP ES step will be called with (--mlp-pairs; sizes the per-pair buffers)
    int max_batch = 4096;        // largest minibatch M (--mlp-batch; sizes the per-slot feature / target buffers)
    uint64_t seed = 1;
};

// --qes tables, computed on the host (main.cpp qes_fit / qes_rebuild) and uploaded by set_qes():
// per level and channel the range [lo, hi], range = max(hi - lo, 1e-6) and the offset of the
// channel's 2^bits grid values in the flat grid array. The device recomputes only the index
// (clamp(rn(((z - lo) / range) * levels))) and looks the grid value up, never recomputes it.
struct QesDesc {
    int bits[MAX_LEVELS];        // 0 = level not snapped
    int live;                    // 0 until set_qes: the snapped copy is then a plain copy of the shadow
    int frozen;                  // 1: the shadow is clamped to [lo, hi] on every snap
    float lo[MAX_LEVELS][MAX_QES_CH], hi[MAX_LEVELS][MAX_QES_CH], range[MAX_LEVELS][MAX_QES_CH];
    int grid_off[MAX_LEVELS][MAX_QES_CH];
};

// [DCT] --dct-q level-0 description, uploaded by set_dct() together with the host-fitted scale codes, the symbols and the
// [DCT] integer step tables (C x 16 x 64 ints, host-built from q / N / dc_step / code). live = 0 until set_dct: the device
// [DCT] then only clamps level 0 of the shadow (k_dct_clamp0); live = 1: k_dct_snap after every latent step.
struct DctDesc {   // [DCT]
    int live, C, BW, BH, N, dc_step;   // [DCT]
    // [DCT] --dct-lambda (DCT_RATE_PLAN.md section 4): part A / part B flags, the ES rate weight L / (16 * 65025 * npix) and
    // [DCT] lambda_t(k) = L / (16 g_k^2) per scale code, all host-computed floats (zero without the flag: no kernel, no truncation)
    int rate_es, rate_trunc; float rate_scale; float lam_t16[16];   // [DCT]
    int lo_w16;   // [DCT] --dct-lambda-lo: the first-order pair's token cost is (token * lo_w16 + 8) >> 4 (16 = unweighted); = DctLevel::lo_w16
    int deadzone;   // [DCT] --dct-deadzone: 1 = the dead-zone AC quantizer with the first-order exemption, 0 = plain rounding on every AC; = DctLevel::deadzone
};   // [DCT]

class Trainer {
public:
    Trainer();
    ~Trainer();
    // Allocates device state and uploads the initial latent, weights and target
    // (interleaved W*H*nout floats). Returns false with a reason for anything the
    // v1 backend does not support (oversized MLPs).
    bool init(const ModelDesc& d, const float* z, const float* p, const float* target, std::string& why);
    std::string banner() const;              // device name / arch / toolkit line for the run header

    void upload_model(const float* z, const float* p);   // also rebuilds the snapped decode copy
    void download_model(float* z, float* p);             // the fp32 shadow and the weights
    // --qes: upload the host-fitted ranges and grid table (grid_n floats), then snap (k_snap: zq = grid[index(z)],
    // shadow clamped when frozen). snap_latent() alone rebuilds the copy; download_zq() reads the decode buffer
    // (= the shadow when no level has --qes).
    void set_qes(const QesDesc& q, const float* grid, size_t grid_n);
    void snap_latent();
    void shadow_pull(size_t off, size_t n, float f);   // [DCT] --dct-shadow-pull / --dct-shadow-reset: d_z[off..off+n) += f * (d_zq - d_z), then the snap rebuilds d_zq
    void download_zq(float* zq);
    // [DCT] --dct-q: upload the description, the scale codes ([block][channel] uint8), the symbols ([block][channel][64]
    // [DCT] int16) and the step tables (C * 16 * 64 ints), then rebuild level 0 of the decode copy from the symbols
    // [DCT] (k_dct_recon; the shadow is not touched). download_dct() reads the device's symbols and codes back.
    void set_dct(const DctDesc& d, const uint8_t* code, const int16_t* sym, const int* step);   // [DCT]
    void download_dct(int16_t* sym, uint8_t* code);   // [DCT]
    // [DCT] --dct-lambda figures (every pointer may be null): the proxy bits per (block, channel) of the last k_dct_snap / k_dct_recon,
    // [DCT] the last pair's bit differences of the last lat_step, the truncation count and pre-truncation nonzero ACs of the last
    // [DCT] snap, and the hit count / evaluation count of the last lat_step (0 when part A is off; evals is host state, dct_n * K).
    void download_dct_rate(int32_t* bits, int32_t* dbits_last_pair, int* ntrunc, int* nz_before, int* hits, int* evals);   // [DCT]

    // One antithetic ES step on the MLP (= MlpTrainer::step). N pairs, minibatch of M
    // pixels drawn on the device.
    void mlp_step(int it, float sigma, float lr, int N, int M);
    // One central-differences step (= MlpTrainer::step_fd).
    void mlp_step_fd(int it, float h, float lr, int M);
    void reset_mlp_adam();
    // Stats of the most recent MLP step: mean minibatch loss and dstd (ES) or fd-rms (FD).
    void mlp_stats(double& batch_loss, double& diff_std);

    // One latent ES step (= LatentTrainer::step): K pairs, one sigma for every ES-trained level.
    void lat_step(int it, float sigma, float lr, int K);
    // Exact per-texel search on the discrete level 0 (= qat_search).
    void qat_search();
    // Decode the whole image with the device latent (z_host == null) or a host-supplied
    // latent (e.g. the post-hoc quantized one) into img (W*H*nout floats).
    void decode_full(const float* z_host, float* img);

    // Verification helpers (--cuda-check): the minibatch indices and per-pair loss
    // differences of the last MLP ES step, and the per-weight differences of the last FD step.
    void debug_last_batch(std::vector<int>& bidx) const;
    void debug_last_dl(std::vector<double>& dl) const;
    void debug_last_zgrad(std::vector<float>& g) const;   // the latent ES gradient of the last lat_step (before Adam)

    struct Impl;   // device state (public so the file-local launch helpers can reach it)
    Impl* im;
};

} // namespace ntc_cuda
