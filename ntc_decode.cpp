// ntc_decode: standalone SSE4.1 multithreaded CPU decoder for the model files written by ntc (main.cpp).
//
// [DCT] Reads a v9..v13 model.bin, quantizes the levels the trainer left continuous to 8 bits exactly as
// its bitrate_stats does (so the output matches recon_q_final.png; levels quantized in training,
// --qat level 0 and v12 / v13 --qes levels, are stored on-grid and used as they are; [DCT] a v13 file's
// [DCT] DCT-coded level 0 is reconstructed from its symbols by the trainer's strict-FP IDCT), evaluates the
// per-texel MLP with SSE4.1 in a
// texels-in-lanes layout that keeps the trainer's strictly sequential accumulation order, and
// writes RGB8 PNGs. A scalar strict-fp reference path (a line-for-line port of features() and
// mlp_forward()) is built in for --verify and as a fallback for model features the SIMD path
// does not cover.
//
// DEPENDENCY: this file mirrors main.cpp's save_model() / loader (v12: per-level --qes bits and
// ranges; version floor v9), bilinear_tap(), sample_latent(), PosEnc::encode() (positional kinds
// `uv`, `lv1local`, `none` only), features(), mlp_forward(), the leaky ReLU (activate()),
// QAT_GRID, the --qes rule (a level marked quantized in the file holds its snapped decode values
// and is never re-quantized here) and the 8-bit latent quantization in bitrate_stats() for the
// remaining levels. main.cpp carries the matching note at its top; a change on either side must be
// re-verified with --compare against the trainer's recon_q_final.png and with --verify.
// [DCT] It also mirrors the experimental --dct-q level 0 (main.cpp "DCT" region): the DCT_BASIS_BITS /
// [DCT] DCT_AK_BITS literals, dct_build_steps(), dct_inv8() (zero-skipping, bit-exact), dct_dequant_dc/ac() and
// [DCT] dct_recon_level0() (zq = clamp(w' / 32 - 1); here the block rows are split across the decoder's thread pool with per-thread stats, a decoder-local wrapper outside the mirrored math), plus the v13 fields (N, dc_step, C, q[C]) and payload (int16
// [DCT] symbols, uint8 scale codes, then the floats of levels >= 1). The strict block here has no fp_contract(off)
// [DCT] on MSVC; this target is built without /arch, so no FMA can be emitted (CMakeLists.txt).
// [NTCB] And the bit-packed container (main.cpp "NTCB" region, --write-ntcb, NTCB_PLAN.md section 1): load_ntcb here is
// [NTCB] a hand-written mirror of main.cpp's ntcb_read_header / ntcb_restore (header fields and ranges, level records, section
// [NTCB] headers, LSB-first bit packing, the grid dequantization lo + k / (float)levels * range in strict FP, the DCT token
// [NTCB] stream, the fp16 MLP). A change on either side must be re-verified with --compare / --verify on a .ntcb (NTCB_NOTES.md).
//
// Build: see CMakeLists.txt (MSVC: /O2 /fp:fast, no /arch flag; GCC/Clang: -O3 -msse4.1 -ffast-math (the /O2 / -O3 only in the non-Debug configurations; a Debug build, ntc_decode_d, is unoptimized)).
// Everything that must be bit-exact with the trainer (geometry tables, quantization, the scalar
// reference) is compiled with strict floating point via STRICT_FP_BEGIN/END; the SIMD kernel is
// explicit intrinsics and is unaffected by the compiler's fast-math mode.

#define _CRT_SECURE_NO_WARNINGS
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#include <smmintrin.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#if defined(_MSC_VER) && !defined(__clang__)
#define STRICT_FP_BEGIN __pragma(float_control(precise, on, push)) __pragma(fp_contract(off))   // [DCT] contraction off: the DCT / table code has real multiply-add pairs
#define STRICT_FP_END   __pragma(float_control(pop))
#elif defined(__clang__)
#define STRICT_FP_BEGIN _Pragma("float_control(precise, on, push)")
#define STRICT_FP_END   _Pragma("float_control(pop)")
#else
#define STRICT_FP_BEGIN _Pragma("GCC push_options") _Pragma("GCC optimize(\"no-fast-math\")")
#define STRICT_FP_END   _Pragma("GCC pop_options")
#endif

static const int MAXT = 4;          // textures per material (main.cpp MAXT)
static const int MAXOUT = 3 * MAXT; // MLP outputs
static const int MAXH = 128;        // max units in any layer, incl. the input (main.cpp MAXH)
static const int MAXL = 8;          // max hidden layers
static const int MAXLV = 3;         // max latent levels

// ---------------------------------------------------------------- model file
struct Level { int W = 0, H = 0, C = 0; bool nearest = false; size_t off = 0; size_t size() const { return (size_t)W * H * C; } };

struct Model {
    int version = 0;
    int nin = 0, act = 0, T = 1;   // act: the file's activation int; must be 0 (leaky) since v0.8
    float leak = 0.01f;
    int deblock = 0; float deblock_falloff = 1.0f;
    int qat_bits = 0; std::vector<int> qat_ch;
    std::vector<int> qes_bits;                       // v12: per level, --qes bits (0 = continuous); older files all 0
    std::vector<std::vector<float>> qes_lo, qes_hi;  // v12: per level, per channel range (informational: the stored values are on-grid)
    // [DCT] v13: level 0 is a DCT-coded selector plane (dct_N = 0 otherwise); symbols [block][channel][64], codes [block][channel]
    int dct_N = 0, dct_dc_step = 0, dct_C = 0; std::vector<int> dct_q;   // [DCT]
    int dct_deadzone = 1;   // [DCT] v14 / ntcb v2: the AC quantizer that produced the symbols (1 = dead zone with the first-order exemption, 0 = plain rounding); 1 for a v13 / ntcb v1 file
    std::vector<int16_t> dct_sym; std::vector<uint8_t> dct_code;   // [DCT]
    std::vector<Level> lv;
    std::vector<int> hidden;
    std::string pos;
    std::vector<float> z, p;
    int imgW = 0, imgH = 0, srcW = 0, srcH = 0, clamp = -1;   // v11 only
    // [NTCB] the bit-packed container (main.cpp "NTCB" region, NTCB_PLAN.md): per level its coding mode (0 q8 / 1 qes / 2 qat / 3 dct) and bit depth,
    // [NTCB] and the size terms of the file line; version is set to 13 so the v11+ size / crop / clamp branches apply
    bool ntcb = false; int ntcb_version = 0; std::vector<int> ntcb_mode, ntcb_bits;   // [NTCB]
    size_t ntcb_bytes = 0; int ntcb_H = 0, ntcb_nsections = 0; double ntcb_content_bits = 0, ntcb_pad_bits = 0, ntcb_sync_bits = 0;   // [NTCB]
    std::vector<int> widths() const { std::vector<int> w; w.push_back(nin); for (int h : hidden) w.push_back(h); w.push_back(3 * T); return w; }
    size_t nparams() const { std::vector<int> w = widths(); size_t n = 0; for (size_t l = 1; l < w.size(); l++) n += (size_t)w[l] * w[l - 1] + w[l]; return n; }
    size_t mlp_macs() const { std::vector<int> w = widths(); size_t n = 0; for (size_t l = 1; l < w.size(); l++) n += (size_t)w[l] * w[l - 1]; return n; }
    int bilinear_macs() const { int n = 0; for (const Level& L : lv) if (!L.nearest) n += 4 * L.C; return n; }
};

struct Reader {
    const uint8_t* p; size_t n, pos = 0; bool ok = true;
    Reader(const uint8_t* p_, size_t n_) : p(p_), n(n_) {}
    bool get(void* dst, size_t k) { if (!ok || pos + k > n) { ok = false; return false; } memcpy(dst, p + pos, k); pos += k; return true; }
    int i32() { int v = 0; get(&v, 4); return v; }
    float f32() { float v = 0; get(&v, 4); return v; }
};

static const uint32_t NTCB_MAGIC = 0x6263746Eu;   // [NTCB] the bytes "ntcb" as a little-endian uint32 (main.cpp NTCB_MAGIC)
static bool load_ntcb(const std::vector<uint8_t>& buf, Model& m, std::string& err);   // [NTCB] defined in the NTCB region (strict-FP section)
static bool load_model(const std::string& path, Model& m, std::string& err) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { err = "cannot open " + path; return false; }
    std::vector<uint8_t> buf;
    { fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET); if (sz <= 0) { fclose(f); err = "empty file"; return false; } buf.resize((size_t)sz); if (fread(buf.data(), 1, buf.size(), f) != buf.size()) { fclose(f); err = "read error"; return false; } fclose(f); }
    Reader r(buf.data(), buf.size());
    int magic = r.i32();
    if (buf.size() >= 4 && buf[0] == 0x6Eu && buf[1] == 0x74u && buf[2] == 0x63u && buf[3] == 0x62u) return load_ntcb(buf, m, err);   // [NTCB] hook: the container's magic bytes "ntcb" (byte-wise, no host-order read), parsed by its own reader
    if (!r.ok || (magic & 0xFFFFFF00) != 0x4E544300) { err = "not an ntc model file (bad magic)"; return false; }
    int v = magic - 0x4E544330;
    if (v < 9 || v > 14) { err = "unsupported model version v" + std::to_string(v) + " (v9..v14 supported; older layouts were retired in v0.8)"; return false; }   // [DCT] hook: v14
    m.version = v;
    int LW0 = r.i32(), LH0 = r.i32(), LC0 = r.i32();
    m.nin = r.i32();
    int poslen = r.i32();
    m.act = r.i32();
    int nhidden = r.i32();
    int nlevels = r.i32();
    m.T = r.i32();
    if (!r.ok) { err = "truncated header"; return false; }
    if (nhidden < 1 || nhidden > MAXL || nlevels < 1 || nlevels > MAXLV || m.T < 1 || m.T > MAXT || LC0 < 1 || LC0 > MAXH || LW0 < 1 || LH0 < 1 || poslen < 0 || poslen >= 4096)
        { err = "header out of range"; return false; }
    std::vector<int> filt(nlevels);
    for (int l = 0; l < nlevels; l++) { filt[l] = r.i32(); if (filt[l] != 0 && filt[l] != 1) { err = "bad filter flag"; return false; } }
    m.deblock = r.i32(); m.deblock_falloff = r.f32();   // v7 pair (layout only; a set deblock flag is refused by main)
    m.leak = r.f32();
    m.qat_bits = r.i32(); if (m.qat_bits < 0 || m.qat_bits > 8) { err = "bad qat bits"; return false; }
    m.qat_ch.assign(LC0, m.qat_bits);
    if (v >= 10) {
        int mx = 0;
        for (int c = 0; c < LC0; c++) { m.qat_ch[c] = r.i32(); if (m.qat_ch[c] < 0 || m.qat_ch[c] > 8) { err = "bad per-channel qat bits"; return false; } mx = std::max(mx, m.qat_ch[c]); }
        if (mx != m.qat_bits) { err = "qat max field disagrees with the per-channel list"; return false; }
    }
    if (m.qat_bits == 0) m.qat_ch.clear();
    if (v >= 11) {
        m.imgW = r.i32(); m.imgH = r.i32(); m.srcW = r.i32(); m.srcH = r.i32(); m.clamp = r.i32();
        if (m.imgW < 1 || m.imgH < 1 || m.srcW < 1 || m.srcH < 1 || (m.clamp != 0 && m.clamp != 1)) { err = "bad v11 image fields"; return false; }
    }
    m.lv.resize(nlevels);
    m.lv[0].W = LW0; m.lv[0].H = LH0; m.lv[0].C = LC0;
    for (int l = 1; l < nlevels; l++) { m.lv[l].W = r.i32(); m.lv[l].H = r.i32(); m.lv[l].C = r.i32(); if (m.lv[l].W < 1 || m.lv[l].H < 1 || m.lv[l].C < 1 || m.lv[l].C > MAXH) { err = "bad level dims"; return false; } }
    m.qes_bits.assign(nlevels, 0); m.qes_lo.resize(nlevels); m.qes_hi.resize(nlevels);
    if (v >= 12) for (int l = 0; l < nlevels; l++) {   // per level: --qes bits, then C lo and C hi floats
        m.qes_bits[l] = r.i32();
        if (!r.ok || !(m.qes_bits[l] == 0 || (m.qes_bits[l] >= 2 && m.qes_bits[l] <= 12))) { err = "bad qes bits"; return false; }
        m.qes_lo[l].resize(m.lv[l].C); m.qes_hi[l].resize(m.lv[l].C);
        for (int c = 0; c < m.lv[l].C; c++) m.qes_lo[l][c] = r.f32();
        for (int c = 0; c < m.lv[l].C; c++) m.qes_hi[l][c] = r.f32();
        if (!r.ok) { err = "truncated qes ranges"; return false; }
        if (m.qes_bits[l] > 0 && l == 0 && !m.qat_ch.empty()) { err = "level 0 marked both --qat and --qes"; return false; }
    }
    if (v >= 13) {   // [DCT] N, dc_step, C (= level-0 channels), then C ints q; level 0 must be a full-resolution nearest level, a multiple of N, neither --qat nor --qes
        m.dct_N = r.i32(); m.dct_dc_step = r.i32(); m.dct_C = r.i32();   // [DCT]
        if (!r.ok || m.dct_N != 8 || m.dct_dc_step < 1 || m.dct_dc_step > 64 || m.dct_C != LC0 || m.dct_C < 1 || m.dct_C > 4) { err = "bad v13 dct fields"; return false; }   // [DCT]
        if (v >= 14) { m.dct_deadzone = r.i32(); if (!r.ok || (m.dct_deadzone != 0 && m.dct_deadzone != 1)) { err = "bad v14 dct_deadzone field (0 or 1)"; return false; } }   // [DCT] v14: the --dct-deadzone flag after the three DCT ints (a v13 file: 1)
        m.dct_q.resize(m.dct_C);   // [DCT]
        for (int c = 0; c < m.dct_C; c++) { m.dct_q[c] = r.i32(); if (m.dct_q[c] < 1 || m.dct_q[c] > 100) { err = "bad v13 dct quality"; return false; } }   // [DCT]
        if (!r.ok) { err = "truncated v13 dct fields"; return false; }   // [DCT]
        if (!m.qat_ch.empty() || m.qes_bits[0] > 0) { err = "v13 level 0 marked --qat or --qes as well as DCT-coded"; return false; }   // [DCT]
        if (filt[0] != 1 || m.imgW != LW0 || m.imgH != LH0 || m.imgW % 8 != 0 || m.imgH % 8 != 0) { err = "v13 level 0 must be a full-resolution nearest level with a size that is a multiple of 8"; return false; }   // [DCT]
    }   // [DCT]
    size_t off = 0;
    for (int l = 0; l < nlevels; l++) { m.lv[l].nearest = filt[l] == 1; m.lv[l].off = off; off += m.lv[l].size(); }
    m.hidden.resize(nhidden);
    for (int l = 0; l < nhidden; l++) { m.hidden[l] = r.i32(); if (m.hidden[l] < 1 || m.hidden[l] > MAXH) { err = "bad hidden width"; return false; } }
    m.pos.resize(poslen);
    if (poslen) r.get(&m.pos[0], poslen);
    if (!r.ok) { err = "truncated header"; return false; }
    if (m.nin < 1 || m.nin > MAXH) { err = "bad nin"; return false; }
    m.z.resize(off + 4);   // 4 floats of slack: the SIMD sampler reads whole 4-channel vectors
    if (m.dct_N > 0) {   // [DCT] v13 payload: symbols, codes, then the floats of levels >= 1; level 0 stays zero until dct_recon_level0
        const size_t nblk = (size_t)(m.imgW / 8) * (m.imgH / 8), n0 = m.lv[0].size();   // [DCT]
        m.dct_sym.resize(nblk * m.dct_C * 64); m.dct_code.resize(nblk * m.dct_C);   // [DCT]
        if (!r.get(m.dct_sym.data(), m.dct_sym.size() * 2)) { err = "truncated dct symbols"; return false; }   // [DCT]
        if (!r.get(m.dct_code.data(), m.dct_code.size())) { err = "truncated dct scale codes"; return false; }   // [DCT]
        for (uint8_t k : m.dct_code) if (k > 15) { err = "bad dct scale code"; return false; }   // [DCT]
        if (!r.get(m.z.data() + n0, (off - n0) * 4)) { err = "truncated latent"; return false; }   // [DCT]
    } else   // [DCT]
    if (!r.get(m.z.data(), off * 4)) { err = "truncated latent"; return false; }
    m.p.resize(m.nparams());
    if (!r.get(m.p.data(), m.p.size() * 4)) { err = "truncated MLP parameters"; return false; }
    if (r.pos != r.n) { err = "trailing bytes after the MLP parameters (" + std::to_string(r.n - r.pos) + ")"; return false; }
    return true;
}

// ---------------------------------------------------------------- positional encoding spec
enum PosKind { POS_UV, POS_LV1LOCAL, POS_COUNT };   // same order as PosKind in main.cpp (the other kinds were removed in v0.8)
static const char* POS_NAMES[POS_COUNT] = { "uv", "lv1local" };
struct PosFeature { PosKind kind; };

static bool parse_pos(const std::string& s, std::vector<PosFeature>& feats) {
    feats.clear();
    if (s == "none" || s.empty()) return true;
    for (size_t a = 0; a < s.size();) {
        size_t e = s.find(',', a); if (e == std::string::npos) e = s.size();
        std::string item = s.substr(a, e - a); a = e + 1;
        if (item.empty()) continue;
        PosFeature f;
        int k = 0; while (k < POS_COUNT && item != POS_NAMES[k]) k++;   // a removed kind (or a ":N" suffix) matches no name
        if (k == POS_COUNT) return false;
        f.kind = (PosKind)k;
        feats.push_back(f);
    }
    return true;
}
static int pos_count(const PosFeature&) { return 2; }   // both kinds are an (x, y) pair

// ---------------------------------------------------------------- strict-fp section: geometry, quantization, scalar reference
STRICT_FP_BEGIN

struct Tap { int x0, x1, y0, y1; float fx, fy; };

// main.cpp bilinear_tap (L318-339), verbatim semantics.
static Tap ref_tap(const Level& L, float u, float v) {
    Tap t;
    if (L.nearest) {
        float x = u * L.W, y = v * L.H;
        int ix = (int)std::floor(x), iy = (int)std::floor(y);
        t.fx = x - ix; t.fy = y - iy;
        t.x0 = t.x1 = std::max(0, std::min(L.W - 1, ix));
        t.y0 = t.y1 = std::max(0, std::min(L.H - 1, iy));
        return t;
    }
    float x = u * L.W - 0.5f, y = v * L.H - 0.5f;
    int x0 = (int)std::floor(x), y0 = (int)std::floor(y);
    t.fx = x - x0; t.fy = y - y0;
    t.x0 = std::max(0, std::min(L.W - 1, x0));
    t.x1 = std::max(0, std::min(L.W - 1, x0 + 1));
    t.y0 = std::max(0, std::min(L.H - 1, y0));
    t.y1 = std::max(0, std::min(L.H - 1, y0 + 1));
    return t;
}
// main.cpp sample_latent (L341-354).
static void ref_sample(const Level& L, const float* z, const Tap& t, float* out) {
    if (L.nearest) { const float* a = &z[((size_t)t.y0 * L.W + t.x0) * L.C]; for (int k = 0; k < L.C; k++) out[k] = a[k]; return; }
    const float* a = &z[((size_t)t.y0 * L.W + t.x0) * L.C];
    const float* b = &z[((size_t)t.y0 * L.W + t.x1) * L.C];
    const float* c = &z[((size_t)t.y1 * L.W + t.x0) * L.C];
    const float* d = &z[((size_t)t.y1 * L.W + t.x1) * L.C];
    float w00 = (1 - t.fx) * (1 - t.fy), w10 = t.fx * (1 - t.fy);
    float w01 = (1 - t.fx) * t.fy, w11 = t.fx * t.fy;
    for (int k = 0; k < L.C; k++) out[k] = w00 * a[k] + w10 * b[k] + w01 * c[k] + w11 * d[k];
}
// main.cpp PosEnc::encode (L575-653).
static int ref_pos_encode(const std::vector<PosFeature>& feats, float u, float v, const Tap& t, const Tap* t1, float* f) {
    int k = 0;
    for (const PosFeature& pf : feats) {
        switch (pf.kind) {
        case POS_UV: f[k++] = u * 2.0f - 1.0f; f[k++] = v * 2.0f - 1.0f; break;
        case POS_LV1LOCAL: f[k++] = t1->fx * 2.0f - 1.0f; f[k++] = t1->fy * 2.0f - 1.0f; break;
        default: break;
        }
    }
    return k;
}
// main.cpp Decoder::features (L733-748).
static void ref_features(const Model& m, const std::vector<PosFeature>& feats, int W, int H, int px, int py, float* f) {
    float u = (px + 0.5f) / W, v = (py + 0.5f) / H;
    const Level& L0 = m.lv[0];
    Tap t = ref_tap(L0, u, v);
    ref_sample(L0, m.z.data(), t, f);
    int k = L0.C;
    Tap t1;
    for (size_t l = 1; l < m.lv.size(); l++) {
        const Level& L = m.lv[l];
        Tap tl = ref_tap(L, u, v);
        if (l == 1) t1 = tl;
        ref_sample(L, m.z.data() + L.off, tl, f + k);
        k += L.C;
    }
    ref_pos_encode(feats, u, v, t, m.lv.size() > 1 ? &t1 : nullptr, f + k);
}
// main.cpp activate: leaky ReLU (the only hidden activation since v0.8).
static inline float ref_activate(float x, float leak) { return x > 0 ? x : leak * x; }
// main.cpp mlp_forward (L433-456); also returns the pre-nonlinearity outputs.
static void ref_mlp(const Model& m, const float* in, float* pre, float* out, bool clamp_out) {
    float bufA[MAXH], bufB[MAXH];
    const float* cur = in; int ncur = m.nin; float* nxt = bufA; const float* p = m.p.data();
    for (size_t l = 0; l < m.hidden.size(); l++) {
        int nh = m.hidden[l];
        const float* W = p; const float* b = W + (size_t)nh * ncur;
        for (int j = 0; j < nh; j++) {
            float s = b[j]; const float* w = W + (size_t)j * ncur;
            for (int i = 0; i < ncur; i++) s += w[i] * cur[i];
            nxt[j] = ref_activate(s, m.leak);
        }
        p = b + nh; cur = nxt; ncur = nh; nxt = (nxt == bufA) ? bufB : bufA;
    }
    const int nout = 3 * m.T;
    const float* W = p; const float* b = W + (size_t)nout * ncur;
    for (int j = 0; j < nout; j++) {
        float s = b[j]; const float* w = W + (size_t)j * ncur;
        for (int i = 0; i < ncur; i++) s += w[i] * cur[i];
        pre[j] = s;
        if (clamp_out) out[j] = std::min(1.0f, std::max(0.0f, s + 0.5f));
        else out[j] = 1.0f / (1.0f + std::exp(-s));
    }
}
// main.cpp save_png quantization (L217).
static inline unsigned char ref_q8(float v) { return (unsigned char)std::lround(std::min(1.0f, std::max(0.0f, v)) * 255.0f); }

// --qat index of a level-0 value (main.cpp qat_index, L1143-1146). The stored level-0 values are used as
// they are: they are the trainer's own grid points (computed under its fast-math build, e.g. 4-bit
// 0x3f7ffffe rather than 1.0f), and the trainer's loader re-snaps to that same grid, so nothing moves.
static inline int qat_index(float v, int bits) { const int levels = (1 << bits) - 1; int k = (int)std::lround((v + 1.0f) * 0.5f * levels); return std::max(0, std::min(levels, k)); }
// A level the trainer quantized itself: --qat level 0 or a v12 --qes level. Its stored values are the
// trainer's own grid points (snapped decode copy) and are used as they are.
static inline bool level_quantized(const Model& m, size_t l) { return (l == 0 && !m.qat_ch.empty()) || m.qes_bits[l] > 0 || (l == 0 && m.dct_N > 0); }   // [DCT] hook: || DCT-coded level 0
// Post-hoc 8-bit latent quantization, main.cpp bitrate_stats: every level the trainer left continuous.
static void q8_quantize(Model& m, int qbits) {
    for (size_t l = 0; l < m.lv.size(); l++) {
        const Level& L = m.lv[l];
        if (level_quantized(m, l)) continue;
        float* z = m.z.data() + L.off;
        const size_t nl = L.size();
        const int levels = (1 << qbits) - 1;
        for (int c = 0; c < L.C; c++) {
            float lo = 1e30f, hi = -1e30f;
            for (size_t i = c; i < nl; i += L.C) { lo = std::min(lo, z[i]); hi = std::max(hi, z[i]); }
            float range = std::max(hi - lo, 1e-6f);
            for (size_t i = c; i < nl; i += L.C) {
                int qi = (int)std::lround((z[i] - lo) / range * levels);
                qi = std::max(0, std::min(levels, qi));
                z[i] = lo + qi / (float)levels * range;
            }
        }
    }
}

// ---------------------------------------------------------------- DCT: transform-coded level 0 [DCT]
// [DCT] Verbatim copies of main.cpp's DCT region (the tables as the same float bit patterns, the same source text
// [DCT] for the step builder, the inverse transform and the dequantizers, under the same strict-FP macro), so the
// [DCT] level-0 plane decoded here is bit for bit the trainer's dct_recon_level0 / the device's k_dct_recon.
static const uint32_t DCT_BASIS_BITS[64] = {
    0x3EB504F3u, 0x3EB504F3u, 0x3EB504F3u, 0x3EB504F3u, 0x3EB504F3u, 0x3EB504F3u, 0x3EB504F3u, 0x3EB504F3u,
    0x3EFB14BEu, 0x3ED4DB31u, 0x3E8E39DAu, 0x3DC7C5C2u, 0xBDC7C5C2u, 0xBE8E39DAu, 0xBED4DB31u, 0xBEFB14BEu,
    0x3EEC835Eu, 0x3E43EF15u, 0xBE43EF15u, 0xBEEC835Eu, 0xBEEC835Eu, 0xBE43EF15u, 0x3E43EF15u, 0x3EEC835Eu,
    0x3ED4DB31u, 0xBDC7C5C2u, 0xBEFB14BEu, 0xBE8E39DAu, 0x3E8E39DAu, 0x3EFB14BEu, 0x3DC7C5C2u, 0xBED4DB31u,
    0x3EB504F3u, 0xBEB504F3u, 0xBEB504F3u, 0x3EB504F3u, 0x3EB504F3u, 0xBEB504F3u, 0xBEB504F3u, 0x3EB504F3u,
    0x3E8E39DAu, 0xBEFB14BEu, 0x3DC7C5C2u, 0x3ED4DB31u, 0xBED4DB31u, 0xBDC7C5C2u, 0x3EFB14BEu, 0xBE8E39DAu,
    0x3E43EF15u, 0xBEEC835Eu, 0x3EEC835Eu, 0xBE43EF15u, 0xBE43EF15u, 0x3EEC835Eu, 0xBEEC835Eu, 0x3E43EF15u,
    0x3DC7C5C2u, 0xBE8E39DAu, 0x3ED4DB31u, 0xBEFB14BEu, 0x3EFB14BEu, 0xBED4DB31u, 0x3E8E39DAu, 0xBDC7C5C2u,
};
static const uint32_t DCT_AK_BITS[16] = {
    0x40924925u, 0x40410671u, 0x3FFEB2BFu, 0x3FA809C5u, 0x3F5DBA49u, 0x3F124925u, 0x3EC10671u, 0x3E7EB2BFu,
    0x3E2809C5u, 0x3DDDBA49u, 0x3D924925u, 0x3D410671u, 0x3CFEB2BFu, 0x3CA809C5u, 0x3C5DBA49u, 0x3C124925u,
};
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
static float DCT_BASIS[8][8];
static float DCT_AK[16];
static int DCT_ZIGZAG[64];
static void dct_init_tables() {
    memcpy(&DCT_BASIS[0][0], DCT_BASIS_BITS, sizeof(DCT_BASIS));
    memcpy(DCT_AK, DCT_AK_BITS, sizeof(DCT_AK));
    int idx = 0;
    for (int s = 0; s < 15; s++) {
        const int x_start = (s < 8) ? 0 : (s - 7), x_end = (s < 8) ? s : 7, n = x_end - x_start + 1;
        int diag[8];
        for (int x = x_start, j = 0; x <= x_end; x++, j++) diag[j] = x + (s - x) * 8;
        if (s & 1) for (int k = n - 1; k >= 0; k--) DCT_ZIGZAG[idx++] = diag[k];
        else for (int k = 0; k < n; k++) DCT_ZIGZAG[idx++] = diag[k];
    }
}
static float dct_quality_scale(int q) {
    q = std::max(1, std::min(100, q));
    float S = q < 50 ? 5000.0f / (float)q : 200.0f - 2.0f * (float)q;
    S *= (1.0f / 100.0f);
    return S;
}
static float dct_level_scale(int q, int k) { return dct_quality_scale(q) * DCT_AK[k]; }
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
static void dct_build_steps(int q, int N, int dc_step, int* out /* 16 * N * N */) {
    for (int k = 0; k < 16; k++) dct_build_steps_scaled(q, dct_level_scale(q, k), N, dc_step, out + (size_t)k * N * N);
}
// Inverse 8x8 DCT, the trainer's dct_inv8 with zero coefficients skipped in the first pass (bit-exact: adding 0.0f * B to a
// float sum leaves it unchanged, and the sum never becomes -0.0f). Returns the number of multiply-adds performed.
static int dct_inv8(const float* C, float* w) {
    float T[64]; int macs = 0;
    for (int x = 0; x < 8; x++)
        for (int v = 0; v < 8; v++) { float s = 0.0f; for (int u = 0; u < 8; u++) { if (C[u * 8 + v] == 0.0f) continue; s += DCT_BASIS[u][x] * C[u * 8 + v]; macs++; } T[x * 8 + v] = s; }
    for (int x = 0; x < 8; x++)
        for (int y = 0; y < 8; y++) { float s = 0.0f; for (int v = 0; v < 8; v++) s += DCT_BASIS[v][y] * T[x * 8 + v]; w[x * 8 + y] = s; }
    return macs + 512;
}
static inline float dct_clampf(float v, float lo, float hi) { if (!(v >= lo)) return lo; if (v > hi) return hi; return v; }
static inline float dct_dequant_dc(int q, int S) { return (float)q * (float)S; }
static inline bool dct_first_order(int u, int v) { return (u == 1 && v == 0) || (u == 0 && v == 1); }
static inline float dct_dequant_ac(int q, int L, int u, int v, bool dz) {   // [DCT] hook: dz = the file's dct_deadzone flag (false: q * L on every AC, main.cpp dct_dequant_ac)
    if (!dz || dct_first_order(u, v)) return (float)q * (float)L;   // [DCT] hook: `!dz ||`
    if (q == 0) return 0.0f;
    const float tau = 0.5f * (float)L;
    const float mag = tau + (float)std::abs(q) * (float)L;
    return q < 0 ? -mag : mag;
}
// Level 0 of m.z from the symbols: per block and channel dequantize with the block's step table, inverse DCT, clamp(w' / 32 - 1).
// Also the cheap statistics: nonzero ACs, last-nonzero zigzag index, EOB-only blocks, clamped texels, code histogram, MACs.
struct DctReconStats { size_t nblk = 0, nnz = 0, eob0 = 0, clamped = 0, macs = 0, codehist[16] = { 0 }; double lnz_sum = 0; size_t lnz_n = 0, neob = 0; int dc_raw_bits = 8;   // [DCT] hook: was one line, closed here
    size_t nnz_c[4] = { 0, 0, 0, 0 }, eob0_c[4] = { 0, 0, 0, 0 }; };   // [DCT] the nonzero ACs and EOB-only blocks per channel (the per-channel line when dct_C > 1)
static DctReconStats dct_recon_level0(Model& m, int nthreads, const std::function<void(int, std::function<void(int)>)>& run) {   // [DCT] block rows are split across nthreads via run(nthreads, job) (the decode stage's pool); every block writes only its own 8x8 region; the stats are per-thread and summed
    DctReconStats S;
    dct_init_tables();
    const Level& L0 = m.lv[0];
    const int BW = m.imgW / 8, BH = m.imgH / 8, C = m.dct_C;
    std::vector<int> step((size_t)C * 16 * 64);
    for (int c = 0; c < C; c++) dct_build_steps(m.dct_q[c], 8, m.dct_dc_step, &step[(size_t)c * 16 * 64]);
    { const int lv = 512 / m.dct_dc_step + 1; int b = 0; while ((1 << b) < lv) b++; S.dc_raw_bits = b; }
    S.nblk = (size_t)BW * BH;
    float* z = m.z.data() + L0.off;
    nthreads = std::max(1, std::min(nthreads, BH));   // the caller already clamps; kept so the split is safe on its own
    std::vector<DctReconStats> part((size_t)nthreads);
    auto job = [&](int tid) {
      DctReconStats& Sp = part[(size_t)tid];   // this thread's share; summed into S below
      const int by0 = (int)((long long)BH * tid / nthreads), by1 = (int)((long long)BH * (tid + 1) / nthreads);
      for (int by = by0; by < by1; by++)
        for (int bx = 0; bx < BW; bx++)
            for (int c = 0; c < C; c++) {
                const size_t i = ((size_t)by * BW + bx) * C + c;
                const int16_t* sym = &m.dct_sym[i * 64];
                const int* st = &step[((size_t)c * 16 + m.dct_code[i]) * 64];
                Sp.codehist[m.dct_code[i]]++;
                float Cf[64], w[64];
                int nz = 0, lnz = 0;
                for (int k = 0; k < 64; k++) Cf[k] = k == 0 ? dct_dequant_dc(sym[0], m.dct_dc_step) : dct_dequant_ac(sym[k], st[k], k / 8, k % 8, m.dct_deadzone != 0);   // [DCT] hook: the deadzone flag
                for (int p = 1; p < 64; p++) if (sym[DCT_ZIGZAG[p]] != 0) { nz++; lnz = p; }
                Sp.nnz += nz; if (nz == 0) Sp.eob0++; else { Sp.lnz_sum += lnz; Sp.lnz_n++; }
                Sp.nnz_c[c] += nz; if (nz == 0) Sp.eob0_c[c]++;   // [DCT]
                if (lnz < 63) Sp.neob++;
                Sp.macs += dct_inv8(Cf, w);
                for (int y = 0; y < 8; y++)
                    for (int x = 0; x < 8; x++) {
                        const size_t idx = ((size_t)(by * 8 + y) * L0.W + (bx * 8 + x)) * L0.C + c;
                        const float raw = w[y * 8 + x] / 32.0f - 1.0f;
                        const float v = dct_clampf(raw, -1.0f, 1.0f);
                        if (v != raw) Sp.clamped++;
                        z[idx] = v;
                    }
            }
    };
    run(nthreads, job);
    for (int t = 0; t < nthreads; t++) { const DctReconStats& P = part[(size_t)t];
        S.nnz += P.nnz; S.eob0 += P.eob0; S.clamped += P.clamped; S.macs += P.macs; S.lnz_sum += P.lnz_sum; S.lnz_n += P.lnz_n; S.neob += P.neob;
        for (int k = 0; k < 16; k++) S.codehist[k] += P.codehist[k];
        for (int k = 0; k < 4; k++) { S.nnz_c[k] += P.nnz_c[k]; S.eob0_c[k] += P.eob0_c[k]; } }
    return S;
}
// ---------------------------------------------------------------- end of the DCT region [DCT]

// ---- NTCB: bit-packed container [NTCB]
// [NTCB] Reader for the container main.cpp's ntcb_write produces (NTCB_PLAN.md section 1; the layout is documented there and in
// [NTCB] main.cpp's NTCB region, which this is a hand-written mirror of). Every multi-byte field is read byte by byte, little-endian;
// [NTCB] bit fields are LSB-first. Inside the strict-FP section: the grid dequantization lo + k / (float)levels * range must produce the
// [NTCB] trainer's qes_rebuild bit patterns. Six-bit sync markers open and close every level body and the MLP body and open every
// [NTCB] channel of a DCT body; each is checked and a mismatch is refused by name. The hash field covers the whole file (the field
// [NTCB] itself read as zero). Runs only on the "ntcb" magic; nothing else in this file changes for a model.bin.
static const uint32_t NTCB_SYNC_LEVEL = 0x2Bu, NTCB_SYNC_CHANNEL = 0x16u, NTCB_SYNC_MLP = 0x31u, NTCB_SYNC_END = 0x0Cu;   // [NTCB] main.cpp NTCB_SYNC_*
static const int NTCB_SYNC_BITS = 6;   // [NTCB]
static inline uint32_t ntcb_sync_value(uint32_t base, int level, int channel) { return (base + 13u * (uint32_t)level + 7u * (uint32_t)channel) & 63u; }   // [NTCB] main.cpp ntcb_sync_value: markers differ per level / channel; the MLP uses level = nlevels
struct NtcbByteReader {   // [NTCB] bounds-checked little-endian field reader
    const uint8_t* p; size_t n, pos = 0; bool ok = true;
    NtcbByteReader(const uint8_t* p_, size_t n_) : p(p_), n(n_) {}
    uint32_t u8() { if (pos + 1 > n) { ok = false; return 0; } return p[pos++]; }
    uint32_t u16() { const uint32_t a = u8(), b = u8(); return a | (b << 8); }
    uint32_t u32() { const uint32_t a = u16(), b = u16(); return a | (b << 16); }
    float f32() { const uint32_t u = u32(); float f; memcpy(&f, &u, 4); return f; }
};
struct NtcbBitReader {   // [NTCB] the writer's mirror (main.cpp NtcbBitWriter): `ok` drops on the first read past `end`
    const uint8_t* p; size_t pos, end; uint64_t acc = 0; int nacc = 0; bool ok = true;
    NtcbBitReader(const uint8_t* p_, size_t begin, size_t end_) : p(p_), pos(begin), end(end_) {}
    uint32_t get(int n) {
        while (nacc < n) { if (pos >= end) { ok = false; return 0; } acc |= (uint64_t)p[pos++] << nacc; nacc += 8; }
        const uint32_t v = (uint32_t)(acc & ((1ull << n) - 1ull)); acc >>= n; nacc -= n; return v;
    }
    bool marker(uint32_t m) { const uint32_t v = get(NTCB_SYNC_BITS); sync += NTCB_SYNC_BITS; return ok && v == m; }   // false: drifted or truncated
    double sync = 0;   // marker bits read so far
    void align() { acc = 0; nacc = 0; }   // drop the rest of the current byte; pos is then the next byte
};
static uint32_t ntcb_file_hash(const uint8_t* p, size_t n) {   // [NTCB] FNV-1a 32 over the whole file with the hash field (28..31) read as zero (main.cpp ntcb_file_hash)
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < n; i++) { h ^= (i >= 28 && i < 32) ? 0u : p[i]; h *= 16777619u; }
    return h;
}
static inline bool ntcb_finite32(float f) { uint32_t u; memcpy(&u, &f, 4); return (u & 0x7F800000u) != 0x7F800000u; }   // [NTCB] by bit pattern: std::isfinite folds under -ffast-math
static inline bool ntcb_finite16(uint16_t h) { return (h & 0x7C00u) != 0x7C00u; }   // [NTCB]
static float ntcb_f16_to_f32(uint16_t h) {   // [NTCB] IEEE binary16 -> binary32 (main.cpp ntcb_f16_to_f32)
    const uint32_t sign = ((uint32_t)h & 0x8000u) << 16; const int e = (h >> 10) & 0x1F; uint32_t man = h & 0x3FFu, u;
    if (e == 0) {
        if (man == 0) u = sign;
        else { int ee = -1; do { ee++; man <<= 1; } while ((man & 0x400u) == 0); u = sign | ((uint32_t)(127 - 15 - ee) << 23) | ((man & 0x3FFu) << 13); }
    } else if (e == 31) u = sign | 0x7F800000u | (man << 13);
    else u = sign | ((uint32_t)(e - 15 + 127) << 23) | (man << 13);
    float f; memcpy(&f, &u, 4); return f;
}
struct NtcbLevel {   // [NTCB] one level record (main.cpp NtcbLevel)
    int W = 0, H = 0, C = 0, filter = 0, mode = 0, dc_step = 0;
    int deadzone = 1;   // [DCT] mode 3: bit 0 of the version-2 dct_flags byte (1 in a version-1 file)
    std::vector<int> bits, q; std::vector<float> lo, hi; std::vector<std::vector<float>> palette;
};
struct NtcbHeader {   // [NTCB] main.cpp NtcbHeader
    int version = 0, H = 0, img_w = 0, img_h = 0, src_w = 0, src_h = 0; uint32_t file_bytes = 0, hash = 0;
    int latent_cell = 0, dct_block = 0, T = 1, nlevels = 1, act = 0, clamp = 0, nhidden = 1;
    float leak = 0.01f; int nin = 0, nout = 0; std::vector<int> cpt, hidden; std::string pos; std::vector<NtcbLevel> lv;
};
// [NTCB] main.cpp ntcb_min_section_bytes: the fewest bytes the header's sections can occupy (64-bit; a DCT body's floor is DC + EOB per block).
static uint64_t ntcb_min_section_bytes(const NtcbHeader& h) {
    uint64_t total = 0;
    for (const NtcbLevel& L : h.lv) {
        uint64_t bits = 2 * NTCB_SYNC_BITS;
        if (L.mode == 3) {
            const uint64_t nblk = ((uint64_t)L.W / 8) * ((uint64_t)L.H / 8);
            int dc_raw_bits = 0; { const int lvl = 512 / L.dc_step + 1; while ((1 << dc_raw_bits) < lvl) dc_raw_bits++; }
            bits += (uint64_t)L.C * (NTCB_SYNC_BITS + nblk * (uint64_t)(4 + dc_raw_bits + 7));
        } else for (int c = 0; c < L.C; c++) bits += (uint64_t)L.W * (uint64_t)L.H * (uint64_t)L.bits[c];
        total += 8 + (bits + 7) / 8;
    }
    uint64_t nparams = 0; { int prev = h.nin; for (int w : h.hidden) { nparams += (uint64_t)w * prev + w; prev = w; } nparams += (uint64_t)h.nout * prev + h.nout; }
    total += 8 + (16 * nparams + 2 * NTCB_SYNC_BITS + 7) / 8;
    return total;
}
// [NTCB] main.cpp ntcb_read_header, same checks in the same order (ranges, header length, file length, file hash, records, nin, minimum
// [NTCB] section size); nothing is allocated from a header field before the last check passes.
static bool ntcb_read_header(const std::vector<uint8_t>& buf, NtcbHeader& h, std::string& err) {
    const size_t n = buf.size();
    if (n < 8) { err = "not an ntcb file (shorter than 8 bytes)"; return false; }
    NtcbByteReader r(buf.data(), n);
    if (r.u32() != NTCB_MAGIC) { err = "not an ntcb file (bad magic)"; return false; }
    h.version = (int)r.u16(); h.H = (int)r.u16();
    if (h.version != 1 && h.version != 2) { err = "ntcb version " + std::to_string(h.version) + " (this build reads versions 1 and 2)"; return false; }   // [DCT] hook: version 2 (the dct_flags byte) accepted
    if (h.H < 52 || (size_t)h.H > n) { err = "header length " + std::to_string(h.H) + " exceeds the file size " + std::to_string(n) + " (truncated or corrupt)"; return false; }
    r.n = (size_t)h.H;
    h.img_w = (int)r.u32(); h.img_h = (int)r.u32(); h.src_w = (int)r.u32(); h.src_h = (int)r.u32();
    h.file_bytes = r.u32(); h.hash = r.u32();
    if (h.file_bytes != n) { err = "file_bytes " + std::to_string(h.file_bytes) + " != actual size " + std::to_string(n) + " (truncated or extended)"; return false; }
    if (ntcb_file_hash(buf.data(), n) != h.hash) { err = "file hash mismatch (corrupt file)"; return false; }
    h.latent_cell = (int)r.u8(); h.dct_block = (int)r.u8(); h.T = (int)r.u8(); h.nlevels = (int)r.u8(); h.act = (int)r.u8(); h.clamp = (int)r.u8(); h.nhidden = (int)r.u8();
    const int reserved = (int)r.u8();
    h.leak = r.f32(); h.nin = (int)r.u32(); h.nout = (int)r.u32();
    if (!r.ok) { err = "truncated header"; return false; }
    if (h.img_w < 1 || h.img_h < 1 || h.src_w < 1 || h.src_h < 1 || h.src_w > h.img_w || h.src_h > h.img_h || h.img_w > 16384 || h.img_h > 16384) { err = "bad image size fields (decode size at most 16384 x 16384, source size within it)"; return false; }
    if (h.T < 1 || h.T > MAXT || h.nlevels < 1 || h.nlevels > MAXLV || h.act != 0 || (h.clamp != 0 && h.clamp != 1) || h.nhidden < 1 || h.nhidden > MAXL || reserved != 0)
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
            if (L.mode <= 1 && B != L.bits[0]) { err = lv + "per-channel bit depths on a grid level are not supported"; return false; }
        }
        if (L.mode == 3) {
            L.dc_step = (int)r.u8();   // [DCT] hook: was `L.dc_step = (int)r.u8(); L.q.resize(L.C); for (...) L.q[c] = (int)r.u8();` on one line
            const int flags = h.version >= 2 ? (int)r.u8() : 1;   // [DCT] version 2: dct_flags after dc_step (bit 0 = dead zone); version 1: dead zone implied (main.cpp ntcb_read_header)
            L.q.resize(L.C); for (int c = 0; c < L.C; c++) L.q[c] = (int)r.u8();   // [DCT] hook: was on the dc_step line
            if (!r.ok) { err = lv + "truncated level record"; return false; }
            if (L.dc_step < 1 || L.dc_step > 64) { err = lv + "dc_step " + std::to_string(L.dc_step) + " out of range (1..64)"; return false; }
            if (flags & ~1) { err = lv + "dct_flags " + std::to_string(flags) + " has reserved bits set (only bit 0, the dead zone, is defined)"; return false; }   // [DCT]
            L.deadzone = flags & 1;   // [DCT]
            for (int c = 0; c < L.C; c++) if (L.q[c] < 1 || L.q[c] > 100) { err = lv + "dct quality " + std::to_string(L.q[c]) + " out of range (1..100)"; return false; }
            if (h.dct_block != 8 || L.filter != 1 || L.W != h.img_w || L.H != h.img_h || L.W % 8 != 0 || L.H % 8 != 0 || L.C > 4)
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
    { std::vector<PosFeature> feats; if (!parse_pos(h.pos, feats)) { err = "positional spec \"" + h.pos + "\" is not one this build reads"; return false; }
      int pc = 0; for (const PosFeature& f : feats) pc += pos_count(f);
      if (chsum + pc != h.nin) { err = "nin " + std::to_string(h.nin) + " != latent channels " + std::to_string(chsum) + " + positional inputs " + std::to_string(pc); return false; } }
    { const uint64_t need = ntcb_min_section_bytes(h), have = (uint64_t)n - (uint64_t)h.H;
      if (need > have) { err = "the header's sections need at least " + std::to_string(need) + " bytes but the file has " + std::to_string(have) + " after the header (truncated or corrupt)"; return false; } }
    return true;
}
// [NTCB] Parse the container into the Model the v13 path decodes: grid levels dequantized here (strict FP), a dct level 0 as symbols and
// [NTCB] codes for dct_recon_level0, the MLP from fp16. Section lengths, the token stream, the hash, the file length and trailing bytes are
// [NTCB] all checked; `err` names the first problem. Mirrors main.cpp ntcb_restore.
static bool load_ntcb(const std::vector<uint8_t>& buf, Model& m, std::string& err) {
    NtcbHeader h;
    if (!ntcb_read_header(buf, h, err)) return false;
    m.version = 13; m.ntcb = true; m.ntcb_version = h.version; m.ntcb_bytes = buf.size(); m.ntcb_H = h.H;
    m.nin = h.nin; m.act = h.act; m.T = h.T; m.leak = h.leak; m.deblock = 0; m.deblock_falloff = 1.0f;
    m.hidden = h.hidden; m.pos = h.pos; m.imgW = h.img_w; m.imgH = h.img_h; m.srcW = h.src_w; m.srcH = h.src_h; m.clamp = h.clamp;
    m.lv.resize(h.nlevels); m.qat_bits = 0; m.qat_ch.clear(); m.qes_bits.assign(h.nlevels, 0); m.qes_lo.assign(h.nlevels, std::vector<float>()); m.qes_hi.assign(h.nlevels, std::vector<float>());
    m.ntcb_mode.assign(h.nlevels, 0); m.ntcb_bits.assign(h.nlevels, 0);
    size_t off = 0;   // (allocation from the header fields: ntcb_read_header has checked the minimum section size against the file)
    for (int l = 0; l < h.nlevels; l++) {
        const NtcbLevel& N = h.lv[l];
        m.lv[l].W = N.W; m.lv[l].H = N.H; m.lv[l].C = N.C; m.lv[l].nearest = N.filter == 1; m.lv[l].off = off; off += m.lv[l].size();
        m.ntcb_mode[l] = N.mode; m.ntcb_bits[l] = N.bits[0];
        if (N.mode == 2) { m.qat_ch = N.bits; for (int b : N.bits) m.qat_bits = std::max(m.qat_bits, b); }
        if (N.mode <= 1) { m.qes_bits[l] = N.bits[0]; m.qes_lo[l] = N.lo; m.qes_hi[l] = N.hi; }
        if (N.mode == 3) { m.dct_N = 8; m.dct_dc_step = N.dc_step; m.dct_C = N.C; m.dct_q = N.q; m.dct_deadzone = N.deadzone; }   // [DCT] hook: m.dct_deadzone
    }
    m.z.assign(off + 4, 0.0f);   // 4 floats of slack: the SIMD sampler reads whole 4-channel vectors
    m.p.resize(m.nparams());
    const size_t n = buf.size(); size_t pos = (size_t)h.H;
    double content = 0, pad = 0, sync = 0; int nsec = 0;
    auto marker = [&](NtcbBitReader& br, uint32_t mk, const std::string& where) -> bool {
        if (!br.marker(mk)) { err = "sync marker mismatch " + where; return false; }
        return true;
    };
    auto section = [&](int kind, int level, size_t& body, size_t& len) -> bool {
        const std::string what = kind == 1 ? "level " + std::to_string(level) : std::string("the mlp");
        if (pos + 8 > n) { err = "truncated at the section header of " + what; return false; }
        NtcbByteReader r(buf.data() + pos, 8);
        const int k = (int)r.u8(), coding = (int)r.u8(), lv = (int)r.u8(), res = (int)r.u8(); len = r.u32();
        if (coding != 0) { err = (coding == 1 || coding == 2) ? "coded sections (coding " + std::to_string(coding) + ") need a newer decoder" : "bad section coding " + std::to_string(coding); return false; }
        if (k != kind || lv != level || res != 0) { err = "unexpected section header (kind " + std::to_string(k) + ", level " + std::to_string(lv) + ", reserved " + std::to_string(res) + ") where " + what + " was expected"; return false; }
        body = pos + 8;
        if (len > n - body) { err = "section of " + what + " runs past the end of the file"; return false; }
        return true;
    };
    for (int l = 0; l < h.nlevels; l++) {
        const NtcbLevel& N = h.lv[l]; const Level& L = m.lv[l]; const std::string lv = "level " + std::to_string(l) + ": ", ls = std::to_string(l);
        size_t body = 0, len = 0;
        if (!section(1, l, body, len)) return false;
        NtcbBitReader br(buf.data(), body, body + len);
        float* z = m.z.data() + L.off;
        if (!marker(br, ntcb_sync_value(NTCB_SYNC_LEVEL, l, 0), "before level " + ls + " body")) return false;
        if (N.mode == 3) {
            dct_init_tables();
            const size_t nb = (size_t)(N.W / 8) * (N.H / 8); const int C = N.C;
            int dc_raw_bits = 0; { const int lvl = 512 / N.dc_step + 1; while ((1 << dc_raw_bits) < lvl) dc_raw_bits++; }
            m.dct_sym.assign(nb * C * 64, 0); m.dct_code.assign(nb * C, 0);
            for (int c = 0; c < C; c++) {
                if (!marker(br, ntcb_sync_value(NTCB_SYNC_CHANNEL, l, c), "before level " + ls + " channel " + std::to_string(c))) return false;
                for (size_t b = 0; b < nb; b++) m.dct_code[b * C + c] = (uint8_t)br.get(4);
                for (size_t b = 0; b < nb && br.ok; b++) {
                    int16_t* sym = &m.dct_sym[(b * C + c) * 64];
                    sym[0] = (int16_t)br.get(dc_raw_bits);
                    int p = 1;
                    while (p < 64 && br.ok) {
                        const int run = (int)br.get(7);
                        if (!br.ok || run == 64) break;
                        if (run > 62) { err = lv + "bad run symbol " + std::to_string(run) + " at block " + std::to_string(b) + " (0..62 or the EOB 64)"; return false; }
                        p += run;
                        if (p >= 64) { err = lv + "token stream runs past the 63 AC positions at block " + std::to_string(b); return false; }
                        const int mag = (int)br.get(8) + 1, sign = (int)br.get(1);
                        sym[DCT_ZIGZAG[p]] = (int16_t)(sign ? -mag : mag);
                        p++;
                    }
                }
            }
            if (!br.ok) { err = lv + "truncated token stream"; return false; }
        } else {
            for (int c = 0; c < L.C; c++) {
                const int bits = N.bits[c], levels = (1 << bits) - 1;
                const float lo = N.mode <= 1 ? N.lo[c] : 0.0f, range = N.mode <= 1 ? std::max(N.hi[c] - N.lo[c], 1e-6f) : 0.0f;   // main.cpp qes_rebuild L415
                for (size_t i = c; i < L.size() && br.ok; i += L.C) {
                    const int k = (int)br.get(bits);
                    z[i] = N.mode == 2 ? N.palette[c][k] : lo + k / (float)levels * range;   // main.cpp qes_rebuild L416, strict FP
                }
            }
            if (!br.ok) { err = lv + "truncated grid indices"; return false; }
        }
        if (!marker(br, ntcb_sync_value(NTCB_SYNC_END, l, 0), "after level " + ls + " body")) return false;
        content += (double)((br.pos - body) * 8 - (size_t)br.nacc) - br.sync; sync += br.sync; pad += (double)br.nacc;
        br.align();
        if (br.pos != body + len) { err = lv + "section length " + std::to_string(len) + " != the " + std::to_string(br.pos - body) + " bytes its content occupies"; return false; }
        pos = body + len; nsec++;
    }
    {
        size_t body = 0, len = 0;
        if (!section(2, 0, body, len)) return false;
        if (len != (m.p.size() * 16 + 2 * NTCB_SYNC_BITS + 7) / 8) { err = "mlp section length " + std::to_string(len) + " != the bytes of " + std::to_string(m.p.size()) + " fp16 weights and two sync markers"; return false; }
        NtcbBitReader br(buf.data(), body, body + len);
        if (!marker(br, ntcb_sync_value(NTCB_SYNC_MLP, h.nlevels, 0), "before the mlp body")) return false;
        for (size_t i = 0; i < m.p.size(); i++) { const uint16_t hh = (uint16_t)br.get(16); m.p[i] = ntcb_f16_to_f32(hh); if (br.ok && !ntcb_finite16(hh)) { err = "non-finite mlp weight " + std::to_string(i); return false; } }
        if (!br.ok) { err = "truncated mlp section"; return false; }
        if (!marker(br, ntcb_sync_value(NTCB_SYNC_END, h.nlevels, 0), "after the mlp body")) return false;
        content += 16.0 * (double)m.p.size(); sync += br.sync; pad += (double)br.nacc;
        br.align();
        if (br.pos != body + len) { err = "mlp section length " + std::to_string(len) + " != the " + std::to_string(br.pos - body) + " bytes its content occupies"; return false; }
        pos = body + len; nsec++;
    }
    if (pos != n) { err = "trailing bytes after the mlp section (" + std::to_string(n - pos) + ")"; return false; }
    m.ntcb_content_bits = content; m.ntcb_pad_bits = pad; m.ntcb_sync_bits = sync; m.ntcb_nsections = nsec;
    return true;
}
// ---- end of the NTCB region [NTCB]
// Per-axis tap tables: the x (or y) half of ref_tap for every pixel column (row).
struct Axis { std::vector<int> i0, i1; std::vector<float> f, lv; };   // lv = f*2-1; all padded by 8 entries past n
static void build_axis(int n, int L, bool nearest, Axis& a) {
    a.i0.resize(n + 8); a.i1.resize(n + 8); a.f.assign(n + 8, 0.0f); a.lv.assign(n + 8, 0.0f);
    for (int p = 0; p < n; p++) {
        float u = (p + 0.5f) / n;
        if (nearest) {
            float x = u * L; int ix = (int)std::floor(x);
            a.f[p] = x - ix; a.i0[p] = a.i1[p] = std::max(0, std::min(L - 1, ix));
        } else {
            float x = u * L - 0.5f; int x0 = (int)std::floor(x);
            a.f[p] = x - x0; a.i0[p] = std::max(0, std::min(L - 1, x0)); a.i1[p] = std::max(0, std::min(L - 1, x0 + 1));
        }
        a.lv[p] = a.f[p] * 2.0f - 1.0f;
    }
    for (int p = n; p < n + 8; p++) { a.i0[p] = a.i0[n - 1]; a.i1[p] = a.i1[n - 1]; }   // lanes past the edge read valid texels
}

// Separable positional features for the SIMD path: each MLP input is a per-column value or a
// per-row value. Tables are filled with the same expressions as ref_pos_encode.
enum { PE_COL, PE_ROW };
struct PosEntry { int type; std::vector<float> col, row; };
static void build_pos_entries(const std::vector<PosFeature>& feats, int W, int H, const Axis* ax1, const Axis* ay1, std::vector<PosEntry>& out) {
    out.clear();
    auto colv = [&]() { PosEntry e; e.type = PE_COL; e.col.assign(W + 8, 0.0f); return e; };
    auto rowv = [&]() { PosEntry e; e.type = PE_ROW; e.row.assign(H, 0.0f); return e; };
    for (const PosFeature& pf : feats) {
        switch (pf.kind) {
        case POS_UV: {
            PosEntry c = colv(), r = rowv();
            for (int px = 0; px < W; px++) { float u = (px + 0.5f) / W; c.col[px] = u * 2.0f - 1.0f; }
            for (int py = 0; py < H; py++) { float v = (py + 0.5f) / H; r.row[py] = v * 2.0f - 1.0f; }
            out.push_back(c); out.push_back(r); break;
        }
        case POS_LV1LOCAL: {
            PosEntry c = colv(), r = rowv();
            for (int px = 0; px < W; px++) c.col[px] = ax1->f[px] * 2.0f - 1.0f;
            for (int py = 0; py < H; py++) r.row[py] = ay1->f[py] * 2.0f - 1.0f;
            out.push_back(c); out.push_back(r); break;
        }
        }
    }
}

STRICT_FP_END

// ---------------------------------------------------------------- SIMD kernel
// Layout: lanes = texels. Eight consecutive texels of one image row form two __m128 groups; every
// weight and bias is pre-broadcast to a __m128 so the inner MAC is mulps + addps against a memory
// operand, in the trainer's sequential order (s = b; s += w[i]*x[i] for i = 0..ncur-1).
// Compiled strictly as well: MSVC treats intrinsics as opaque, but GCC/Clang lower them to generic
// vector IR and would reassociate the sums under -ffast-math (measured: 1.48M of 2.52M pre-sigmoid
// values off by up to 2.4e-6 without this).
STRICT_FP_BEGIN
#if defined(_MSC_VER) && !defined(__clang__)
#define NTC_INLINE __forceinline
#else
#define NTC_INLINE inline __attribute__((always_inline))
#endif

struct PackedMLP {
    int nin = 0, nout = 0, nlayers = 0;
    std::vector<int> widths;
    std::vector<size_t> loff;    // __m128 offset of each layer's weights: output j at loff[l] + j*(ncur+1): [bias][w_0..w_ncur-1]
    std::vector<__m128> w;
    int actmode = 0;             // 1: leaky via max(x, leak*x) (0 <= leak < 1), 3: leaky via blend (any other leak)
    __m128 leak4;
    bool clamp = false;
};

static void pack_mlp(const Model& m, PackedMLP& P) {
    P.widths = m.widths(); P.nin = m.nin; P.nout = 3 * m.T; P.nlayers = (int)P.widths.size() - 1;
    size_t total = 0;
    for (int l = 1; l <= P.nlayers; l++) { P.loff.push_back(total); total += (size_t)P.widths[l] * (P.widths[l - 1] + 1); }
    P.w.resize(total);
    const float* p = m.p.data();
    for (int l = 1; l <= P.nlayers; l++) {
        int ncur = P.widths[l - 1], nh = P.widths[l];
        const float* W = p; const float* b = W + (size_t)nh * ncur;
        for (int j = 0; j < nh; j++) {
            __m128* d = &P.w[P.loff[l - 1] + (size_t)j * (ncur + 1)];
            d[0] = _mm_set1_ps(b[j]);
            for (int i = 0; i < ncur; i++) d[1 + i] = _mm_set1_ps(W[(size_t)j * ncur + i]);
        }
        p = b + nh;
    }
    P.leak4 = _mm_set1_ps(m.leak);
    P.actmode = (m.leak >= 0.0f && m.leak < 1.0f) ? 1 : 3;
}

struct Scratch {
    __m128 in[MAXH][2], bufA[MAXH][2], bufB[MAXH][2], out[MAXOUT][2];
    __m128 rowv[MAXH];              // per-row positional values
    alignas(16) uint8_t rgb[32];    // RGB8 of one 8-texel group (row tail)
};

// NJ outputs x 8 texels: NJ*2 accumulators, 2 inputs, 1 broadcast weight in registers.
template<int NJ, int NC, int ACT>
static NTC_INLINE void mlp_chunk(const __m128* cur, int ncur_rt, const __m128* w, __m128* out, __m128 leak4) {
    const int ncur = NC ? NC : ncur_rt;
    const int stride = ncur + 1;
    __m128 a0[NJ], a1[NJ];
    for (int j = 0; j < NJ; j++) { a0[j] = w[j * stride]; a1[j] = a0[j]; }
    for (int i = 0; i < ncur; i++) {
        const __m128 c0 = cur[2 * i], c1 = cur[2 * i + 1];
        for (int j = 0; j < NJ; j++) {
            const __m128 ww = w[j * stride + 1 + i];
            a0[j] = _mm_add_ps(a0[j], _mm_mul_ps(ww, c0));
            a1[j] = _mm_add_ps(a1[j], _mm_mul_ps(ww, c1));
        }
    }
    for (int j = 0; j < NJ; j++) {
        __m128 x0 = a0[j], x1 = a1[j];
        if (ACT == 1) { x0 = _mm_max_ps(x0, _mm_mul_ps(leak4, x0)); x1 = _mm_max_ps(x1, _mm_mul_ps(leak4, x1)); }
        else if (ACT == 3) { x0 = _mm_blendv_ps(_mm_mul_ps(leak4, x0), x0, _mm_cmpgt_ps(x0, _mm_setzero_ps())); x1 = _mm_blendv_ps(_mm_mul_ps(leak4, x1), x1, _mm_cmpgt_ps(x1, _mm_setzero_ps())); }
        out[2 * j] = x0; out[2 * j + 1] = x1;
    }
}
template<int NC, int ACT>
static NTC_INLINE void mlp_layer(const __m128* cur, int ncur, int nh, const __m128* w, __m128* out, __m128 leak4) {
    const int stride = (NC ? NC : ncur) + 1;
    int j = 0;
    if (nh % 4 == 1 && nh >= 5) { mlp_chunk<5, NC, ACT>(cur, ncur, w, out, leak4); j = 5; }   // 17 = 5+4+4+4: a 1-output chunk would be latency-bound
    for (; j + 4 <= nh; j += 4) mlp_chunk<4, NC, ACT>(cur, ncur, w + (size_t)j * stride, out + 2 * j, leak4);
    switch (nh - j) {
    case 3: mlp_chunk<3, NC, ACT>(cur, ncur, w + (size_t)j * stride, out + 2 * j, leak4); break;
    case 2: mlp_chunk<2, NC, ACT>(cur, ncur, w + (size_t)j * stride, out + 2 * j, leak4); break;
    case 1: mlp_chunk<1, NC, ACT>(cur, ncur, w + (size_t)j * stride, out + 2 * j, leak4); break;
    default: break;
    }
}
template<int NC>
static NTC_INLINE void mlp_layer_act(int actmode, const __m128* cur, int ncur, int nh, const __m128* w, __m128* out, __m128 leak4) {
    switch (actmode) {
    case 1: mlp_layer<NC, 1>(cur, ncur, nh, w, out, leak4); break;
    default: mlp_layer<NC, 3>(cur, ncur, nh, w, out, leak4); break;
    }
}
typedef void (*EvalFn)(const PackedMLP&, Scratch&);
// Two hidden layers with compile-time widths (the inner i loops get constant trip counts).
template<int N0, int N1, int N2>
static void eval_2h(const PackedMLP& P, Scratch& S) {
    mlp_layer_act<N0>(P.actmode, &S.in[0][0], P.widths[0], P.widths[1], P.w.data() + P.loff[0], &S.bufA[0][0], P.leak4);
    mlp_layer_act<N1>(P.actmode, &S.bufA[0][0], P.widths[1], P.widths[2], P.w.data() + P.loff[1], &S.bufB[0][0], P.leak4);
    mlp_layer<N2, 0>(&S.bufB[0][0], P.widths[2], P.widths[3], P.w.data() + P.loff[2], &S.out[0][0], P.leak4);
}
static void eval_generic(const PackedMLP& P, Scratch& S) {
    const __m128* cur = &S.in[0][0]; __m128* nxt = &S.bufA[0][0];
    for (int l = 1; l < P.nlayers; l++) {
        mlp_layer_act<0>(P.actmode, cur, P.widths[l - 1], P.widths[l], P.w.data() + P.loff[l - 1], nxt, P.leak4);
        cur = nxt; nxt = (nxt == &S.bufA[0][0]) ? &S.bufB[0][0] : &S.bufA[0][0];
    }
    mlp_layer<0, 0>(cur, P.widths[P.nlayers - 1], P.widths[P.nlayers], P.w.data() + P.loff[P.nlayers - 1], &S.out[0][0], P.leak4);
}
static EvalFn pick_eval(const PackedMLP& P, const char** name) {
    if (P.nlayers == 3) {
        int a = P.widths[0], b = P.widths[1], c = P.widths[2];
        if (a == 7 && b == 17 && c == 17) { *name = "7,17,17 specialization"; return eval_2h<7, 17, 17>; }
        if (a == 7 && b == 27 && c == 27) { *name = "7,27,27 specialization"; return eval_2h<7, 27, 27>; }
        if (a == 8 && b == 36 && c == 36) { *name = "8,36,36 specialization"; return eval_2h<8, 36, 36>; }
        *name = "2-hidden-layer generic"; return eval_2h<0, 0, 0>;
    }
    *name = "generic"; return eval_generic;
}

// exp(x), Cephes-style: ~1 ulp. Input clamped to [-87, 88] (exp(88) < FLT_MAX; 2^round(x*log2e) stays finite).
static inline __m128 exp_exact(__m128 x) {
    x = _mm_min_ps(x, _mm_set1_ps(88.0f)); x = _mm_max_ps(x, _mm_set1_ps(-87.0f));
    __m128 fx = _mm_round_ps(_mm_mul_ps(x, _mm_set1_ps(1.44269504088896341f)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    x = _mm_sub_ps(x, _mm_mul_ps(fx, _mm_set1_ps(0.693359375f)));
    x = _mm_sub_ps(x, _mm_mul_ps(fx, _mm_set1_ps(-2.12194440e-4f)));
    __m128 z = _mm_mul_ps(x, x);
    __m128 y = _mm_set1_ps(1.9875691500E-4f);
    y = _mm_add_ps(_mm_mul_ps(y, x), _mm_set1_ps(1.3981999507E-3f));
    y = _mm_add_ps(_mm_mul_ps(y, x), _mm_set1_ps(8.3334519073E-3f));
    y = _mm_add_ps(_mm_mul_ps(y, x), _mm_set1_ps(4.1665795894E-2f));
    y = _mm_add_ps(_mm_mul_ps(y, x), _mm_set1_ps(1.6666665459E-1f));
    y = _mm_add_ps(_mm_mul_ps(y, x), _mm_set1_ps(5.0000001201E-1f));
    y = _mm_add_ps(_mm_add_ps(_mm_mul_ps(y, z), x), _mm_set1_ps(1.0f));
    __m128i e = _mm_slli_epi32(_mm_add_epi32(_mm_cvttps_epi32(fx), _mm_set1_epi32(127)), 23);
    return _mm_mul_ps(y, _mm_castsi128_ps(e));
}
// exp(x), fast: single-constant range reduction and a degree-4 relative-minimax polynomial on
// [-ln2/2, ln2/2] (max rel. error 2.7e-6 in float Horner form).
static inline __m128 exp_fast(__m128 x) {
    x = _mm_min_ps(x, _mm_set1_ps(88.0f)); x = _mm_max_ps(x, _mm_set1_ps(-87.0f));
    __m128 fx = _mm_round_ps(_mm_mul_ps(x, _mm_set1_ps(1.44269504088896341f)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m128 r = _mm_sub_ps(x, _mm_mul_ps(fx, _mm_set1_ps(0.6931471805599453f)));
    __m128 y = _mm_set1_ps(0.04145858809351921f);
    y = _mm_add_ps(_mm_mul_ps(y, r), _mm_set1_ps(0.16790908575057983f));
    y = _mm_add_ps(_mm_mul_ps(y, r), _mm_set1_ps(0.5000435709953308f));
    y = _mm_add_ps(_mm_mul_ps(y, r), _mm_set1_ps(0.9999634027481079f));
    y = _mm_add_ps(_mm_mul_ps(y, r), _mm_set1_ps(0.9999992847442627f));
    __m128i e = _mm_slli_epi32(_mm_add_epi32(_mm_cvttps_epi32(fx), _mm_set1_epi32(127)), 23);
    return _mm_mul_ps(y, _mm_castsi128_ps(e));
}
static inline __m128 sigmoid_exact(__m128 s) { return _mm_div_ps(_mm_set1_ps(1.0f), _mm_add_ps(_mm_set1_ps(1.0f), exp_exact(_mm_sub_ps(_mm_setzero_ps(), s)))); }
static inline __m128 sigmoid_fast(__m128 s) {
    __m128 d = _mm_add_ps(_mm_set1_ps(1.0f), exp_fast(_mm_sub_ps(_mm_setzero_ps(), s)));
    __m128 y = _mm_rcp_ps(d);
    return _mm_mul_ps(y, _mm_sub_ps(_mm_set1_ps(2.0f), _mm_mul_ps(d, y)));   // one Newton step
}
static inline __m128 clamp_out(__m128 s) { return _mm_min_ps(_mm_set1_ps(1.0f), _mm_max_ps(_mm_setzero_ps(), _mm_add_ps(s, _mm_set1_ps(0.5f)))); }
// lround(min(1, max(0, o)) * 255): round-to-nearest-even, then ties (exactly .5) rounded up as lround does for x >= 0.
static inline __m128i to_q8(__m128 o) {
    o = _mm_min_ps(_mm_set1_ps(1.0f), _mm_max_ps(_mm_setzero_ps(), o));
    __m128 x = _mm_mul_ps(o, _mm_set1_ps(255.0f));
    __m128i q = _mm_cvtps_epi32(x);
    __m128 tie = _mm_cmpeq_ps(_mm_sub_ps(x, _mm_cvtepi32_ps(q)), _mm_set1_ps(0.5f));
    return _mm_sub_epi32(q, _mm_castps_si128(tie));
}

struct LevelCtx {
    const Level* L; const float* z;   // z: this level's first value
    bool planar = false;              // level 0 full-res nearest: read the planar copy directly
    const float* plane[MAXH];         // planar fp32 (plane[c][y*W + x], +8 slack)
    const uint8_t* idx[MAXH];         // --pack-selectors: B-bit indices instead of plane[]
    __m128i lut[MAXH][4];             // per channel: byte b of the 16 grid values, for a pshufb lookup
    const Axis* ax; const Axis* ay;
};
struct DecodeCtx {
    int W = 0, H = 0, T = 1, nout = 3, nlv = 0;
    LevelCtx lv[MAXLV];
    std::vector<PosEntry> pos;
    const PackedMLP* P = nullptr;
    EvalFn eval = nullptr;
    bool simd = true, fast = true, clamp = false;
    uint8_t* img[MAXT];               // W*H*3 each
    float* pre = nullptr;             // --verify: pre-nonlinearity outputs, (y*W+x)*nout + c
    float* post = nullptr;            // --verify: fp32 outputs
    int strip_rows = 8;
    // scalar path
    const Model* m = nullptr; const std::vector<PosFeature>* feats = nullptr;
};

// Sample a bilinear level for 4 texels x0..x0+3 of row (ry0, ry1) into in[C][h]: the four corners of
// every texel are read as 4-channel vectors ((y, x, c) storage, c fastest), the bilinear sum is
// evaluated per texel in that layout with the texel's weights broadcast, exactly as sample_latent
// does per channel (w00*a + w10*b + w01*c + w11*d, left to right), and each 4-channel block is then
// transposed into the lane layout. Channel counts that are not multiples of 4 read (but do not use)
// up to 3 values past the texel; the latent vector carries 4 floats of slack for that.
static NTC_INLINE void sample_bilinear4(const LevelCtx& L, const float* r0, const float* r1, int x0, __m128 w00v, __m128 w10v, __m128 w01v, __m128 w11v, __m128 (*in)[2], int h) {
    const int C = L.L->C;
    __m128 s[MAXH / 4][4];
    for (int j = 0; j < 4; j++) {
        const int x = x0 + j, cx0 = L.ax->i0[x], cx1 = L.ax->i1[x];
        const float* a = r0 + (size_t)cx0 * C; const float* b = r0 + (size_t)cx1 * C;
        const float* cc = r1 + (size_t)cx0 * C; const float* d = r1 + (size_t)cx1 * C;
        __m128 w00, w10, w01, w11;
        switch (j) {
        case 0: w00 = _mm_shuffle_ps(w00v, w00v, 0x00); w10 = _mm_shuffle_ps(w10v, w10v, 0x00); w01 = _mm_shuffle_ps(w01v, w01v, 0x00); w11 = _mm_shuffle_ps(w11v, w11v, 0x00); break;
        case 1: w00 = _mm_shuffle_ps(w00v, w00v, 0x55); w10 = _mm_shuffle_ps(w10v, w10v, 0x55); w01 = _mm_shuffle_ps(w01v, w01v, 0x55); w11 = _mm_shuffle_ps(w11v, w11v, 0x55); break;
        case 2: w00 = _mm_shuffle_ps(w00v, w00v, 0xAA); w10 = _mm_shuffle_ps(w10v, w10v, 0xAA); w01 = _mm_shuffle_ps(w01v, w01v, 0xAA); w11 = _mm_shuffle_ps(w11v, w11v, 0xAA); break;
        default: w00 = _mm_shuffle_ps(w00v, w00v, 0xFF); w10 = _mm_shuffle_ps(w10v, w10v, 0xFF); w01 = _mm_shuffle_ps(w01v, w01v, 0xFF); w11 = _mm_shuffle_ps(w11v, w11v, 0xFF); break;
        }
        for (int kc = 0; kc < C; kc += 4)
            s[kc / 4][j] = _mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_mul_ps(w00, _mm_loadu_ps(a + kc)), _mm_mul_ps(w10, _mm_loadu_ps(b + kc))), _mm_mul_ps(w01, _mm_loadu_ps(cc + kc))), _mm_mul_ps(w11, _mm_loadu_ps(d + kc)));
    }
    for (int kc = 0; kc < C; kc += 4) {
        __m128 t0 = s[kc / 4][0], t1 = s[kc / 4][1], t2 = s[kc / 4][2], t3 = s[kc / 4][3];
        _MM_TRANSPOSE4_PS(t0, t1, t2, t3);
        in[kc][h] = t0;
        if (kc + 1 < C) in[kc + 1][h] = t1;
        if (kc + 2 < C) in[kc + 2][h] = t2;
        if (kc + 3 < C) in[kc + 3][h] = t3;
    }
}
// Same for a nearest level (one texel per lane, cells wider than a pixel).
static NTC_INLINE void sample_nearest4(const LevelCtx& L, const float* r0, int x0, __m128 (*in)[2], int h) {
    const int C = L.L->C;
    __m128 s[MAXH / 4][4];
    for (int j = 0; j < 4; j++) {
        const float* a = r0 + (size_t)L.ax->i0[x0 + j] * C;
        for (int kc = 0; kc < C; kc += 4) s[kc / 4][j] = _mm_loadu_ps(a + kc);
    }
    for (int kc = 0; kc < C; kc += 4) {
        __m128 t0 = s[kc / 4][0], t1 = s[kc / 4][1], t2 = s[kc / 4][2], t3 = s[kc / 4][3];
        _MM_TRANSPOSE4_PS(t0, t1, t2, t3);
        in[kc][h] = t0;
        if (kc + 1 < C) in[kc + 1][h] = t1;
        if (kc + 2 < C) in[kc + 2][h] = t2;
        if (kc + 3 < C) in[kc + 3][h] = t3;
    }
}

static void decode_rows_simd(const DecodeCtx& c, Scratch& S, int ys, int ye) {
    const int W = c.W;
    const __m128 one = _mm_set1_ps(1.0f);
    const __m128i m1 = _mm_setr_epi8(0, 8, -128, 1, 9, -128, 2, 10, -128, 3, 11, -128, 4, 12, -128, 5);
    const __m128i m2 = _mm_setr_epi8(-128, -128, 0, -128, -128, 1, -128, -128, 2, -128, -128, 3, -128, -128, 4, -128);
    const __m128i m3 = _mm_setr_epi8(13, -128, 6, 14, -128, 7, 15, -128, -128, -128, -128, -128, -128, -128, -128, -128);
    const __m128i m4 = _mm_setr_epi8(-128, 5, -128, -128, 6, -128, -128, 7, -128, -128, -128, -128, -128, -128, -128, -128);
    for (int py = ys; py < ye; py++) {
        // per-row constants
        const float* r0[MAXLV]; const float* r1[MAXLV]; __m128 fy4[MAXLV], omfy4[MAXLV];
        for (int l = 0; l < c.nlv; l++) {
            const LevelCtx& L = c.lv[l];
            const size_t rowsz = (size_t)L.L->W * L.L->C;
            r0[l] = L.z + L.ay->i0[py] * rowsz; r1[l] = L.z + L.ay->i1[py] * rowsz;
            const float fy = L.ay->f[py];
            fy4[l] = _mm_set1_ps(fy); omfy4[l] = _mm_set1_ps(1.0f - fy);
        }
        for (size_t e = 0; e < c.pos.size(); e++) if (c.pos[e].type != PE_COL) S.rowv[e] = _mm_set1_ps(c.pos[e].row[py]);
        for (int px = 0; px < W; px += 8) {
            // (1) MLP inputs for texels px..px+7 (lanes past W-1 compute on clamped/padded data and are discarded)
            int ii = 0;
            for (int l = 0; l < c.nlv; l++) {
                const LevelCtx& L = c.lv[l];
                const int C = L.L->C;
                if (L.planar) {
                    const size_t o = (size_t)py * W + px;
                    if (L.idx[0]) {
                        // 8 B-bit indices -> 8 fp32 grid values through the 16-entry LUT: one pshufb per byte lane, then interleave
                        for (int k = 0; k < C; k++, ii++) {
                            const __m128i ix = _mm_loadl_epi64((const __m128i*)(L.idx[k] + o));
                            const __m128i b0 = _mm_shuffle_epi8(L.lut[k][0], ix), b1 = _mm_shuffle_epi8(L.lut[k][1], ix);
                            const __m128i b2 = _mm_shuffle_epi8(L.lut[k][2], ix), b3 = _mm_shuffle_epi8(L.lut[k][3], ix);
                            const __m128i lo16 = _mm_unpacklo_epi8(b0, b1), hi16 = _mm_unpacklo_epi8(b2, b3);
                            S.in[ii][0] = _mm_castsi128_ps(_mm_unpacklo_epi16(lo16, hi16));
                            S.in[ii][1] = _mm_castsi128_ps(_mm_unpackhi_epi16(lo16, hi16));
                        }
                    } else {
                        for (int k = 0; k < C; k++, ii++) { S.in[ii][0] = _mm_loadu_ps(L.plane[k] + o); S.in[ii][1] = _mm_loadu_ps(L.plane[k] + o + 4); }
                    }
                } else if (L.L->nearest) {
                    sample_nearest4(L, r0[l], px, S.in + ii, 0);
                    sample_nearest4(L, r0[l], px + 4, S.in + ii, 1);
                    ii += C;
                } else {
                    for (int h = 0; h < 2; h++) {
                        const __m128 fx = _mm_loadu_ps(L.ax->f.data() + px + 4 * h);
                        const __m128 omfx = _mm_sub_ps(one, fx);
                        sample_bilinear4(L, r0[l], r1[l], px + 4 * h, _mm_mul_ps(omfx, omfy4[l]), _mm_mul_ps(fx, omfy4[l]), _mm_mul_ps(omfx, fy4[l]), _mm_mul_ps(fx, fy4[l]), S.in + ii, h);
                    }
                    ii += C;
                }
            }
            for (size_t e = 0; e < c.pos.size(); e++, ii++) {
                const PosEntry& pe = c.pos[e];
                if (pe.type == PE_COL) { S.in[ii][0] = _mm_loadu_ps(pe.col.data() + px); S.in[ii][1] = _mm_loadu_ps(pe.col.data() + px + 4); }
                else { S.in[ii][0] = S.rowv[e]; S.in[ii][1] = S.rowv[e]; }
            }
            // (2) MLP
            c.eval(*c.P, S);
            if (c.pre) {
                for (int o = 0; o < c.nout; o++) for (int h = 0; h < 2; h++) {
                    alignas(16) float t[4]; _mm_store_ps(t, S.out[o][h]);
                    for (int q = 0; q < 4; q++) { int x = px + 4 * h + q; if (x < W) c.pre[((size_t)py * W + x) * c.nout + o] = t[q]; }
                }
            }
            // (3) output nonlinearity
            for (int o = 0; o < c.nout; o++) for (int h = 0; h < 2; h++) {
                __m128 s = S.out[o][h];
                S.out[o][h] = c.clamp ? clamp_out(s) : (c.fast ? sigmoid_fast(s) : sigmoid_exact(s));
            }
            if (c.post) {
                for (int o = 0; o < c.nout; o++) for (int h = 0; h < 2; h++) {
                    alignas(16) float t[4]; _mm_store_ps(t, S.out[o][h]);
                    for (int q = 0; q < 4; q++) { int x = px + 4 * h + q; if (x < W) c.post[((size_t)py * W + x) * c.nout + o] = t[q]; }
                }
            }
            // (4) RGB8 pack: 8 texels -> 24 bytes per texture, written straight into the image (via a temp for the row tail)
            const int nvalid = std::min(8, W - px);
            for (int t = 0; t < c.T; t++) {
                __m128i r16 = _mm_packs_epi32(to_q8(S.out[3 * t][0]), to_q8(S.out[3 * t][1]));
                __m128i g16 = _mm_packs_epi32(to_q8(S.out[3 * t + 1][0]), to_q8(S.out[3 * t + 1][1]));
                __m128i b16 = _mm_packs_epi32(to_q8(S.out[3 * t + 2][0]), to_q8(S.out[3 * t + 2][1]));
                __m128i rg = _mm_packus_epi16(r16, g16);     // r0..r7 g0..g7
                __m128i bb = _mm_packus_epi16(b16, b16);     // b0..b7 b0..b7
                __m128i o0 = _mm_or_si128(_mm_shuffle_epi8(rg, m1), _mm_shuffle_epi8(bb, m2));   // r0 g0 b0 ... r5
                __m128i o1 = _mm_or_si128(_mm_shuffle_epi8(rg, m3), _mm_shuffle_epi8(bb, m4));   // g5 b5 r6 g6 b6 r7 g7 b7
                uint8_t* d = c.img[t] + ((size_t)py * W + px) * 3;
                if (nvalid == 8) { _mm_storeu_si128((__m128i*)d, o0); _mm_storel_epi64((__m128i*)(d + 16), o1); }
                else { _mm_storeu_si128((__m128i*)S.rgb, o0); _mm_storel_epi64((__m128i*)(S.rgb + 16), o1); memcpy(d, S.rgb, (size_t)nvalid * 3); }
            }
        }
    }
}

// Scalar strict-fp path: reference for --verify, and the decoder for model features outside the SIMD path.
static void decode_rows_scalar(const DecodeCtx& c, int ys, int ye) {
    float f[MAXH], pre[MAXOUT], out[MAXOUT];
    for (int py = ys; py < ye; py++)
        for (int px = 0; px < c.W; px++) {
            ref_features(*c.m, *c.feats, c.W, c.H, px, py, f);
            ref_mlp(*c.m, f, pre, out, c.clamp);
            const size_t pi = (size_t)py * c.W + px;
            if (c.pre) for (int o = 0; o < c.nout; o++) c.pre[pi * c.nout + o] = pre[o];
            if (c.post) for (int o = 0; o < c.nout; o++) c.post[pi * c.nout + o] = out[o];
            for (int t = 0; t < c.T; t++) for (int ch = 0; ch < 3; ch++) c.img[t][pi * 3 + ch] = ref_q8(out[3 * t + ch]);
        }
}
STRICT_FP_END

// ---------------------------------------------------------------- thread pool
struct Pool {
    struct Worker { std::thread th; std::mutex m; std::condition_variable cv; std::atomic<int> gen{0}; };
    std::vector<std::unique_ptr<Worker>> ws;
    std::function<void(int)> job;
    std::atomic<int> done{0};
    std::atomic<bool> quit{false};
    explicit Pool(int nworkers) {
        for (int i = 0; i < nworkers; i++) ws.push_back(std::unique_ptr<Worker>(new Worker()));
        for (int i = 0; i < nworkers; i++) ws[i]->th = std::thread([this, i] { loop(i); });
    }
    ~Pool() {
        quit = true;
        for (auto& w : ws) { { std::lock_guard<std::mutex> lk(w->m); w->gen++; } w->cv.notify_one(); }
        for (auto& w : ws) w->th.join();
    }
    void loop(int i) {
        _mm_setcsr(0x1F80);   // default MXCSR: round-to-nearest, no FTZ/DAZ, same as the main thread
        Worker& w = *ws[i]; int seen = 0;
        for (;;) {
            int spins = 0;
            while (w.gen.load(std::memory_order_acquire) == seen) {
                if (++spins < 100000) _mm_pause();
                else { std::unique_lock<std::mutex> lk(w.m); w.cv.wait(lk, [&] { return w.gen.load(std::memory_order_acquire) != seen; }); }
            }
            seen = w.gen.load(std::memory_order_acquire);
            if (quit.load()) return;
            job(i + 1);
            done.fetch_add(1, std::memory_order_release);
        }
    }
    // Run job(tid) on tid = 0 (the caller) .. nthreads-1; returns when all are done.
    void run(int nthreads, std::function<void(int)> f) {
        const int nw = std::min(nthreads - 1, (int)ws.size());
        job = std::move(f);
        done.store(0);
        for (int i = 0; i < nw; i++) { { std::lock_guard<std::mutex> lk(ws[i]->m); ws[i]->gen++; } ws[i]->cv.notify_one(); }
        job(0);
        while (done.load(std::memory_order_acquire) < nw) _mm_pause();
    }
};

static void decode_image(const DecodeCtx& c, Pool& pool, int nthreads, std::vector<std::unique_ptr<Scratch>>& scratch) {
    std::atomic<int> next{0};
    const int SR = c.strip_rows, nstrips = (c.H + SR - 1) / SR;
    pool.run(nthreads, [&](int tid) {
        Scratch& S = *scratch[tid];
        for (;;) {
            int s = next.fetch_add(1);
            if (s >= nstrips) break;
            int ys = s * SR, ye = std::min(c.H, ys + SR);
            if (c.simd) decode_rows_simd(c, S, ys, ye); else decode_rows_scalar(c, ys, ye);
        }
    });
}

// ---------------------------------------------------------------- helpers
static uint64_t fnv1a(const uint8_t* p, size_t n, uint64_t h = 1469598103934665603ull) { for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 1099511628211ull; } return h; }
static int float_ulp_diff(float a, float b) {
    int32_t ia, ib; memcpy(&ia, &a, 4); memcpy(&ib, &b, 4);
    if (ia < 0) ia = INT32_MIN - ia;   // monotone mapping of the bit pattern
    if (ib < 0) ib = INT32_MIN - ib;
    int64_t d = (int64_t)ia - ib; if (d < 0) d = -d; return d > INT32_MAX ? INT32_MAX : (int)d;
}
static double median(std::vector<double> v) { std::sort(v.begin(), v.end()); size_t n = v.size(); return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]); }
static std::vector<std::string> split(const std::string& s, char sep) { std::vector<std::string> r; size_t a = 0; while (a <= s.size()) { size_t e = s.find(sep, a); if (e == std::string::npos) e = s.size(); r.push_back(s.substr(a, e - a)); a = e + 1; } return r; }

static void usage() {
    printf("ntc_decode model.bin [options]\n"
           "  -o PREFIX            output PNG prefix (default \"decoded\"): PREFIX.png, or PREFIX_tK.png for T > 1\n"
           "  --size W H           decode (padded) size; v11 files store it, older files default to level 0's size\n"
           "  --crop W H           write only the top-left W x H (the unpadded source); v11 files store it\n"
           "  --no-crop            write the full decoded (padded) image\n"
           "  --fp32-latent        use the stored latent values as they are (matches recon_NNNNNN.png of a run with continuous levels)\n"
           "  --q8                 apply the trainer's post-hoc 8-bit quantization to the levels it left continuous (default; matches\n"
           "                       recon_q_final.png). No effect on levels quantized in training (--qat level 0, --qes levels, a v13 DCT-coded level 0)\n"
           "  --clamp | --sigmoid  output nonlinearity (v11 files store it; older files default to sigmoid)\n"
           "  --threads N          1..64 (default: hardware_concurrency)\n"
           "  --threads-sweep      run at 1,2,4,8,16,32 threads (capped at hardware_concurrency) and print a table\n"
           "  --reps N             decode N times (default 1; 10 with --bench or --threads-sweep); report min / median\n"
           "  --bench              timing only: no PNG write\n"
           "  --fast               fast sigmoid: rcpps + Newton step, degree-4 exp polynomial (default)\n"
           "  --exact              exact sigmoid: ~1 ulp vector expf and a true divide\n"
           "  --compare PNG[,PNG]  per-texture reference PNGs: PSNR, max |diff|, diff histogram\n"
           "  --verify             also run the scalar strict-fp reference and report ulp / 8-bit differences\n"
           "  --ghz F              clock for a cycles/sample column\n"
           "  --pack-selectors     hold level 0 as B-bit indices (uint8) and expand through a LUT on the fly (full-res nearest --qat level 0, B <= 4)\n");
}

int main(int argc, char** argv) {
#if defined(DEBUG) || defined(_DEBUG)
    printf("DEBUG build\n");   // an unoptimized build announces itself (CMakeLists.txt gives it the _d suffix)
#endif
    _mm_setcsr(0x1F80);
    if (argc >= 2 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) { usage(); return 0; }
    if (argc < 2) { usage(); return 1; }
    std::string path = argv[1], outprefix = "decoded", compare;
    int sizeW = 0, sizeH = 0, cropW = 0, cropH = 0, threads = 0, reps = 0, clampf = -1;
    bool nocrop = false, fp32 = false, sweep = false, bench = false, fast = true, verify = false, pack = false;
    double ghz = 0;
    for (int i = 2; i < argc; i++) {
        std::string a = argv[i];
        auto need = [&](int k) { if (i + k >= argc) { printf("%s needs %d argument(s)\n", a.c_str(), k); exit(1); } };
        if (a == "-o") { need(1); outprefix = argv[++i]; }
        else if (a == "--size") { need(2); sizeW = atoi(argv[++i]); sizeH = atoi(argv[++i]); }
        else if (a == "--crop") { need(2); cropW = atoi(argv[++i]); cropH = atoi(argv[++i]); }
        else if (a == "--no-crop") nocrop = true;
        else if (a == "--fp32-latent") fp32 = true;
        else if (a == "--q8") fp32 = false;
        else if (a == "--clamp") clampf = 1;
        else if (a == "--sigmoid") clampf = 0;
        else if (a == "--threads") { need(1); threads = atoi(argv[++i]); if (threads < 1 || threads > 64) { printf("--threads needs 1..64\n"); return 1; } }
        else if (a == "--threads-sweep") sweep = true;
        else if (a == "--reps") { need(1); reps = atoi(argv[++i]); if (reps < 1) { printf("--reps needs >= 1\n"); return 1; } }
        else if (a == "--bench") bench = true;
        else if (a == "--fast") fast = true;
        else if (a == "--exact") fast = false;
        else if (a == "--compare") { need(1); compare = argv[++i]; }
        else if (a == "--verify") verify = true;
        else if (a == "--ghz") { need(1); ghz = atof(argv[++i]); }
        else if (a == "--pack-selectors") pack = true;
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else { printf("unknown option %s\n", a.c_str()); usage(); return 1; }
    }
    if (reps == 0) reps = (bench || sweep) ? 10 : 1;

    // ---- load
    Model m; std::string err;
    if (!load_model(path, m, err)) { printf("%s: %s\n", path.c_str(), err.c_str()); return 1; }
    if (m.act != 0) { printf("%s: activation %d is not leaky (relu/tanh/sine were removed in v0.8)\n", path.c_str(), m.act); return 2; }
    if (m.deblock) { printf("%s: deblocking models (removed in v0.8) are not supported\n", path.c_str()); return 2; }
    std::vector<PosFeature> feats;
    if (!parse_pos(m.pos, feats)) { printf("%s: positional spec \"%s\" uses a kind removed in v0.8 (only uv, lv1local, none are decoded)\n", path.c_str(), m.pos.c_str()); return 2; }
    for (const PosFeature& pf : feats) if (pf.kind == POS_LV1LOCAL && m.lv.size() < 2) { printf("%s: positional spec \"%s\" needs a second latent level but the file has one\n", path.c_str(), m.pos.c_str()); return 2; }

    // ---- resolve decode size, crop, clamp
    int W = 0, H = 0;
    if (sizeW > 0 && sizeH > 0) { W = sizeW; H = sizeH; }
    else if (m.version >= 11) { W = m.imgW; H = m.imgH; }
    else if (m.lv[0].nearest) { W = m.lv[0].W; H = m.lv[0].H; }
    else { printf("%s: level 0 is bilinear and the file (v%d) does not store the image size; pass --size W H\n", path.c_str(), m.version); return 2; }
    int cw = W, ch = H;
    if (cropW > 0 && cropH > 0) { cw = cropW; ch = cropH; }
    else if (m.version >= 11 && m.srcW > 0 && m.srcH > 0) { cw = m.srcW; ch = m.srcH; }
    if (nocrop) { cw = W; ch = H; }
    if (cw > W || ch > H) { printf("--crop %dx%d exceeds the decode size %dx%d\n", cw, ch, W, H); return 1; }
    bool clampo = clampf >= 0 ? clampf == 1 : (m.version >= 11 ? m.clamp == 1 : false);
    { int pc = 0; for (const PosFeature& f : feats) pc += pos_count(f); int lc = 0; for (const Level& L : m.lv) lc += L.C;
      if (lc + pc != m.nin) { printf("%s: nin %d != latent channels %d + positional features %d\n", path.c_str(), m.nin, lc, pc); return 2; } }
    const int nout = 3 * m.T;

    // ---- latent preparation (excluded from timing)
    {
        bool any_cont = false; for (size_t l = 0; l < m.lv.size(); l++) if (!level_quantized(m, l)) any_cont = true;
        if (!any_cont) printf("note: every level was quantized in training; --q8 and --fp32-latent decode the same values\n");
    }
    if (!fp32) q8_quantize(m, 8);
    // ---- threads
    const int hc = std::max(1u, std::thread::hardware_concurrency());
    std::vector<int> tlist;
    if (sweep) { const int pts[] = { 1, 2, 4, 8, 16, 32 }; for (int p : pts) if (p <= hc) tlist.push_back(p); if (tlist.empty()) tlist.push_back(1); }
    else tlist.push_back(threads > 0 ? threads : std::min(hc, 64));
    int maxthreads = 1; for (int t : tlist) maxthreads = std::max(maxthreads, t);
    Pool pool(maxthreads - 1);
    std::vector<std::unique_ptr<Scratch>> scratch;
    for (int t = 0; t < maxthreads; t++) scratch.push_back(std::unique_ptr<Scratch>(new Scratch()));
    const int dct_threads = std::min(maxthreads, std::max(1, m.imgH / 8));   // [DCT] the inverse DCT runs on the same pool as the decode stage, at most one thread per block row (this is the count the idct line prints)
    DctReconStats dct; double dct_ms = 0;   // [DCT] hook: a v13 file's level 0 is reconstructed from its symbols here (its own timed stage, excluded from the decode timing below)
    if (m.dct_N > 0) {   // [DCT]
        pool.run(maxthreads, [](int) {});   // [DCT] wake the workers (they sleep on a condition variable after their spin budget) so the timed stage is measured warm, like the decode stage below
        auto t0 = std::chrono::steady_clock::now();   // [DCT]
        dct = dct_recon_level0(m, dct_threads, [&](int n, std::function<void(int)> f) { pool.run(n, f); });   // [DCT]
        dct_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();   // [DCT]
    }   // [DCT]

    // ---- geometry tables
    Axis ax[MAXLV], ay[MAXLV];
    for (size_t l = 0; l < m.lv.size(); l++) { build_axis(W, m.lv[l].W, m.lv[l].nearest, ax[l]); build_axis(H, m.lv[l].H, m.lv[l].nearest, ay[l]); }
    std::vector<PosEntry> pos;
    build_pos_entries(feats, W, H, m.lv.size() > 1 ? &ax[1] : &ax[0], m.lv.size() > 1 ? &ay[1] : &ay[0], pos);
    bool simd = true;   // every surviving model feature is covered; the scalar path remains for --verify

    // level 0 planar copy (full-res nearest) / packed indices
    std::vector<std::vector<float>> planes, luts; std::vector<std::vector<uint8_t>> idxs;
    bool planar = m.lv[0].nearest && m.lv[0].W == W && m.lv[0].H == H;
    if (planar) for (int p = 0; p < W && planar; p++) planar = ax[0].i0[p] == p;
    if (planar) for (int p = 0; p < H && planar; p++) planar = ay[0].i0[p] == p;
    bool packed = false;
    if (planar) {
        const Level& L0 = m.lv[0];
        if (pack && !m.qat_ch.empty()) {
            // B-bit indices plus a 16-entry LUT of the stored grid values per channel; the LUT must
            // reproduce every stored value exactly (it does whenever the file is on the trainer's grid).
            packed = true;
            idxs.resize(L0.C); luts.assign(L0.C, std::vector<float>(16, 0.0f));
            for (int c = 0; c < L0.C && packed; c++) {
                if (m.qat_ch[c] > 4) { packed = false; break; }
                std::vector<bool> seen(16, false);
                idxs[c].assign(L0.size() / L0.C + 8, 0);
                for (size_t i = 0; i < L0.size() / L0.C; i++) {
                    const float v = m.z[i * L0.C + c]; const int k = qat_index(v, m.qat_ch[c]);
                    if (!seen[k]) { seen[k] = true; luts[c][k] = v; } else if (luts[c][k] != v) { packed = false; break; }
                    idxs[c][i] = (uint8_t)k;
                }
            }
            if (!packed) printf("note: --pack-selectors ignored (channels wider than 4 bits, or level-0 values not on one 2^B-point grid)\n");
        }
        if (!packed) {
            planes.resize(L0.C);
            for (int c = 0; c < L0.C; c++) { planes[c].assign(L0.size() / L0.C + 8, 0.0f); for (size_t i = 0; i < L0.size() / L0.C; i++) planes[c][i] = m.z[i * L0.C + c]; }
        }
    }
    if (pack && !packed) printf("note: --pack-selectors ignored (%s)\n", m.dct_N > 0 ? "--pack-selectors does not apply to a DCT-coded level 0" : (planar ? "level 0 is not --qat" : "level 0 is not a full-resolution nearest level"));   // [DCT] hook: the DCT reason

    PackedMLP P; pack_mlp(m, P); P.clamp = clampo;
    const char* evalname = "";
    EvalFn eval = pick_eval(P, &evalname);

    DecodeCtx c;
    c.W = W; c.H = H; c.T = m.T; c.nout = nout; c.nlv = (int)m.lv.size(); c.pos = pos; c.P = &P; c.eval = eval;
    c.simd = simd; c.fast = fast; c.clamp = clampo; c.m = &m; c.feats = &feats;
    for (int l = 0; l < c.nlv; l++) {
        LevelCtx& L = c.lv[l]; L.L = &m.lv[l]; L.z = m.z.data() + m.lv[l].off; L.ax = &ax[l]; L.ay = &ay[l];
        for (int k = 0; k < MAXH; k++) { L.plane[k] = nullptr; L.idx[k] = nullptr; }
        if (l == 0 && planar) {
            L.planar = true;
            for (int k = 0; k < m.lv[0].C; k++) {
                if (packed) {
                    L.idx[k] = idxs[k].data();
                    uint8_t bytes[4][16];
                    for (int e = 0; e < 16; e++) { uint32_t u; memcpy(&u, &luts[k][e], 4); for (int b = 0; b < 4; b++) bytes[b][e] = (uint8_t)(u >> (8 * b)); }
                    for (int b = 0; b < 4; b++) L.lut[k][b] = _mm_loadu_si128((const __m128i*)bytes[b]);
                } else L.plane[k] = planes[k].data();
            }
        }
    }
    if (c.nlv > 1 && m.lv[1].H > 0 && H % m.lv[1].H == 0 && H / m.lv[1].H >= 1 && H / m.lv[1].H <= 32) c.strip_rows = H / m.lv[1].H; else c.strip_rows = 8;

    // ---- summary
    int rc = 0;   // [NTCB] hook: was declared at the compare block; a file: MISMATCH sets it here
    const size_t nsamples = (size_t)W * H;
    const size_t macs_mlp = m.mlp_macs(); const int macs_bil = m.bilinear_macs(); const size_t macs_total = macs_mlp + macs_bil;
    printf("model    : %s (%s)\n", path.c_str(), m.ntcb ? ("ntcb v" + std::to_string(m.ntcb_version)).c_str() : ("v" + std::to_string(m.version)).c_str());   // [NTCB] hook: was `(v%d)`
    printf("image    : %dx%d decode, %dx%d output%s; %zu samples x %d texture(s) = %zu texels\n", W, H, cw, ch, (cw != W || ch != H) ? " (cropped)" : "", nsamples, m.T, nsamples * m.T);
    for (size_t l = 0; l < m.lv.size(); l++) {
        const Level& L = m.lv[l];
        std::string q;
        if (l == 0 && !m.qat_ch.empty()) { q = "qat "; for (size_t k = 0; k < m.qat_ch.size(); k++) q += (k ? "," : "") + std::to_string(m.qat_ch[k]); if (packed) q += ", packed uint8 indices"; else if (l == 0 && planar) q += ", fp32 planar"; }
        else if (l == 0 && m.dct_N > 0) {   // [DCT] hook: the DCT-coded level 0 (no bit depth: quality and DC step only)
            char b[160];   // [DCT]
            std::string qs; for (int c = 0; c < m.dct_C; c++) qs += (c ? "," : "") + std::to_string(m.dct_q[c]);   // [DCT]
            snprintf(b, sizeof(b), "dct %dx%d, q %s, dc step %d%s, 4-bit scale codes, %.1f nonzero ACs/block, %.1f%% EOB-only", m.dct_N, m.dct_N, qs.c_str(), m.dct_dc_step, m.dct_deadzone ? "" : ", plain-rounding ACs (dz 0)",   // [DCT] hook: the %s after the dc step = the --dct-deadzone 0 clause
                dct.nblk ? (double)dct.nnz / (dct.nblk * m.dct_C) : 0.0, dct.nblk ? 100.0 * dct.eob0 / (dct.nblk * m.dct_C) : 0.0);   // [DCT]
            q = b; q += planar ? ", fp32 planar" : "";   // [DCT]
        }   // [DCT]
        else if (m.qes_bits[l] > 0) {
            char b[64]; q = std::string(m.ntcb && m.ntcb_mode[l] == 0 ? "ntcb q8 (post-hoc grid) " : "qes ") + std::to_string(m.qes_bits[l]) + "-bit, range";   // [NTCB] hook: a container's mode-0 level is the run's post-hoc grid, decoded through this branch
            for (int c = 0; c < L.C; c++) { snprintf(b, sizeof(b), " [%.4g, %.4g]", m.qes_lo[l][c], m.qes_hi[l][c]); q += b; }
            q += (l == 0 && planar) ? ", fp32 planar" : "";
        }
        else q = fp32 ? "fp32" : "q8 per channel";
        printf("level %zu  : %dx%dx%d %s, %s%s\n", l, L.W, L.H, L.C, L.nearest ? "nearest" : "bilinear", q.c_str(), (l == 0 && !planar && L.nearest) ? ", gather" : "");
    }
    if (m.dct_N > 0) {   // [DCT] hook: the cheap statistics computed while dequantizing (no entropy tables here) and the IDCT stage
        const size_t nbc = dct.nblk * m.dct_C;   // [DCT]
        const double raw_bits = (double)nbc * dct.dc_raw_bits + (double)dct.nnz * 16.0 + (double)dct.neob * 7.0 + 4.0 * nbc;   // [DCT] same figure as the trainer's [dct: raw ...] bracket (the 96-bit per-channel header is charged separately there)
        printf("dct      : nz mean %.2f/blk, eob0 %.1f%%, lnz mean %.1f, clamped %.2f%%, codes k0..k15", nbc ? (double)dct.nnz / nbc : 0.0, nbc ? 100.0 * dct.eob0 / nbc : 0.0,   // [DCT]
            dct.lnz_n ? dct.lnz_sum / dct.lnz_n : 0.0, nsamples ? 100.0 * dct.clamped / (nsamples * m.dct_C) : 0.0);   // [DCT]
        for (int k = 0; k < 16; k++) printf(" %zu", dct.codehist[k]);   // [DCT]
        printf("; raw %.1f bits/%s = %.4f bpp (fixed-length simulator: %d DC + 16 per nonzero + 7 per EOB + 4 code bits)\n", nbc ? raw_bits / nbc : 0.0, m.dct_C > 1 ? "(blk,ch)" : "blk", raw_bits / nsamples, dct.dc_raw_bits);   // [DCT] hook: `bits/(blk,ch)` when dct_C > 1 (nbc = blocks x channels), `bits/blk` as before when C == 1
        if (m.dct_C > 1) {   // [DCT] per channel: q, nonzero ACs per block, EOB-only blocks, the nonzero count (sums to the line above)
            printf("dct      : per channel");   // [DCT]
            for (int c = 0; c < m.dct_C; c++) printf("%s [%d] q %d nz %.2f/blk eob0 %.1f%% nnz %zu", c ? ";" : "", c, m.dct_q[c], dct.nblk ? (double)dct.nnz_c[c] / dct.nblk : 0.0, dct.nblk ? 100.0 * dct.eob0_c[c] / dct.nblk : 0.0, dct.nnz_c[c]);   // [DCT]
            printf("\n");   // [DCT]
        }   // [DCT]
        printf("idct     : %.2f ms on %d thread(s) (%zu blocks; %.1f MAC/texel with zero-skipping, 16 dense; nnz %zu)\n", dct_ms, dct_threads, nbc, nsamples ? (double)dct.macs / (nsamples * m.dct_C) : 0.0, dct.nnz);   // [DCT]
    }   // [DCT]
    if (m.ntcb) {   // [NTCB] hook: the container's size next to the simulator figure recomputed from its header (mode 0/1/2: W*H*bits per channel + 64 side-info bits
        // [NTCB] per mode-0/1 channel; mode 3: the token bits of the reconstructed symbols + 96 per channel; the MLP at 16 per weight) and the exact-match verdict
        double lvl_bits = 0, side = 0;   // [NTCB]
        for (size_t l = 0; l < m.lv.size(); l++) {   // [NTCB]
            const Level& L = m.lv[l];   // [NTCB]
            if (m.ntcb_mode[l] == 3) { const size_t nbc = dct.nblk * m.dct_C; lvl_bits += (double)nbc * dct.dc_raw_bits + (double)dct.nnz * 16.0 + (double)dct.neob * 7.0 + 4.0 * nbc; side += 96.0 * L.C; }   // [NTCB]
            else { for (int c = 0; c < L.C; c++) lvl_bits += (double)L.W * L.H * (m.ntcb_mode[l] == 2 ? m.qat_ch[c] : m.ntcb_bits[l]); if (m.ntcb_mode[l] <= 1) side += 64.0 * L.C; }   // [NTCB]
        }   // [NTCB]
        const double mlp_bits = 16.0 * (double)m.p.size(), sim = lvl_bits + side + mlp_bits, np = (double)m.imgW * m.imgH;   // [NTCB]
        const bool match = m.ntcb_content_bits == lvl_bits + mlp_bits;   // [NTCB]
        char tex[64] = "", stex[64] = "";   // [NTCB]
        if (m.T > 1) { snprintf(tex, sizeof(tex), " (%.3f/tex)", 8.0 * m.ntcb_bytes / np / m.T); snprintf(stex, sizeof(stex), " (%.3f/tex)", sim / np / m.T); }   // [NTCB]
        printf("file     : %s %zu bytes = %.3f bpp%s (simulator raw %.3f bpp%s: %.0f level bits + %.0f side-info (stored in the header) + %.0f mlp bits; content %.0f bits %s; container header %d + section headers %d + sync %.0f + padding %.0f bits; bpp over the %dx%d decode size, as every bpp in this log)\n",   // [NTCB]
            path.c_str(), m.ntcb_bytes, 8.0 * m.ntcb_bytes / np, tex, sim / np, stex, lvl_bits, side, mlp_bits, m.ntcb_content_bits,   // [NTCB]
            match ? "matches the simulator exactly" : "MISMATCH against the simulator", 8 * m.ntcb_H, 64 * m.ntcb_nsections, m.ntcb_sync_bits, m.ntcb_pad_bits, m.imgW, m.imgH);   // [NTCB]
        if (!match) rc = 1;   // [NTCB] a MISMATCH is a failure (the token bookkeeping here disagrees with the file): exit 1 after the summary and the outputs
    }   // [NTCB]
    { int pc = m.nin; for (const Level& L : m.lv) pc -= L.C; printf("pos      : %s (%d inputs)\n", m.pos.empty() ? "none" : m.pos.c_str(), pc); }
    { std::string s = std::to_string(m.nin); for (int h : m.hidden) s += " -> " + std::to_string(h); s += " -> " + std::to_string(nout);
      printf("mlp      : %s = %zu params, leaky hidden (leak %s), %s output\n", s.c_str(), m.p.size(), std::to_string(m.leak).c_str(), clampo ? "clamp" : "sigmoid"); }
    printf("mac      : %zu mlp + %d bilinear = %zu per sample\n", macs_mlp, macs_bil, macs_total);
    { std::string pth = std::string("SSE4.1, ") + evalname;
      printf("path     : %s; nonlinearity %s\n", pth.c_str(), clampo ? "clamp" : (fast ? "fast (rcpps+NR, degree-4 exp)" : "exact (vector expf, divps)")); }

    std::vector<std::vector<uint8_t>> img(m.T);
    for (int t = 0; t < m.T; t++) { img[t].assign(nsamples * 3, 0); c.img[t] = img[t].data(); }

    // ---- decode / benchmark
    printf("timing   : quantized latent in memory -> RGB8 in memory (latent sampling, MLP, nonlinearity, RGB8 pack); excludes file load, quantization, weight packing, geometry tables, PNG encode%s\n", m.dct_N > 0 ? " and the level-0 IDCT (the idct line above)" : "");   // [DCT] hook: the %s
    printf("%-6s %7s %9s %10s %11s %10s %8s %10s%s   %s\n", "mode", "threads", "min ms", "median ms", "Msamples/s", "Mtexel/s", "GMAC/s", "ns/sample", ghz > 0 ? "  cyc/sample" : "", "rgb8 hash");
    uint64_t hash0 = 0; bool hash_same = true;
    for (size_t ti = 0; ti < tlist.size(); ti++) {
        const int nt = tlist[ti];
        std::vector<double> ms;
        pool.run(maxthreads, [](int) {});   // wake the workers before the timed reps (they went to sleep during the setup above); the same warm-up the idct stage gets
        for (int r = 0; r < reps; r++) {
            auto t0 = std::chrono::steady_clock::now();
            decode_image(c, pool, nt, scratch);
            auto t1 = std::chrono::steady_clock::now();
            ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        uint64_t h = 1469598103934665603ull; for (int t = 0; t < m.T; t++) h = fnv1a(img[t].data(), img[t].size(), h);
        if (ti == 0) hash0 = h; else if (h != hash0) hash_same = false;
        double mn = *std::min_element(ms.begin(), ms.end()), md = median(ms);
        double sps = nsamples / (mn * 1e-3);
        printf("%-6s %7d %9.3f %10.3f %11.2f %10.2f %8.2f %10.2f", clampo ? "clamp" : (fast ? "fast" : "exact"), nt, mn, md, sps * 1e-6, sps * m.T * 1e-6, sps * macs_total * 1e-9, 1e9 / sps);
        if (ghz > 0) printf(" %12.1f", ghz * 1e9 / sps);
        printf("   %016llx\n", (unsigned long long)h);
    }
    if (tlist.size() > 1) printf("rgb8 output %s across thread counts\n", hash_same ? "identical" : "DIFFERS");
    printf("(min-time rates; GMAC/s counts %zu MAC/sample; %d rep(s) per row)\n", macs_total, reps);

    // ---- outputs
    if (!bench) {
        for (int t = 0; t < m.T; t++) {
            std::string name = outprefix + (m.T > 1 ? "_t" + std::to_string(t) : "") + ".png";
            if (!stbi_write_png(name.c_str(), cw, ch, 3, img[t].data(), W * 3)) { printf("cannot write %s\n", name.c_str()); return 1; }
            printf("wrote %s (%dx%d)\n", name.c_str(), cw, ch);
        }
    }

    // ---- compare
    if (!compare.empty()) {
        std::vector<std::string> refs = split(compare, ',');
        if ((int)refs.size() != m.T) { printf("--compare: %zu file(s) for %d texture(s)\n", refs.size(), m.T); return 1; }
        for (int t = 0; t < m.T; t++) {
            int rw = 0, rh = 0, rn = 0;
            unsigned char* ref = stbi_load(refs[t].c_str(), &rw, &rh, &rn, 3);
            if (!ref) { printf("cannot load %s\n", refs[t].c_str()); return 1; }
            // Basis Universal's image_metrics::calc: compare on the minimum common size, 8-bit values,
            // MSE over all pixels and channels, PSNR = 20 log10(255 / rms) clamped to [0, 100].
            const int ow = cw > 0 ? cw : W, oh = ch > 0 ? ch : H;   // the written output extent
            const int vw = std::min(rw, ow), vh = std::min(rh, oh);
            if (rw != ow || rh != oh) printf("compare t%d: %s is %dx%d, the output is %dx%d; comparing the common %dx%d\n", t, refs[t].c_str(), rw, rh, ow, oh, vw, vh);
            size_t hist[4] = { 0, 0, 0, 0 }; int mx = 0; double se = 0;
            for (int y = 0; y < vh; y++) for (int x = 0; x < vw; x++) for (int k = 0; k < 3; k++) {
                int d = std::abs((int)img[t][((size_t)y * W + x) * 3 + k] - (int)ref[((size_t)y * rw + x) * 3 + k]);
                se += (double)d * d; mx = std::max(mx, d); hist[std::min(d, 3)]++;
            }
            const double n = (double)vw * vh * 3, mse = se / n;
            char psnr[32]; snprintf(psnr, sizeof(psnr), "%.2f", mse > 0 ? std::min(100.0, std::max(0.0, 10.0 * std::log10(255.0 * 255.0 / mse))) : 100.0);
            printf("compare t%d: %s (%dx%d): PSNR %s dB, max |diff| %d, |diff| histogram: 0: %zu (%.4f%%), 1: %zu (%.4f%%), 2: %zu (%.4f%%), >2: %zu (%.4f%%)\n",
                t, refs[t].c_str(), vw, vh, psnr, mx,
                hist[0], 100.0 * hist[0] / n, hist[1], 100.0 * hist[1] / n, hist[2], 100.0 * hist[2] / n, hist[3], 100.0 * hist[3] / n);
            stbi_image_free(ref);
        }
    }

    // ---- verify: scalar strict reference vs the SIMD path
    if (verify) {
        if (!simd) printf("verify: the SIMD path is not in use for this model; nothing to compare\n");
        else {
            std::vector<float> pre_s(nsamples * nout), post_s(nsamples * nout), pre_r(nsamples * nout), post_r(nsamples * nout);
            std::vector<std::vector<uint8_t>> img_r(m.T);
            DecodeCtx cs = c; cs.pre = pre_s.data(); cs.post = post_s.data();
            decode_image(cs, pool, maxthreads, scratch);
            DecodeCtx cr = c; cr.simd = false; cr.pre = pre_r.data(); cr.post = post_r.data();
            for (int t = 0; t < m.T; t++) { img_r[t].assign(nsamples * 3, 0); cr.img[t] = img_r[t].data(); }
            decode_image(cr, pool, maxthreads, scratch);
            int max_ulp_pre = 0, max_ulp_post = 0; double max_abs_post = 0; size_t n_pre_diff = 0, n8 = 0;
            for (size_t i = 0; i < nsamples * nout; i++) {
                int u = float_ulp_diff(pre_s[i], pre_r[i]); if (u) n_pre_diff++; max_ulp_pre = std::max(max_ulp_pre, u);
                max_ulp_post = std::max(max_ulp_post, float_ulp_diff(post_s[i], post_r[i]));
                max_abs_post = std::max(max_abs_post, (double)std::fabs(post_s[i] - post_r[i]));
            }
            for (int t = 0; t < m.T; t++) for (size_t i = 0; i < nsamples * 3; i++) if (img[t][i] != img_r[t][i]) n8++;
            printf("verify   : pre-nonlinearity: max %d ulp (%zu of %zu values differ); output fp32: max %d ulp, max |diff| %.3g; RGB8 mismatches: %zu of %zu (%.5f%%)\n",
                max_ulp_pre, n_pre_diff, nsamples * nout, max_ulp_post, max_abs_post, n8, nsamples * nout, 100.0 * n8 / (nsamples * nout));
            if (max_ulp_pre != 0) rc = 3;
        }
    }
    return rc;
}
