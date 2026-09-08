// Counter-based, stateless Gaussian noise shared by the CUDA backend and its host-side
// checks. Every perturbation the GPU trainer uses is a pure function of
// (seed, stream, iteration, pair, flat index), so the same value can be regenerated
// wherever it is needed (the perturbed evaluation and the gradient accumulation) and
// the host can reproduce any device draw to float rounding (logf/cosf differ between
// the CRT and the CUDA math library by an ulp or two). Nothing is stored, nothing can
// desynchronize, and two device runs with the same seed are byte-identical.
#pragma once
#include <stdint.h>
#include <math.h>

#ifdef __CUDACC__
#define NTC_HD __host__ __device__ __forceinline__
#else
#define NTC_HD inline
#endif

enum NtcNoiseStream : uint32_t { NS_LAT = 1, NS_MLP = 2, NS_BATCH_PIX = 3, NS_BATCH_TEX = 4 };

NTC_HD uint64_t ntc_mix64(uint64_t x) {   // splitmix64 finalizer
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

NTC_HD uint64_t ntc_noise_key(uint64_t seed, uint32_t stream, uint32_t step, uint32_t pair, uint64_t index) {
    uint64_t a = ntc_mix64(seed ^ ((uint64_t)stream << 56) ^ ((uint64_t)step << 32) ^ (uint64_t)pair);
    return ntc_mix64(a + index * 0x9E3779B97F4A7C15ull);
}

NTC_HD uint32_t ntc_noise_u32(uint64_t seed, uint32_t stream, uint32_t step, uint32_t pair, uint64_t index) {
    return (uint32_t)(ntc_noise_key(seed, stream, step, pair, index) >> 32);
}

// Uniform integer in [0, n) from a 32-bit draw (multiply-high map).
NTC_HD uint32_t ntc_noise_uniform(uint32_t u, uint32_t n) {
    return (uint32_t)(((uint64_t)u * (uint64_t)n) >> 32);
}

// One standard normal sample (Box-Muller, cosine branch). u1 in (0, 1], u2 in [0, 1).
NTC_HD float ntc_gauss(uint64_t seed, uint32_t stream, uint32_t step, uint32_t pair, uint64_t index) {
    uint64_t h = ntc_noise_key(seed, stream, step, pair, index);
    float u1 = (float)((h >> 40) + 1ull) * (1.0f / 16777216.0f);          // 24 bits, (0, 1]
    float u2 = (float)(h & 0xFFFFFFull) * (1.0f / 16777216.0f);            // 24 bits, [0, 1)
    return sqrtf(-2.0f * logf(u1)) * cosf(6.283185307179586f * u2);
}
