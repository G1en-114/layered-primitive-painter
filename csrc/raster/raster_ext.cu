// Tile rasterizer for ordered generalized 2D Gaussians: forward + backward.
//
// One source for both vendors:
//   NVIDIA  compiled by nvcc (warpSize = 32)
//   Hygon DCU / AMD  hipified automatically by torch.utils.cpp_extension, compiled by hipcc
//                    (wavefront, i.e. warpSize = 64)
// Never hard-code 32: use warpSize, and the SHFL_DOWN macro below.
//
// Same math as python/lpp/render/reference.py (the oracle) and triton_raster.py:
//   a(p) = min(alpha * (g(min(d2, c2)) - g(c2)) / (1 - g(c2)), max_alpha),  g(t) = exp(-0.5 t^beta)
//   forward (back to front):   P <- P (1 - a) + c a,   T <- T (1 - a)
//   backward (front to back), S = transmittance in front, F = paint from i and everything in front:
//       dL/dc_i = sum_px gP * a_i S
//       dL/da_i = S <gP, c_i> - (<gP, P - F> + gT T) / (1 - a_i)
//
// Work decomposition (v1): one thread block per (image, 16x16 tile), one thread per pixel.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#define TILE 16
#define BLOCK (TILE * TILE)
#define NPARAM 12
#define NGRAD 10
#define MAX_WAVES (BLOCK / 32)   // 8 on NVIDIA; DCU uses 4 of them

#if defined(__HIPCC__) || defined(USE_ROCM)
#define SHFL_DOWN(v, d) __shfl_down((v), (d))
#else
#define SHFL_DOWN(v, d) __shfl_down_sync(0xffffffffu, (v), (d))
#endif

struct Prim {
  float x, y, A, B, C, beta, alpha, r, g, b, gcut, dgcut;
};

__device__ __forceinline__ Prim load_prim(const float* __restrict__ p) {
  return Prim{p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11]};
}

struct Eval {
  float a, a_raw, f, g, tpow, d2, d2c, dx, dy;
};

__device__ __forceinline__ Eval eval_alpha(const Prim& q, float px, float py, float cut2, float max_alpha) {
  Eval e;
  e.dx = px - q.x;
  e.dy = py - q.y;
  e.d2 = q.A * e.dx * e.dx + 2.f * q.B * e.dx * e.dy + q.C * e.dy * e.dy;
  e.d2c = fminf(fmaxf(e.d2, 1e-12f), cut2);
  e.tpow = exp2f(q.beta * log2f(e.d2c));     // d2c ^ beta
  e.g = expf(-0.5f * e.tpow);
  e.f = (e.g - q.gcut) / (1.f - q.gcut);
  e.a_raw = q.alpha * e.f;
  e.a = fminf(e.a_raw, max_alpha);
  return e;
}

// ------------------------------------------------------------------------------------ forward
__global__ void fwd_kernel(const float* __restrict__ prm, const int* __restrict__ lists,
                           const int* __restrict__ counts, float* __restrict__ P, float* __restrict__ T,
                           int N, int H, int W, int tiles_x, int n_tiles, float inv_scale, float cut2,
                           float max_alpha) {
  const int pid = blockIdx.x, b = pid / n_tiles, t = pid % n_tiles;
  const int iy = (t / tiles_x) * TILE + threadIdx.x / TILE;
  const int ix = (t % tiles_x) * TILE + threadIdx.x % TILE;
  const bool inb = iy < H && ix < W;
  const float px = (ix + 0.5f) * inv_scale, py = (iy + 0.5f) * inv_scale;

  float ar = 0.f, ag = 0.f, ab = 0.f, tr = 1.f;
  const int count = counts[pid];
  const int* lst = lists + (int64_t)pid * N;
  for (int j = 0; j < count; ++j) {
    const Prim q = load_prim(prm + ((int64_t)b * N + lst[j]) * NPARAM);
    const float a = inb ? eval_alpha(q, px, py, cut2, max_alpha).a : 0.f;
    const float om = 1.f - a;
    ar = ar * om + q.r * a;
    ag = ag * om + q.g * a;
    ab = ab * om + q.b * a;
    tr *= om;
  }
  if (inb) {
    const int64_t img = (int64_t)b * H * W, pix = (int64_t)iy * W + ix;
    P[img * 3 + pix] = ar;
    P[img * 3 + (int64_t)H * W + pix] = ag;
    P[img * 3 + 2 * (int64_t)H * W + pix] = ab;
    T[img + pix] = tr;
  }
}

// ------------------------------------------------------------------------------------ backward
__device__ __forceinline__ float wave_sum(float v) {
  for (int off = warpSize / 2; off > 0; off >>= 1) v += SHFL_DOWN(v, off);
  return v;  // valid in lane 0
}

__global__ void bwd_kernel(const float* __restrict__ prm, const int* __restrict__ lists,
                           const int* __restrict__ counts, const float* __restrict__ P,
                           const float* __restrict__ T, const float* __restrict__ gP,
                           const float* __restrict__ gT, float* __restrict__ grads, int N, int H, int W,
                           int tiles_x, int n_tiles, float inv_scale, float cut2, float max_alpha) {
  // double-buffered per-wave partial sums: one __syncthreads per primitive
  __shared__ float part[2][MAX_WAVES][NGRAD];
  const int pid = blockIdx.x, b = pid / n_tiles, t = pid % n_tiles;
  const int lane = threadIdx.x % warpSize, wave = threadIdx.x / warpSize;
  const int n_waves = BLOCK / warpSize;
  const int iy = (t / tiles_x) * TILE + threadIdx.x / TILE;
  const int ix = (t % tiles_x) * TILE + threadIdx.x % TILE;
  const bool inb = iy < H && ix < W;
  const float px = (ix + 0.5f) * inv_scale, py = (iy + 0.5f) * inv_scale;

  const int64_t HW = (int64_t)H * W, img = (int64_t)b * HW, pix = (int64_t)iy * W + ix;
  float Pr = 0.f, Pg = 0.f, Pb = 0.f, Tf = 0.f, gr = 0.f, gg = 0.f, gb = 0.f, gt = 0.f;
  if (inb) {
    Pr = P[img * 3 + pix]; Pg = P[img * 3 + HW + pix]; Pb = P[img * 3 + 2 * HW + pix];
    Tf = T[img + pix];
    gr = gP[img * 3 + pix]; gg = gP[img * 3 + HW + pix]; gb = gP[img * 3 + 2 * HW + pix];
    gt = gT[img + pix];
  }
  const float gtT = gt * Tf;
  float S = 1.f, Fr = 0.f, Fg = 0.f, Fb = 0.f;

  const int count = counts[pid];
  const int* lst = lists + (int64_t)pid * N;
  for (int jj = 0; jj < count; ++jj) {
    const int n = lst[count - 1 - jj];
    const Prim q = load_prim(prm + ((int64_t)b * N + n) * NPARAM);
    Eval e = eval_alpha(q, px, py, cut2, max_alpha);
    const float a = inb ? e.a : 0.f;

    const float w = a * S;
    Fr += q.r * w; Fg += q.g * w; Fb += q.b * w;
    const float inv1m = 1.f / (1.f - a);
    float dLda = S * (gr * q.r + gg * q.g + gb * q.b)
               - (gr * (Pr - Fr) + gg * (Pg - Fg) + gb * (Pb - Fb) + gtT) * inv1m;
    S *= 1.f - a;

    const bool live = inb && e.a_raw < max_alpha;   // clamped pixels pass no gradient
    dLda = live ? dLda : 0.f;
    const float one_m_gc = 1.f - q.gcut;
    const bool in_range = e.d2 > 1e-12f && e.d2 < cut2;
    const float dfdd2 = in_range ? e.g * (-0.5f * q.beta * e.tpow / e.d2c) / one_m_gc : 0.f;
    const float dgdbeta = e.g * (-0.5f * e.tpow * logf(e.d2c));
    const float dfdbeta = (dgdbeta - q.dgcut) / one_m_gc + (e.g - q.gcut) * q.dgcut / (one_m_gc * one_m_gc);
    const float dLdf = dLda * q.alpha;
    const float dLdd2 = dLdf * dfdd2;

    float v[NGRAD] = {
        dLdd2 * -(2.f * q.A * e.dx + 2.f * q.B * e.dy),
        dLdd2 * -(2.f * q.B * e.dx + 2.f * q.C * e.dy),
        dLdd2 * e.dx * e.dx,
        dLdd2 * 2.f * e.dx * e.dy,
        dLdd2 * e.dy * e.dy,
        dLdf * dfdbeta,
        dLda * e.f,
        gr * w, gg * w, gb * w,
    };
    const int buf = jj & 1;
#pragma unroll
    for (int k = 0; k < NGRAD; ++k) {
      const float s = wave_sum(v[k]);
      if (lane == 0) part[buf][wave][k] = s;
    }
    __syncthreads();
    if (threadIdx.x < NGRAD) {   // fixed summation order over waves: deterministic
      float s = 0.f;
      for (int w2 = 0; w2 < n_waves; ++w2) s += part[buf][w2][threadIdx.x];
      grads[((int64_t)pid * N + n) * NGRAD + threadIdx.x] = s;
    }
    // the next iteration writes the other buffer; the one after that is only reached
    // after the next __syncthreads, by which time these reads are done
  }
}

// ------------------------------------------------------------------------------------ host
static void check(const torch::Tensor& x, const char* name) {
  TORCH_CHECK(x.is_cuda(), name, " must be a GPU tensor");
  TORCH_CHECK(x.is_contiguous(), name, " must be contiguous");
}

std::vector<torch::Tensor> raster_fwd(torch::Tensor prm, torch::Tensor lists, torch::Tensor counts,
                                      int64_t B, int64_t N, int64_t size, int64_t tiles_x, double inv_scale,
                                      double cut2, double max_alpha) {
  check(prm, "prm"); check(lists, "lists"); check(counts, "counts");
  auto opts = prm.options();
  auto P = torch::empty({B, 3, size, size}, opts);
  auto T = torch::empty({B, 1, size, size}, opts);
  const int n_tiles = tiles_x * tiles_x;
  if (B > 0) {
    fwd_kernel<<<B * n_tiles, BLOCK, 0, at::cuda::getCurrentCUDAStream()>>>(
        prm.data_ptr<float>(), lists.data_ptr<int>(), counts.data_ptr<int>(), P.data_ptr<float>(),
        T.data_ptr<float>(), N, size, size, tiles_x, n_tiles, inv_scale, cut2, max_alpha);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {P, T};
}

torch::Tensor raster_bwd(torch::Tensor prm, torch::Tensor lists, torch::Tensor counts, torch::Tensor P,
                         torch::Tensor T, torch::Tensor gP, torch::Tensor gT, int64_t B, int64_t N,
                         int64_t size, int64_t tiles_x, double inv_scale, double cut2, double max_alpha) {
  for (auto* x : {&prm, &lists, &counts, &P, &T, &gP, &gT}) check(*x, "input");
  const int n_tiles = tiles_x * tiles_x;
  auto grads = torch::zeros({B, n_tiles, N, NGRAD}, prm.options());
  if (B > 0) {
    bwd_kernel<<<B * n_tiles, BLOCK, 0, at::cuda::getCurrentCUDAStream()>>>(
        prm.data_ptr<float>(), lists.data_ptr<int>(), counts.data_ptr<int>(), P.data_ptr<float>(),
        T.data_ptr<float>(), gP.data_ptr<float>(), gT.data_ptr<float>(), grads.data_ptr<float>(), N, size,
        size, tiles_x, n_tiles, inv_scale, cut2, max_alpha);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return grads.sum(1);   // [B, N, NGRAD], fixed-order reduction over tiles
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fwd", &raster_fwd, "tile rasterizer forward");
  m.def("bwd", &raster_bwd, "tile rasterizer backward");
}
