// Keep ATen's reduction configuration/tree while sharing residual reads.
#include <ATen/ATen.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/cuda/Reduce.cuh>
#include <c10/cuda/CUDAGuard.h>

namespace {
struct Control {
  float *last, *rho, *beta;
  const float *threshold;
  bool *active;
  int *steps, *stalled;
  const int *cap;

  __device__ void update(float energy, float norm) const {
    if (active == nullptr || !*active) return;
    const float improvement=__fdiv_rn(__fsub_rn(*last,norm),__fadd_rn(*last,1e-8f));
    const int idle=improvement<1e-4f ? *stalled+1 : 0;
    const int step=*steps+1;
    *beta=__fdiv_rn(energy,*rho==0.f ? 1e-8f : *rho);
    *rho=energy;
    *last=norm;
    *active=!(norm<=*threshold) && idle<10 && step<*cap;
    *steps=step;
    *stalled=idle;
  }
};

struct StatsOps {
  using acc_t=thrust::pair<float,float>;
  Control control{};

  __device__ acc_t reduce(acc_t a,float v,int64_t) const {
    // sum(r*r) first rounds each product; NormTwoOps contracts its multiply-add.
    // Preserve both accumulators rather than replacing norm with sqrt(energy).
    return {__fadd_rn(a.first,__fmul_rn(v,v)),__fmaf_rn(v,v,a.second)};
  }
  __device__ acc_t combine(acc_t a,acc_t b) const {
    return {a.first+b.first,a.second+b.second};
  }
  __device__ acc_t project(acc_t a) const {
    const float norm=at::native::device_sqrt(a.second);
    control.update(a.first,norm);
    return {a.first,norm};
  }
  __device__ acc_t translate_idx(acc_t a,int64_t) const {return a;}
  __device__ acc_t warp_shfl_down(acc_t a,int offset) const {
    return {WARP_SHFL_DOWN(a.first,offset),WARP_SHFL_DOWN(a.second,offset)};
  }
};

__global__ void scalar_stats(const float* value,float* energy,float* norm,int batch,Control control) {
  const int b=blockIdx.x*blockDim.x+threadIdx.x;
  if (b<batch) {
    const float x=value[b];
    const float squared=__fmul_rn(x,x);
    const float absolute=fabsf(x);
    energy[b]=squared;
    norm[b]=absolute;
    control.update(squared,absolute);
  }
}

std::vector<at::Tensor> stats(at::Tensor value,Control control) {
  TORCH_CHECK(value.is_cuda() && value.scalar_type()==at::kFloat && value.is_contiguous());
  TORCH_CHECK(value.dim()==3 && value.size(2)==1 && value.numel()>0);
  c10::cuda::CUDAGuard guard(value.device());
  auto energy=at::empty({value.size(0),1,1},value.options());
  auto norm=at::empty_like(energy);
  if (value.size(1)==1) {
    // linalg_vector_norm uses abs, not sqrt(x*x), for a singleton reduction.
    scalar_stats<<<(value.size(0)+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(
      value.data_ptr<float>(),energy.data_ptr<float>(),norm.data_ptr<float>(),int(value.size(0)),control);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {energy,norm};
  }
  auto iter=at::native::make_reduction("poisson_stats",energy,norm,value,{1},true,at::kFloat);
  at::native::gpu_reduce_kernel<float,float>(iter,StatsOps{control},StatsOps::acc_t{0.f,0.f});
  return {energy,norm};
}
}

std::vector<at::Tensor> residual_stats_cuda(at::Tensor value) {
  return stats(value,Control{});
}

void residual_control_cuda(at::Tensor value,at::Tensor last,at::Tensor threshold,
    at::Tensor rho,at::Tensor beta,at::Tensor active,at::Tensor steps,
    at::Tensor stalled,at::Tensor cap) {
  // Only one image: the batch-wide stopping rule becomes a scalar decision.
  TORCH_CHECK(value.size(0)==1);
  Control control{last.data_ptr<float>(),rho.data_ptr<float>(),beta.data_ptr<float>(),
    threshold.data_ptr<float>(),active.data_ptr<bool>(),steps.data_ptr<int>(),
    stalled.data_ptr<int>(),cap.data_ptr<int>()};
  stats(value,control);
}
