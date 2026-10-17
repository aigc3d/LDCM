#include <torch/extension.h>

std::vector<at::Tensor> residual_stats_cuda(at::Tensor value);
void residual_control_cuda(at::Tensor value, at::Tensor last, at::Tensor threshold,
    at::Tensor rho, at::Tensor beta, at::Tensor active, at::Tensor steps,
    at::Tensor stalled, at::Tensor cap);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("residual_stats", &residual_stats_cuda);
  m.def("residual_control", &residual_control_cuda);
}
