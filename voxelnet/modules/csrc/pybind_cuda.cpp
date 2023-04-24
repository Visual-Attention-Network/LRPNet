#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include "scatter/scatter.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("scatter_max_cuda", &scatter_max_cuda, "scatter_max_cuda");
  m.def("scatter_backward_cuda", &scatter_backward_cuda, "scatter_backward_cuda");
}
