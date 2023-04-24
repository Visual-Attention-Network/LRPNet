#pragma once
#include <torch/torch.h>

std::tuple<torch::Tensor, torch::Tensor>
scatter_max_cuda(torch::Tensor src, torch::Tensor index);
torch::Tensor
scatter_backward_cuda(torch::Tensor src, torch::Tensor index);