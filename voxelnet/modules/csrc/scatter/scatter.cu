
#include "scatter.h"
#include "atomics.cuh"

#define THREADS 1024
#define BLOCKS(N) (N + THREADS - 1) / THREADS

template <typename scalar_t>
__global__ void scatter_kernel(const scalar_t *src_data,
               const int64_t *indices_data,
               scalar_t *out_data,
               int64_t N, int64_t M, int64_t C) {

  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

  int m = thread_idx / C;
  int c = thread_idx % C;
  if(m < M){
    int n1 = indices_data[m*2];
    int n2 = indices_data[m*2+1];
    atomMax(out_data+n2*C+c, src_data[n1*C+c]);
  }
}

template <typename scalar_t>
__global__ void scatter_arg_kernel(const scalar_t *src_data,
                    const int64_t *indices_data,
                   const scalar_t *out_data, int64_t *arg_out_data,
                     int N, int M, int C) {

  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

  int m = thread_idx / C;
  int c = thread_idx % C;
  if(m < M){
    int n1 = indices_data[m*2];
    int n2 = indices_data[m*2+1];
    if (src_data[n1*C+c] == out_data[n2*C+c]) {
       arg_out_data[n2*C+c] = n1;
    }
  }
}

template <typename scalar_t>
__global__ void scatter_back_kernel(const scalar_t *src_data,
                    const int64_t *indices_data,
                   scalar_t *out_data,
                     int64_t N, int64_t C) {

  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

  int n = thread_idx / C;
  int c = thread_idx % C;
  if(n < N){
    int n1 = indices_data[n*C+c];
    if (n1>=0){
        atomAdd(out_data + n1*C+c, src_data[n*C+c]);
    }
  }
}

std::tuple<torch::Tensor, torch::Tensor>
scatter_max_cuda(torch::Tensor src, torch::Tensor index) {
  src = src.contiguous();
  auto  N = src.size(0);
  auto  C = src.size(1);
  auto  M = index.size(0);
  auto  O = M*C;

  
  torch::Tensor out = torch::empty({N, C}, src.options());
  torch::Tensor arg_out = torch::full({N, C},-1, index.options());

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        src.type(), "scatter_max_cuda", ([&] {
    auto src_data = src.data_ptr<scalar_t>();
    auto out_data = out.data_ptr<scalar_t>();
    auto indices_data = index.data_ptr<int64_t>();
    auto arg_out_data = arg_out.data_ptr<int64_t>();

    out.fill_(std::numeric_limits<scalar_t>::lowest());

    scatter_kernel<scalar_t><<<BLOCKS(O), THREADS>>>(src_data,indices_data, out_data, N,M,C);

    out.masked_fill_(out == std::numeric_limits<scalar_t>::lowest(),(scalar_t)0);

    scatter_arg_kernel<scalar_t><<<BLOCKS(O), THREADS>>>(
                src_data, indices_data, out_data, arg_out_data,N,M,C);
  }));

  return std::make_tuple(out, arg_out);
}

torch::Tensor scatter_backward_cuda(torch::Tensor src, torch::Tensor index) {
  src = src.contiguous();
  index = index.contiguous();
  auto  N = src.size(0);
  auto  C = src.size(1);
  auto  O = N*C;

  
  torch::Tensor out = torch::zeros({N, C}, src.options());

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        src.type(), "scatter_backward_cuda", ([&] {
    auto src_data = src.data_ptr<scalar_t>();
    auto out_data = out.data_ptr<scalar_t>();
    auto indices_data = index.data_ptr<int64_t>();
    scatter_back_kernel<scalar_t><<<BLOCKS(O), THREADS>>>(src_data,indices_data, out_data, N,C);
  }));

  return out;
}