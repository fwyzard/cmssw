#include <cstdio>
#include <iostream>
#include <cuda_runtime.h>
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

__global__
void scrambler_kernel(const char * message, size_t length)
{
  //printf("blockIdx.x, threadIdx.x: %d, %d\n", blockIdx.x, threadIdx.x);
  for (size_t i = blockDim.x * blockIdx.x + threadIdx.x; i < length; i += blockDim.x * gridDim.x)
    printf("%c", message[i]);
}

void scrambler_wrapper(const char * message, size_t length) {
  scrambler_kernel<<<16,1>>>(message, length);
  cudaCheck(cudaDeviceSynchronize());
  cudaCheck(cudaGetLastError());
  std::cout << std::endl;

  scrambler_kernel<<<4,4>>>(message, length);
  cudaCheck(cudaDeviceSynchronize());
  cudaCheck(cudaGetLastError());
  std::cout << std::endl;

  scrambler_kernel<<<1,16>>>(message, length);
  cudaCheck(cudaDeviceSynchronize());
  cudaCheck(cudaGetLastError());
  std::cout << std::endl;
}
