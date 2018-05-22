#ifndef HeterogeneousCore_Examples_test_kernel_cuh
#define HeterogeneousCore_Examples_test_kernel_cuh

#include <cuda_runtime.h>

__global__
void scrambler_kernel(const char * message, size_t length);

void scrambler_wrapper(const char * message, size_t length);

#endif // HeterogeneousCore_Examples_test_kernel_cuh
