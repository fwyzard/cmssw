#ifndef HeterogeneousCore_CUDAUtilities_interface_CUDALaunchConfiguration_h
#define HeterogeneousCore_CUDAUtilities_interface_CUDALaunchConfiguration_h

#include <cuda_runtime.h>

struct CUDALaunchConfiguration {
  dim3 gridSize;
  dim3 blockSize;
};

#endif // HeterogeneousCore_CUDAUtilities_interface_CUDALaunchConfiguration_h
