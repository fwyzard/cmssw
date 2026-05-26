#ifndef HeterogeneousTest_AlpakaPlugin_plugins_alpaka_Kernels_h
#define HeterogeneousTest_AlpakaPlugin_plugins_alpaka_Kernels_h

#include <cstdint>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::test {

  void wrapper(Queue& queue,
               const float* __restrict__ in1,
               const float* __restrict__ in2,
               float* __restrict__ out,
               uint32_t size);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::test

#endif  // HeterogeneousTest_AlpakaPlugin_plugins_alpaka_Kernels_h
