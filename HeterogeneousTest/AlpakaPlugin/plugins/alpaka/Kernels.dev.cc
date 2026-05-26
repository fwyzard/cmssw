#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "Kernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::test {

  struct Kernel {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  const float* __restrict__ in1,
                                  const float* __restrict__ in2,
                                  float* __restrict__ out,
                                  uint32_t size) const {
      for (auto i : cms::alpakatools::uniform_elements(acc, size)) {
        out[i] = in1[i] + in2[i];
      }
    }
  };

  void wrapper(Queue& queue,
               const float* __restrict__ in1,
               const float* __restrict__ in2,
               float* __restrict__ out,
               uint32_t size) {
    alpaka::exec<Acc1D>(queue, cms::alpakatools::make_workdiv<Acc1D>(32, 32), Kernel{}, in1, in2, out, size);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::test
