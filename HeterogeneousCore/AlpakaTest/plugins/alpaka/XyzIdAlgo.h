#ifndef HeterogeneousCore_AlpakaTest_plugins_alpaka_XyzIdAlgo_h
#define HeterogeneousCore_AlpakaTest_plugins_alpaka_XyzIdAlgo_h

#include <alpaka/alpaka.hpp>

#include "DataFormats/XyzId/interface/alpaka/XyzIdDeviceCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

class XyzIdAlgoKernel {
public:
  template <typename TAcc>
  ALPAKA_FN_ACC void operator()(TAcc const& acc, int32_t* id, int32_t size) const {
    int32_t idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u];
    if (idx < size) {
      id[idx] = idx;
    }
  }
};

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class XyzIdAlgo {
  public:
    void fill(Queue& queue, XyzIdDeviceCollection& collection) const;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // HeterogeneousCore_AlpakaTest_plugins_alpaka_XyzIdAlgo_h
