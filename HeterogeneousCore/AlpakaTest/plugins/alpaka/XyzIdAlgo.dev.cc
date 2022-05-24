// Check that ALPAKA_HOST_ONLY is not defined during device compilation:
#ifdef ALPAKA_HOST_ONLY
#error ALPAKA_HOST_ONLY defined in device compilation
#endif

#include <alpaka/alpaka.hpp>

#include "DataFormats/XyzId/interface/alpaka/XyzIdDeviceCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "XyzIdAlgo.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  void XyzIdAlgo::fill(Queue& queue, XyzIdDeviceCollection& collection) const {
    auto const& deviceProperties = alpaka::getAccDevProps<Acc1D>(alpaka::getDev(queue));
    uint32_t maxThreadsPerBlock = deviceProperties.m_blockThreadExtentMax[0];

    uint32_t threadsPerBlock = maxThreadsPerBlock;
    uint32_t blocksPerGrid = (collection->soaMetadata().size() + threadsPerBlock - 1) / threadsPerBlock;
    uint32_t elementsPerThread = 1;
    auto workDiv = WorkDiv1D{blocksPerGrid, threadsPerBlock, elementsPerThread};

    alpaka::exec<Acc1D>(queue, workDiv, XyzIdAlgoKernel{}, &collection->id(0), collection->soaMetadata().size());
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
