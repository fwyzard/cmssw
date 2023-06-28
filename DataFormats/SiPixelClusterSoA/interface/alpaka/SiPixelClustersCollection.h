#ifndef DataFormats_SiPixelClusterSoA_interface_alpaka_SiPixelClustersCollection_h
#define DataFormats_SiPixelClusterSoA_interface_alpaka_SiPixelClustersCollection_h

#include <cstdint>
#include <alpaka/alpaka.hpp>
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "DataFormats/SiPixelClusterSoA/interface/SiPixelClustersHost.h"
#include "DataFormats/SiPixelClusterSoA/interface/SiPixelClustersDevice.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/SiPixelClusterSoA/interface/SiPixelClustersLayout.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"

// TODO: The class is created via inheritance of the PortableCollection.
// This is generally discouraged, and should be done via composition.
// See: https://github.com/cms-sw/cmssw/pull/40465#discussion_r1067364306
namespace ALPAKA_ACCELERATOR_NAMESPACE {
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
  using SiPixelClustersCollection = SiPixelClustersHost;
#else
  using SiPixelClustersCollection = SiPixelClustersDevice<Device>;

#endif
  using SiPixelClustersSoA = SiPixelClustersCollection;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

namespace cms::alpakatools {
  template <>
  struct CopyToHost<ALPAKA_ACCELERATOR_NAMESPACE::SiPixelClustersSoA> {
    template <typename TQueue>
    static auto copyAsync(TQueue &queue, ALPAKA_ACCELERATOR_NAMESPACE::SiPixelClustersSoA const &srcData) {
      SiPixelClustersHost dstData(srcData->metadata().size(), queue);
      alpaka::memcpy(queue, dstData.buffer(), srcData.buffer());
      dstData.setNClusters(srcData.nClusters(), srcData.offsetBPIX2());
      return dstData;
    }
  };
}  // namespace cms::alpakatools

#endif  // DataFormats_SiPixelClusterSoA_interface_alpaka_SiPixelClustersCollection_h