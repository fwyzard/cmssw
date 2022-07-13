#ifndef CUDADataFormats_SiPixelCluster_interface_SiPixelClustersCUDA_h
#define CUDADataFormats_SiPixelCluster_interface_SiPixelClustersCUDA_h

#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCompat.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/Portable/interface/PortableCUDADeviceCollection.h"

#include <cuda_runtime.h>

GENERATE_SOA_LAYOUT(SiPixelClustersCUDATemplate,
                    SOA_COLUMN(uint32_t, moduleStart),
                    SOA_COLUMN(uint32_t, clusInModule),
                    SOA_COLUMN(uint32_t, moduleId),
                    SOA_COLUMN(uint32_t, clusModuleStart))

// Decorating the Portable collection to add previously existing functions
class SiPixelClustersCUDA : public PortableCUDADeviceCollection<SiPixelClustersCUDATemplate<>> {
public:
  SiPixelClustersCUDA() = default;
  explicit SiPixelClustersCUDA(size_t maxModules, cudaStream_t stream)
      : PortableCUDADeviceCollection<SiPixelClustersCUDATemplate<>>(maxModules + 1, stream) {}

  // movable
  SiPixelClustersCUDA(SiPixelClustersCUDA &&) = default;
  SiPixelClustersCUDA &operator=(SiPixelClustersCUDA &&) = default;

  ~SiPixelClustersCUDA() = default;

  // Restrict view
  using RestrictConstView =
      Layout::ConstViewTemplate<cms::soa::RestrictQualify::Enabled, cms::soa::RangeChecking::Disabled>;

  RestrictConstView restrictConstView() const { return RestrictConstView(layout()); }
  void setNClusters(uint32_t nClusters, int32_t offsetBPIX2) {
    nClusters_h = nClusters;
    offsetBPIX2_h = offsetBPIX2;
  }

  uint32_t nClusters() const { return nClusters_h; }
  int32_t offsetBPIX2() const { return offsetBPIX2_h; }

private:
  uint32_t nClusters_h = 0;
  int32_t offsetBPIX2_h = 0;
};

#endif  // CUDADataFormats_SiPixelCluster_interface_SiPixelClustersCUDA_h