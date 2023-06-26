#ifndef DataFormats_SiPixelClusterSoA_interface_SiPixelClustersDevice_h
#define DataFormats_SiPixelClusterSoA_interface_SiPixelClustersDevice_h

#include <cstdint>
#include <alpaka/alpaka.hpp>
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "DataFormats/SiPixelClusterSoA/interface/SiPixelClustersHost.h"
#include "DataFormats/Portable/interface/PortableDeviceCollection.h"
#include "DataFormats/SiPixelClusterSoA/interface/SiPixelClustersLayout.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"

template <typename TDev>
class SiPixelClustersDevice : public PortableDeviceCollection<SiPixelClustersLayout<>, TDev> {
public:
  SiPixelClustersDevice() = default;
  ~SiPixelClustersDevice() = default;

  template <typename TQueue>
  explicit SiPixelClustersDevice(size_t maxModules, TQueue queue)
      : PortableDeviceCollection<SiPixelClustersLayout<>, TDev>(maxModules + 1, queue) {}

  // Constructor which specifies the SoA size
  explicit SiPixelClustersDevice(size_t maxModules, TDev const &device)
      : PortableDeviceCollection<SiPixelClustersLayout<>, TDev>(maxModules + 1, device) {}

  SiPixelClustersDevice(SiPixelClustersDevice &&) = default;
  SiPixelClustersDevice &operator=(SiPixelClustersDevice &&) = default;

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

#endif  // DataFormats_SiPixelClusterSoA_interface_SiPixelClustersDevice_h