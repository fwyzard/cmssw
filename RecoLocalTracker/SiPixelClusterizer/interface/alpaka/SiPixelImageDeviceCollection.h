#ifndef RecoLocalTracker_SiPixelClusterizer_interface_alpaka_SiPixelImageDeviceCollection_h
#define RecoLocalTracker_SiPixelClusterizer_interface_alpaka_SiPixelImageDeviceCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "RecoLocalTracker/SiPixelClusterizer/interface/SiPixelImageHostCollection.h"
#include "RecoLocalTracker/SiPixelClusterizer/interface/SiPixelImageSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using ::SiPixelImageHostCollection;
  using SiPixelImageDeviceCollection = PortableCollection<SiPixelImageSoA>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

ASSERT_DEVICE_MATCHES_HOST_COLLECTION(SiPixelImageDeviceCollection, SiPixelImageHostCollection);

#endif  // RecoLocalTracker_SiPixelClusterizer_interface_alpaka_SiPixelImageDeviceCollection_h
