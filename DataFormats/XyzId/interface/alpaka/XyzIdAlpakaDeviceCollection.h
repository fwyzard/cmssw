#ifndef DataFormats_XyzId_interface_alpaka_XyzIdAlpakaDeviceCollection_h
#define DataFormats_XyzId_interface_alpaka_XyzIdAlpakaDeviceCollection_h

#include "DataFormats/Portable/interface/alpaka/AlpakaDeviceCollection.h"
#include "DataFormats/XyzId/interface/XyzIdSoA.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // SoA with x, y, z, id fields in device global memory
  using XyzIdAlpakaDeviceCollection = AlpakaDeviceCollection<XyzIdSoA>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // DataFormats_XyzId_interface_alpaka_XyzIdAlpakaDeviceCollection_h
