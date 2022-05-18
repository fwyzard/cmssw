#ifndef DataFormats_XyzId_interface_alpaka_XyzIdDeviceCollection_h
#define DataFormats_XyzId_interface_alpaka_XyzIdDeviceCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/XyzId/interface/XyzIdSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // SoA with x, y, z, id fields in device global memory
  using XyzIdDeviceCollection = PortableCollection<XyzIdSoA>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // DataFormats_XyzId_interface_alpaka_XyzIdDeviceCollection_h
