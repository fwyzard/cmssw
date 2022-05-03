#ifndef DataFormats_Portable_interface_alpaka_AlpakaDeviceCollection_h
#define DataFormats_Portable_interface_alpaka_AlpakaDeviceCollection_h

#include "DataFormats/Portable/interface/AlpakaCollection.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // generic SoA-based product in device memory
  template <typename T>
  using AlpakaDeviceCollection = AlpakaCollection<T, Device>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // DataFormats_Portable_interface_alpaka_AlpakaDeviceCollection_h
