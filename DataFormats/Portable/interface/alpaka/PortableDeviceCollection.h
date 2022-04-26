#ifndef DataFormats_Portable_interface_alpaka_PortableDeviceCollection_h
#define DataFormats_Portable_interface_alpaka_PortableDeviceCollection_h

#include "DataFormats/Portable/interface/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/alpaka/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // generic SoA-based product in device memory
  template <typename T>
  using PortableDeviceCollection = PortableCollection<T, Device>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // DataFormats_Portable_interface_alpaka_PortableDeviceCollection_h
