#ifndef HeterogeneousCore_AlpakaCore_interface_alpaka_device_h
#define HeterogeneousCore_AlpakaCore_interface_alpaka_device_h

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // alpaka accelerator device
  inline const Device device = alpaka::getDevByIdx<Platform>(0u);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // HeterogeneousCore_AlpakaCore_interface_alpaka_device_h
