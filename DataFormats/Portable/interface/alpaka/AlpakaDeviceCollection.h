#pragma once

#include "DataFormats/Portable/interface/AlpakaCollection.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // generic SoA-based product in device memory
  template <typename T>
  using AlpakaDeviceCollection = AlpakaCollection<T, Device>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
