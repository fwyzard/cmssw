#pragma once

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // alpaka accelerator device
  inline const Device device = alpaka::getDevByIdx<Platform>(0u);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
