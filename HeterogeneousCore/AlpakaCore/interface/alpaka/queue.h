#pragma once

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/config.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/device.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // alpaka queue
  inline Queue queue{device};

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
