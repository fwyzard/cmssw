#ifndef DataFormats_BeamSpotSoA_BeamSpotDevice_h
#define DataFormats_BeamSpotSoA_BeamSpotDevice_h

#include <cstdint>
#include <alpaka/alpaka.hpp>
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "DataFormats/BeamSpotSoA/interface/BeamSpotHost.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/BeamSpotSoA/interface/BeamSpotLayout.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using BeamSpotDevice = PortableCollection<BeamSpotLayout<>>;
}


#endif  // DataFormats_BeamSpotSoA_BeamSpotDevice_h
