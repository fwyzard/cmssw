#ifndef DataFormats_BeamSpotSoA_BeamSpotHost_h
#define DataFormats_BeamSpotSoA_BeamSpotHost_h

#include <alpaka/alpaka.hpp>
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"

#include "BeamSpotLayout.h"

using BeamSpotHost = PortableHostCollection<BeamSpotLayout<>>;

#endif  // DataFormats_BeamSpotSoA_BeamSpotHost_h
