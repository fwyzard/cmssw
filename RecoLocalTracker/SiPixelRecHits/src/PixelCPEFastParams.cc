#include "FWCore/Utilities/interface/typelookup.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEFastParams.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"

using PixelCPEFastParamsPhase1 = pixelCPEforDevice::PixelCPEFastParams<alpaka_common::DevHost, pixelTopology::Phase1>;
using PixelCPEFastParamsPhase2 = pixelCPEforDevice::PixelCPEFastParams<alpaka_common::DevHost, pixelTopology::Phase2>;

TYPELOOKUP_DATA_REG(PixelCPEFastParamsPhase1);
TYPELOOKUP_DATA_REG(PixelCPEFastParamsPhase2);