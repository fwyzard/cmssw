#ifndef RecoLocalTracker_SiPixelClusterizer_interface_SiPixelImageSoA_h
#define RecoLocalTracker_SiPixelClusterizer_interface_SiPixelImageSoA_h

#include "DataFormats/SoATemplate/interface/SoALayout.h"

GENERATE_SOA_LAYOUT(SiPixelImageLayout, SOA_COLUMN(uint32_t, id))

using SiPixelImageSoA = SiPixelImageLayout<>;
using SiPixelImageSoAView = SiPixelImageSoA::View;
using SiPixelImageSoAConstView = SiPixelImageSoA::ConstView;

#endif  // RecoLocalTracker_SiPixelClusterizer_interface_SiPixelImageSoA_h
