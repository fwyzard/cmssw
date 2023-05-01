#ifndef DataFormats_SiPixelClusterSoA_SiPixelClustersLayout_h
#define DataFormats_SiPixelClusterSoA_SiPixelClustersLayout_h

#include "DataFormats/SoATemplate/interface/SoALayout.h"

GENERATE_SOA_LAYOUT(BeamSpotLayout,
                    SOA_COLUMN(float, x),
                    SOA_COLUMN(float, y),
                    SOA_COLUMN(float, z),
                    SOA_COLUMN(float, sigmaZ),
                    SOA_COLUMN(float, beamWidthX),
                    SOA_COLUMN(float, beamWidthY),
                    SOA_COLUMN(float, dxdz),
                    SOA_COLUMN(float, dydz),
                    SOA_COLUMN(float, emittanceX),
                    SOA_COLUMN(float, emittanceY),
                    SOA_COLUMN(float, betaStar))

using BeamSpotLayoutSoA = BeamSpotLayout<>;
using BeamSpotLayoutSoAView = BeamSpotLayout<>::View;
using BeamSpotLayoutSoAConstView = BeamSpotLayout<>::ConstView;

#endif  // DataFormats_BeamSpotoA_BeamSpotLayout_h
