#ifndef DataFormats_BeamSpotSoA_BeamSpotLayout_h
#define DataFormats_BeamSpotSoA_BeamSpotLayout_h

#include "DataFormats/SoATemplate/interface/SoALayout.h"

GENERATE_SOA_LAYOUT(BeamSpotLayout,
                    SOA_SCALAR(float, x),
                    SOA_SCALAR(float, y),
                    SOA_SCALAR(float, z),
                    SOA_SCALAR(float, sigmaZ),
                    SOA_SCALAR(float, beamWidthX),
                    SOA_SCALAR(float, beamWidthY),
                    SOA_SCALAR(float, dxdz),
                    SOA_SCALAR(float, dydz),
                    SOA_SCALAR(float, emittanceX),
                    SOA_SCALAR(float, emittanceY),
                    SOA_COLUMN(float, betaStar))

using BeamSpotLayoutSoA = BeamSpotLayout<>;
using BeamSpotLayoutSoAView = BeamSpotLayout<>::View;
using BeamSpotLayoutSoAConstView = BeamSpotLayout<>::ConstView;

#endif  // DataFormats_BeamSpotSoA_BeamSpotLayout_h
