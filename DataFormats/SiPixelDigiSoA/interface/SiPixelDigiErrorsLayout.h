#ifndef DataFormats_SiPixelDigi_SiPixelDigiErrorsLayout_h
#define DataFormats_SiPixelDigi_SiPixelDigiErrorsLayout_h

#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SiPixelRawData/interface/SiPixelErrorCompact.h"
#include "HeterogeneousCore/AlpakaInterface/interface/SimpleVector.h"

using SiPixelErrorCompactVec = cms::alpakatools::SimpleVector<SiPixelErrorCompact>;

GENERATE_SOA_LAYOUT(SiPixelDigiErrorsLayout,
                    SOA_COLUMN(SiPixelErrorCompact, pixelErrors),
                    SOA_SCALAR(SiPixelErrorCompactVec, pixelErrorsVec))

using SiPixelDigiErrorsLayoutSoA = SiPixelDigiErrorsLayout<>;
using SiPixelDigiErrorsLayoutSoAView = SiPixelDigiErrorsLayout<>::View;
using SiPixelDigiErrorsLayoutSoAConstView = SiPixelDigiErrorsLayout<>::ConstView;

#endif  // DataFormats_SiPixelDigi_SiPixelDigisErrorLayout_h
