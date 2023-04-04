#ifndef DataFormats_SiPixelDigi_SiPixelDigisLayout_h
#define DataFormats_SiPixelDigi_SiPixelDigisLayout_h

#include "DataFormats/SoATemplate/interface/SoALayout.h"

GENERATE_SOA_LAYOUT(SiPixelDigisLayout,
                    SOA_COLUMN(int32_t, clus),
                    SOA_COLUMN(uint32_t, pdigi),
                    SOA_COLUMN(uint32_t, rawIdArr),
                    SOA_COLUMN(uint16_t, adc),
                    SOA_COLUMN(uint16_t, xx),
                    SOA_COLUMN(uint16_t, yy),
                    SOA_COLUMN(uint16_t, moduleId))

using SiPixelDigisLayoutSoA = SiPixelDigisLayout<>;
using SiPixelDigisLayoutSoAView = SiPixelDigisLayout<>::View;
using SiPixelDigisLayoutSoAConstView = SiPixelDigisLayout<>::ConstView;

#endif  // DataFormats_SiPixelDigi_SiPixelDigisLayout_h
