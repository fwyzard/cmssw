#ifndef CondFormats_SiPixelObjects_SiPixelGainCalibrationForHLTLayout_h
#define CondFormats_SiPixelObjects_SiPixelGainCalibrationForHLTLayout_h

#include <array>
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"

namespace siPixelGainsSoA {
  struct DecodingStructure {
    uint8_t gain;
    uint8_t ped;
  };

  using Range = std::pair<uint32_t, uint32_t>;
  using RangeAndCols = std::array<std::pair<Range, int>, phase1PixelTopology::numberOfModules>;

}  // namespace siPixelGainsSoA

GENERATE_SOA_LAYOUT(SiPixelGainCalibrationForHLTLayout,
                    SOA_COLUMN(siPixelGainsSoA::DecodingStructure, v_pedestals),
                    SOA_SCALAR(siPixelGainsSoA::RangeAndCols, rangeAndCols),

                    SOA_SCALAR(float, minPed),
                    SOA_SCALAR(float, maxPed),
                    SOA_SCALAR(float, minGain),
                    SOA_SCALAR(float, maxGain),
                    SOA_SCALAR(float, pedPrecision),
                    SOA_SCALAR(float, gainPrecision),

                    SOA_SCALAR(unsigned int, numberOfRowsAveragedOver),
                    SOA_SCALAR(unsigned int, nBinsToUseForEncoding),
                    SOA_SCALAR(unsigned int, deadFlag),
                    SOA_SCALAR(unsigned int, noisyFlag),
                    SOA_SCALAR(float, link))

using SiPixelGainCalibrationForHLTSoA = SiPixelGainCalibrationForHLTLayout<>;
using SiPixelGainCalibrationForHLTSoAView = SiPixelGainCalibrationForHLTLayout<>::View;
using SiPixelGainCalibrationForHLTSoAConstView = SiPixelGainCalibrationForHLTLayout<>::ConstView;

#endif  // CondFormats_SiPixelObjects_SiPixelGainCalibrationForHLTLayout_h
