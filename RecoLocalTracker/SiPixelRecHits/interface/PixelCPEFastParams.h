#ifndef DataFormats_PixelCPEFastParams_interface_PixelCPEFastParams_h
#define DataFormats_PixelCPEFastParams_interface_PixelCPEFastParams_h

#include <alpaka/alpaka.hpp>
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToDevice.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "DataFormats/SiPixelClusterSoA/interface/ClusteringConstants.h"
#include "DataFormats/GeometrySurface/interface/SOARotation.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "DataFormats/TrackingRecHitSoA/interface/SiPixelHitStatus.h"

// Nesting this namespace to prevent conflicts with pixelCPEForDevice.h
namespace CPEFastParametrisation {
  // From https://cmssdt.cern.ch/dxr/CMSSW/source/CondFormats/SiPixelTransient/src/SiPixelGenError.cc#485-486
  // qbin: int (0-4) describing the charge of the cluster
  // [0: 1.5<Q/Qavg, 1: 1<Q/Qavg<1.5, 2: 0.85<Q/Qavg<1, 3: 0.95Qmin<Q<0.85Qavg, 4: Q<0.95Qmin]
  constexpr int kGenErrorQBins = 5;
  // arbitrary number of bins for sampling errors
  constexpr int kNumErrorBins = 16;
}  // namespace CPEFastParametrisation

// Mostly similar to pixelCPEforDevice namespace,
// to prevent conflicts until we completely replace it.
// See pixelCPEForDevice.h.
namespace pixelCPEforDevice {

  using Status = SiPixelHitStatus;
  using Frame = SOAFrame<float>;
  using Rotation = SOARotation<float>;

  // SOA (on device)

  template <uint32_t N>
  struct ClusParamsT {
    uint32_t minRow[N];
    uint32_t maxRow[N];
    uint32_t minCol[N];
    uint32_t maxCol[N];

    int32_t q_f_X[N];
    int32_t q_l_X[N];
    int32_t q_f_Y[N];
    int32_t q_l_Y[N];

    int32_t charge[N];

    float xpos[N];
    float ypos[N];

    float xerr[N];
    float yerr[N];

    int16_t xsize[N];  // (*8) clipped at 127 if negative is edge....
    int16_t ysize[N];

    Status status[N];
  };

  // all modules are identical!
  struct CommonParams {
    float theThicknessB;
    float theThicknessE;
    float thePitchX;
    float thePitchY;

    uint16_t maxModuleStride;
    uint8_t numberOfLaddersInBarrel;
  };

  struct DetParams {
    bool isBarrel;
    bool isPosZ;
    uint16_t layer;
    uint16_t index;
    uint32_t rawId;

    float shiftX;
    float shiftY;
    float chargeWidthX;
    float chargeWidthY;
    uint16_t pixmx;  // max pix charge

    uint16_t nRowsRoc;  //we don't need 2^16 columns, is worth to use 15 + 1 for sign
    uint16_t nColsRoc;
    uint16_t nRows;
    uint16_t nCols;

    uint32_t numPixsInModule;

    float x0, y0, z0;  // the vertex in the local coord of the detector

    float apeXX, apeYY;  // ape^2
    uint8_t sx2, sy1, sy2;
    uint8_t sigmax[CPEFastParametrisation::kNumErrorBins], sigmax1[CPEFastParametrisation::kNumErrorBins],
        sigmay[CPEFastParametrisation::kNumErrorBins];  // in micron
    float xfact[CPEFastParametrisation::kGenErrorQBins], yfact[CPEFastParametrisation::kGenErrorQBins];
    int minCh[CPEFastParametrisation::kGenErrorQBins];

    Frame frame;
  };

  template <typename TrackerTopology>
  struct LayerGeometryT {
    uint32_t layerStart[TrackerTopology::numberOfLayers + 1];
    uint8_t layer[pixelTopology::layerIndexSize<TrackerTopology>];
    uint16_t maxModuleStride;
  };

  // ES Product to be used by RecHit generation
  template <typename TrackerTopology>
  struct ParamsOnDeviceT {
    using LayerGeometry = LayerGeometryT<TrackerTopology>;
    using AverageGeometry = pixelTopology::AverageGeometryT<TrackerTopology>;

    CommonParams m_commonParams;
    // Will contain an array of DetParams instances
    DetParams m_detParams[1440];
    LayerGeometry m_layerGeometry;
    AverageGeometry m_averageGeometry;

    constexpr CommonParams const& __restrict__ commonParams() const { return m_commonParams; }
    constexpr DetParams const& __restrict__ detParams(int i) const { return m_detParams[i]; }
    constexpr LayerGeometry const& __restrict__ layerGeometry() const { return m_layerGeometry; }
    constexpr AverageGeometry const& __restrict__ averageGeometry() const { return m_averageGeometry; }

    ALPAKA_FN_ACC uint8_t layer(uint16_t id) const {
      return m_layerGeometry.layer[id / TrackerTopology::maxModuleStride];
    };
  };

  template <typename TDev, typename TrackerTraits>
  class PixelCPEFastParams {
  public:
    using Buffer = cms::alpakatools::device_buffer<TDev, ParamsOnDeviceT<TrackerTraits>>;
    using ConstBuffer = cms::alpakatools::const_device_buffer<TDev, ParamsOnDeviceT<TrackerTraits>>;

    explicit PixelCPEFastParams(Buffer buffer) : buffer_(std::move(buffer)) {}

    Buffer buffer() { return buffer_; }
    ConstBuffer buffer() const { return buffer_; }
    ConstBuffer const_buffer() const { return buffer_; }
    ParamsOnDeviceT<TrackerTraits> const* data() const { return buffer_.data(); }
    auto size() const { return alpaka::getExtentProduct(buffer_); }

  private:
    Buffer buffer_;
  };

}  // namespace pixelCPEforDevice

namespace cms::alpakatools {
  template <>
  struct CopyToDevice<pixelCPEforDevice::PixelCPEFastParams<alpaka_common::DevHost, pixelTopology::Phase1>> {
    template <typename TQueue>
    static auto copyAsync(
        TQueue& queue, pixelCPEforDevice::PixelCPEFastParams<alpaka_common::DevHost, pixelTopology::Phase1> hostData) {
      using ParamsOnDevice = pixelCPEforDevice::ParamsOnDeviceT<pixelTopology::Phase1>;
      auto deviceData =
          cms::alpakatools::make_device_buffer<pixelCPEforDevice::ParamsOnDeviceT<pixelTopology::Phase1>>(queue);
      alpaka::memcpy(queue, deviceData, hostData.buffer());
      return deviceData;
    }
  };

  template <>
  struct CopyToDevice<pixelCPEforDevice::PixelCPEFastParams<alpaka_common::DevHost, pixelTopology::Phase2>> {
    template <typename TQueue>
    static auto copyAsync(
        TQueue& queue, pixelCPEforDevice::PixelCPEFastParams<alpaka_common::DevHost, pixelTopology::Phase2> hostData) {
      using ParamsOnDevice = pixelCPEforDevice::ParamsOnDeviceT<pixelTopology::Phase1>;
      auto deviceData =
          cms::alpakatools::make_device_buffer<pixelCPEforDevice::ParamsOnDeviceT<pixelTopology::Phase2>>(queue);
      alpaka::memcpy(queue, deviceData, hostData.buffer());
      return deviceData;
    }
  };

}  // namespace cms::alpakatools

#endif
