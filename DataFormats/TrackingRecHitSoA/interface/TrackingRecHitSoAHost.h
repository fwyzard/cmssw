#ifndef DataFormats_RecHits_TrackingRecHitsHost_h
#define DataFormats_RecHits_TrackingRecHitsHost_h

#include <cstdint>
#include <alpaka/alpaka.hpp>

#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsLayout.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

template <typename TrackerTraits>
class TrackingRecHitAlpakaHost : public PortableHostCollection<TrackingRecHitAlpakaLayout<TrackerTraits>> {
public:
  using hitSoA = TrackingRecHitAlpakaSoA<TrackerTraits>;
  //Need to decorate the class with the inherited portable accessors being now a template
  using PortableHostCollection<TrackingRecHitAlpakaLayout<TrackerTraits>>::view;
  using PortableHostCollection<TrackingRecHitAlpakaLayout<TrackerTraits>>::const_view;
  using PortableHostCollection<TrackingRecHitAlpakaLayout<TrackerTraits>>::buffer;
  // using PortableHostCollection<TrackingRecHitAlpakaLayout<TrackerTraits>>::bufferSize;

  TrackingRecHitAlpakaHost() = default;

  using AverageGeometry = typename hitSoA::AverageGeometry;
  using ParamsOnDevice = typename hitSoA::ParamsOnDevice;
  using PhiBinnerStorageType = typename hitSoA::PhiBinnerStorageType;
  using PhiBinner = typename hitSoA::PhiBinner;

  // This SoA Host is used basically only for DQM
  // so we  just need a slim constructor
  explicit TrackingRecHitAlpakaHost(uint32_t nHits)
      : PortableHostCollection<TrackingRecHitAlpakaLayout<TrackerTraits>>(nHits) {}

  template <typename TQueue>
  explicit TrackingRecHitAlpakaHost(uint32_t nHits, TQueue queue)
      : PortableHostCollection<TrackingRecHitAlpakaLayout<TrackerTraits>>(nHits, queue) {}

  // Constructor which specifies the SoA size
  template <typename TQueue>
  explicit TrackingRecHitAlpakaHost(uint32_t nHits,
                                    int32_t offsetBPIX2,
                                    ParamsOnDevice const* cpeParams,
                                    uint32_t const* hitsModuleStart,
                                    TQueue queue)
      : PortableHostCollection<TrackingRecHitAlpakaLayout<TrackerTraits>>(nHits, queue),
        nHits_(nHits),
        cpeParams_(cpeParams),
        hitsModuleStart_(hitsModuleStart),
        offsetBPIX2_(offsetBPIX2) {
    phiBinner_ = &(view().phiBinner());

    const auto host = cms::alpakatools::host();
    const auto device = cms::alpakatools::devices<alpaka_common::PltfHost>()[0];

    auto cpe_h = alpaka::createView(host, cpeParams, 1);
    auto cpe_d = alpaka::createView(device, &(view().cpeParams()), 1);
    alpaka::memcpy(queue, cpe_d, cpe_h, 1);

    auto start_h = alpaka::createView(host, hitsModuleStart, TrackerTraits::numberOfModules + 1);
    auto start_d = alpaka::createView(device, view().hitsModuleStart().data(), TrackerTraits::numberOfModules + 1);
    alpaka::memcpy(queue, start_d, start_h);  //, TrackerTraits::numberOfModules + 1);

    auto nHits_h = alpaka::createView(host, &nHits, 1);
    auto nHits_d = alpaka::createView(device, &(view().nHits()), 1);
    alpaka::memcpy(queue, nHits_d, nHits_h, 1);

    auto off_h = alpaka::createView(host, &offsetBPIX2, 1);
    auto off_d = alpaka::createView(device, &(view().offsetBPIX2()), 1);
    alpaka::memcpy(queue, off_d, off_h, 1);
  }

  uint32_t nHits() const { return nHits_; }
  uint32_t offsetBPIX2() const { return offsetBPIX2_; }
  auto phiBinnerStorage() { return phiBinnerStorage_; }

private:
  uint32_t nHits_;  //Needed for the host SoA size
  ParamsOnDevice const* cpeParams_;
  uint32_t const* hitsModuleStart_;  // added from Device for new constructor
  uint32_t offsetBPIX2_;

  PhiBinnerStorageType* phiBinnerStorage_;
  PhiBinner* phiBinner_;  // added from Device for new constructor
};

using TrackingRecHitAlpakaHostPhase1 = TrackingRecHitAlpakaHost<pixelTopology::Phase1>;
using TrackingRecHitAlpakaHostPhase2 = TrackingRecHitAlpakaHost<pixelTopology::Phase2>;

#endif  // CUDADataFormats_Track_TrackHeterogeneousT_H