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

  uint32_t nHits() const { return nHits_; }
  uint32_t offsetBPIX2() const { return offsetBPIX2_; }
  auto phiBinnerStorage() { return phiBinnerStorage_; }

private:
  uint32_t nHits_;  //Needed for the host SoA size
  ParamsOnDevice const* cpeParams_;
  uint32_t offsetBPIX2_;

  PhiBinnerStorageType* phiBinnerStorage_;
};

using TrackingRecHitAlpakaHostPhase1 = TrackingRecHitAlpakaHost<pixelTopology::Phase1>;
using TrackingRecHitAlpakaHostPhase2 = TrackingRecHitAlpakaHost<pixelTopology::Phase2>;

#endif  // CUDADataFormats_Track_TrackHeterogeneousT_H
