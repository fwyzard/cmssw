#ifndef DataFormats_RecHits_interface_TrackingRecHitSoADevice_h
#define DataFormats_RecHits_interface_TrackingRecHitSoADevice_h

#include <cstdint>
#include <alpaka/alpaka.hpp>
#include "DataFormats/Portable/interface/PortableDeviceCollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsLayout.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitSoAHost.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

template <typename TrackerTraits, typename TDev>
class TrackingRecHitAlpakaDevice : public PortableDeviceCollection<TrackingRecHitAlpakaLayout<TrackerTraits>, TDev> {
public:
  using hitSoA = TrackingRecHitAlpakaSoA<TrackerTraits>;
  //Need to decorate the class with the inherited portable accessors being now a template
  using PortableDeviceCollection<TrackingRecHitAlpakaLayout<TrackerTraits>, TDev>::view;
  using PortableDeviceCollection<TrackingRecHitAlpakaLayout<TrackerTraits>, TDev>::const_view;
  using PortableDeviceCollection<TrackingRecHitAlpakaLayout<TrackerTraits>, TDev>::buffer;

  TrackingRecHitAlpakaDevice() = default;

  using AverageGeometry = typename hitSoA::AverageGeometry;
  using ParamsOnDevice = typename hitSoA::ParamsOnDevice;
  using PhiBinnerStorageType = typename hitSoA::PhiBinnerStorageType;
  using PhiBinner = typename hitSoA::PhiBinner;
  // Constructor which specifies the SoA size
  template <typename TQueue>
  explicit TrackingRecHitAlpakaDevice(uint32_t nHits,
                                      int32_t offsetBPIX2,
                                      ParamsOnDevice const* cpeParams,
                                      uint32_t const* hitsModuleStart,
                                      TQueue queue)
      : PortableDeviceCollection<TrackingRecHitAlpakaLayout<TrackerTraits>, TDev>(nHits, queue),
        nHits_(nHits),
        cpeParams_(cpeParams),
        hitsModuleStart_(hitsModuleStart),
        offsetBPIX2_(offsetBPIX2) {
    phiBinner_ = &(view().phiBinner());

    const auto host = cms::alpakatools::host();
    const auto device = cms::alpakatools::devices<alpaka::Pltf<TDev>>()[0];

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

  uint32_t nHits() const { return nHits_; }  //go to size of view

  auto phiBinnerStorage() { return phiBinnerStorage_; }
  auto hitsModuleStart() const { return hitsModuleStart_; }
  uint32_t offsetBPIX2() const { return offsetBPIX2_; }
  auto phiBinner() { return phiBinner_; }

private:
  uint32_t nHits_;  //Needed for the host SoA size

  //TODO: this is used not that much from the hits (only once in BrokenLineFit), would make sens to remove it from this class.
  ParamsOnDevice const* cpeParams_;
  uint32_t const* hitsModuleStart_;
  uint32_t offsetBPIX2_;

  PhiBinnerStorageType* phiBinnerStorage_;
  PhiBinner* phiBinner_;
};
#endif  // DataFormats_RecHits_interface_TrackingRecHitSoADevice_h