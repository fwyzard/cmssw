#ifndef DataFormats_Track_TrackSoADevice_H
#define DataFormats_Track_TrackSoADevice_H

#include <cstdint>
#include <alpaka/alpaka.hpp>
// #include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "DataFormats/Track/interface/PixelTrackLayout.h"
#include "DataFormats/Track/interface/PixelTrackDefinitions.h"
#include "DataFormats/Track/interface/TrackSoAHost.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"

// TODO: The class is created via inheritance of the PortableCollection.
// This is generally discouraged, and should be done via composition.
// See: https://github.com/cms-sw/cmssw/pull/40465#discussion_r1067364306
namespace ALPAKA_ACCELERATOR_NAMESPACE {
  template <typename TrackerTraits>
  class TrackSoADevice : public PortableCollection<TrackLayout<TrackerTraits>> {
  public:
    static constexpr int32_t S = TrackerTraits::maxNumberOfTuples;  //TODO: this could be made configurable at runtime
    //explicit TrackSoADevice() : PortableCollection<TrackLayout<TrackerTraits>>(S) {} //TODO: check if this is needed somewhere
    TrackSoADevice() = default;  // cms::alpakatools::Product needs this

    using PortableCollection<TrackLayout<TrackerTraits>>::view;
    using PortableCollection<TrackLayout<TrackerTraits>>::const_view;
    using PortableCollection<TrackLayout<TrackerTraits>>::buffer;

    // Constructor which specifies the SoA size
    template <typename TQueue>
    explicit TrackSoADevice<TrackerTraits>(TQueue queue) : PortableCollection<TrackLayout<TrackerTraits>>(S, queue) {}

    // Constructor which specifies the SoA size
    explicit TrackSoADevice(Device const& device) : PortableCollection<TrackLayout<TrackerTraits>>(S, device) {}
  };

  namespace pixelTrack {

    using TrackSoADevicePhase1 = TrackSoADevice<pixelTopology::Phase1>;
    using TrackSoADevicePhase2 = TrackSoADevice<pixelTopology::Phase2>;

  }  // namespace pixelTrack

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

namespace cms::alpakatools {
  // TODO: Is this the right place for the specialization? Or should it be in PortableDeviceProduct?
  template <>
  struct CopyToHost<ALPAKA_ACCELERATOR_NAMESPACE::pixelTrack::TrackSoADevicePhase1> {
    template <typename TQueue>
    static auto copyAsync(TQueue& queue,
                          ALPAKA_ACCELERATOR_NAMESPACE::pixelTrack::TrackSoADevicePhase1 const& deviceData) {
      ::pixelTrack::TrackSoAHostPhase1 hostData(queue);
      alpaka::memcpy(queue, hostData.buffer(), deviceData.buffer());
      return hostData;
    }
  };

  template <>
  struct CopyToHost<ALPAKA_ACCELERATOR_NAMESPACE::pixelTrack::TrackSoADevicePhase2> {
    template <typename TQueue>
    static auto copyAsync(TQueue& queue,
                          ALPAKA_ACCELERATOR_NAMESPACE::pixelTrack::TrackSoADevicePhase2 const& deviceData) {
      ::pixelTrack::TrackSoAHostPhase2 hostData(queue);
      alpaka::memcpy(queue, hostData.buffer(), deviceData.buffer());
      return hostData;
    }
  };
}  // namespace cms::alpakatools

#endif  // DataFormats_Track_TrackSoADevice_H
