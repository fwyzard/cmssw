#ifndef DataFormats_RecHits_interface_alpaka_TrackingRecHitSoACollection_h
#define DataFormats_RecHits_interface_alpaka_TrackingRecHitSoACollection_h

#include <cstdint>
#include <alpaka/alpaka.hpp>
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsLayout.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitSoAHost.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitSoADevice.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
  template <typename TrackerTraits>
  using TrackingRecHitAlpakaCollection = TrackingRecHitAlpakaHost<TrackerTraits>;
#else
  template <typename TrackerTraits>
  using TrackingRecHitAlpakaCollection = TrackingRecHitAlpakaDevice<TrackerTraits, Device>;
#endif
  //Classes definition for Phase1/Phase2, to make the classes_def lighter. Not actually used in the code.
  using TrackingRecHitAlpakaSoAPhase1 = TrackingRecHitAlpakaCollection<pixelTopology::Phase1>;
  using TrackingRecHitAlpakaSoAPhase2 = TrackingRecHitAlpakaCollection<pixelTopology::Phase2>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

namespace cms::alpakatools {
  template <>
  struct CopyToHost<ALPAKA_ACCELERATOR_NAMESPACE::TrackingRecHitAlpakaSoAPhase1> {
    template <typename TQueue>
    static auto copyAsync(TQueue& queue,
                          ALPAKA_ACCELERATOR_NAMESPACE::TrackingRecHitAlpakaSoAPhase1 const& deviceData) {
      TrackingRecHitAlpakaHostPhase1 hostData(deviceData.view().metadata().size(), queue);
      alpaka::memcpy(queue, hostData.buffer(), deviceData.buffer());
      return hostData;
    }
  };

  template <>
  struct CopyToHost<ALPAKA_ACCELERATOR_NAMESPACE::TrackingRecHitAlpakaSoAPhase2> {
    template <typename TQueue>
    static auto copyAsync(TQueue& queue,
                          ALPAKA_ACCELERATOR_NAMESPACE::TrackingRecHitAlpakaSoAPhase2 const& deviceData) {
      TrackingRecHitAlpakaHostPhase2 hostData(deviceData.view().metadata().size(), queue);
      alpaka::memcpy(queue, hostData.buffer(), deviceData.buffer());
      return hostData;
    }
  };
}  // namespace cms::alpakatools

#endif  // DataFormats_RecHits_interface_alpaka_TrackingRecHitSoACollection_h