#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitSoAHost.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitSoADevice.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/TrackingRecHitSoACollection.h"

#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"

#include <alpaka/alpaka.hpp>
#include <unistd.h>

using namespace ALPAKA_ACCELERATOR_NAMESPACE;

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace testTrackingRecHitSoA {

    template <typename TrackerTraits>
    void runKernels(TrackingRecHitAlpakaSoAView<TrackerTraits>& hits, Queue& queue);

  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

int main() {
  /*  const auto host = cms::alpakatools::host();
  const auto device = cms::alpakatools::devices<Platform>()[0];
  Queue queue(device);

  using ParamsOnDevice = TrackingRecHitAlpakaCollection<pixelTopology::Phase1>::ParamsOnDevice;
  // inner scope to deallocate memory before destroying the queue
  {
    uint32_t nHits = 2000;
    int32_t offset = 100;
    uint32_t moduleStart[1856];

    for (size_t i = 0; i < 1856; i++) {
      moduleStart[i] = i * 2;
    }
    auto cpeParams = std::make_unique<ParamsOnDevice>();
    TrackingRecHitAlpakaCollection<pixelTopology::Phase1> tkhit(nHits, offset, cpeParams.get(), &moduleStart[0], queue);

    testTrackingRecHitSoA::runKernels<pixelTopology::Phase1>(tkhit.view(), queue);
    alpaka::wait(queue);
  }
*/
  return 0;
}
