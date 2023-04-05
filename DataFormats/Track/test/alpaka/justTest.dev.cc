#include <alpaka/alpaka.hpp>
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "HeterogeneousCore/AlpakaUtilities/interface/HistoContainer.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/TrackingRecHitSoADevice.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsLayout.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitSoAHost.h"
#include "DataFormats/Track/interface/PixelTrackLayout.h"

using Tuples = typename TrackSoA<pixelTopology::Phase1>::HitContainer;
using TupleMultiplicity = cms::alpakatools::OneToManyAssoc<typename pixelTopology::Phase1::tindex_type,
                                                           pixelTopology::Phase1::maxHitsOnTrack + 1,
                                                           pixelTopology::Phase1::maxNumberOfTuples>;

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using namespace cms::alpakatools;
  template <int N>
  class kernel_testoi {  // TODO
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const &acc,
                                  Tuples const *foundNtuplets,
                                  TupleMultiplicity const *tupleMultiplicity,
                                  TrackingRecHitAlpakaSoAConstView<pixelTopology::Phase1> hh,
                                  typename pixelTopology::Phase1::tindex_type *ptkids,
                                  double *phits,
                                  float *phits_ge,
                                  double *pfast_fit,
                                  uint32_t nHitsL,
                                  uint32_t nHitsH,
                                  int32_t offset) const {
      constexpr uint32_t hitsInFit = N;
      printf("oi %d %d %d", hitsInFit, nHitsL, nHitsH);
      printf("oi %d", hh.nHits());
      ALPAKA_ASSERT_OFFLOAD(ptkids);
      ALPAKA_ASSERT_OFFLOAD(phits);
      ALPAKA_ASSERT_OFFLOAD(phits_ge);
      ALPAKA_ASSERT_OFFLOAD(pfast_fit);
      ALPAKA_ASSERT_OFFLOAD(foundNtuplets);
      ALPAKA_ASSERT_OFFLOAD(tupleMultiplicity);
    }
  };

  // Host function which invokes the two kernels above
  // template <typename TrackerTraits>
  void runKernels(Tuples const *foundNtuplets,
                  TupleMultiplicity const *tupleMultiplicity,
                  TrackingRecHitAlpakaSoAConstView<pixelTopology::Phase1> hh,
                  typename pixelTopology::Phase1::tindex_type *ptkids,
                  double *phits,
                  float *phits_ge,
                  double *pfast_fit,
                  uint32_t nHitsL,
                  uint32_t nHitsH,
                  int32_t offset,
                  Queue &queue) {
    uint32_t items = 64;
    uint32_t groups = divide_up_by(64, items);
    auto workDiv = make_workdiv<Acc1D>(groups, items);
    alpaka::exec<Acc1D>(queue,
                        workDiv,
                        kernel_testoi<3>{},
                        foundNtuplets,
                        tupleMultiplicity,
                        hh,
                        ptkids,
                        phits,
                        phits_ge,
                        pfast_fit,
                        nHitsL,
                        nHitsH,
                        offset);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
