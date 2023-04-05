#ifndef RecoPixelVertexing_PixelTriplets_Alpaka_CAHitNtupletGenerator_h
#define RecoPixelVertexing_PixelTriplets_Alpaka_CAHitNtupletGenerator_h

#include <alpaka/alpaka.hpp>

#include "DataFormats/Track/interface/alpaka/TrackSoADevice.h"
#include "DataFormats/Track/interface/TrackSoAHost.h"
#include "DataFormats/Track/interface/PixelTrackDefinitions.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/TrackingRecHitSoADevice.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitSoAHost.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsLayout.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "HeterogeneousCore/AlpakaUtilities/interface/SimpleVector.h"

#include "CAHitNtupletGeneratorKernels.h"
#include "CACell.h"
#include "HelixFit.h"

namespace edm {
  class Event;
  class EventSetup;
  class ParameterSetDescription;
}  // namespace edm

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  template <typename TrackerTraits>
  class CAHitNtupletGenerator {
  public:
    using HitsView = TrackingRecHitAlpakaSoAView<TrackerTraits>;
    using HitsConstView = TrackingRecHitAlpakaSoAConstView<TrackerTraits>;
    using HitsOnDevice = TrackingRecHitAlpakaDevice<TrackerTraits>;
    using HitsOnHost = TrackingRecHitAlpakaHost<TrackerTraits>;
    using hindex_type = typename TrackingRecHitAlpakaSoA<TrackerTraits>::hindex_type;

    using HitToTuple = caStructures::HitToTupleT<TrackerTraits>;
    using TupleMultiplicity = caStructures::TupleMultiplicityT<TrackerTraits>;
    using OuterHitOfCell = caStructures::OuterHitOfCellT<TrackerTraits>;

    using CACell = CACellT<TrackerTraits>;
    using TkSoAHost = TrackSoAHost<TrackerTraits>;
    using TkSoADevice = TrackSoADevice<TrackerTraits>;
    using HitContainer = typename TrackSoA<TrackerTraits>::HitContainer;
    using Tuple = HitContainer;

    using CellNeighborsVector = caStructures::CellNeighborsVectorT<TrackerTraits>;
    using CellTracksVector = caStructures::CellTracksVectorT<TrackerTraits>;

    using Quality = ::pixelTrack::Quality;

    using QualityCuts = ::pixelTrack::QualityCutsT<TrackerTraits>;
    using Params = caHitNtupletGenerator::ParamsT<TrackerTraits>;
    using Counters = caHitNtupletGenerator::Counters;

  public:
    CAHitNtupletGenerator(const edm::ParameterSet& cfg, edm::ConsumesCollector&& iC) : CAHitNtupletGenerator(cfg, iC){};
    CAHitNtupletGenerator(const edm::ParameterSet& cfg, edm::ConsumesCollector& iC);

    static void fillDescriptions(edm::ParameterSetDescription& desc);
    static void fillDescriptionsCommon(edm::ParameterSetDescription& desc);

    // TODO: Check if still needed
    // void beginJob();
    // void endJob();

    TkSoADevice makeTuplesAsync(HitsOnDevice const& hits_d, float bfield, Queue& queue) const;

  private:
    void buildDoublets(const HitsConstView& hh, Queue& queue) const;

    void hitNtuplets(const HitsConstView& hh, const edm::EventSetup& es, bool useRiemannFit, Queue& queue);

    void launchKernels(const HitsConstView& hh, bool useRiemannFit, Queue& queue) const;

    Params m_params;

    // Counters* m_counters = nullptr;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoPixelVertexing_PixelTriplets_plugins_CAHitNtupletGenerator_h
