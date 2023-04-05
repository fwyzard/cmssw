#include <alpaka/alpaka.hpp>
#include "DataFormats/Portable/interface/Product.h"
#include "DataFormats/Track/interface/alpaka/TrackSoADevice.h"
#include "DataFormats/Track/interface/TrackSoAHost.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/RunningAverage.h"
#include "HeterogeneousCore/AlpakaCore/interface/ScopedContext.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ProducerBase.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoUtilities.h"

#include "CAHitNtupletGenerator.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  template <typename TrackerTraits>
  class CAHitNtupletAlpaka : public stream::EDProducer<> {
    using HitsConstView = TrackingRecHitAlpakaSoAConstView<TrackerTraits>;
    using HitsOnDevice = TrackingRecHitAlpakaDevice<TrackerTraits>;
    using HitsOnHost = TrackingRecHitAlpakaHost<TrackerTraits>;

    using TkSoAHost = TrackSoAHost<TrackerTraits>;
    using TkSoADevice = TrackSoADevice<TrackerTraits>;

    using Algo = CAHitNtupletGenerator<TrackerTraits>;

  public:
    explicit CAHitNtupletAlpaka(const edm::ParameterSet& iConfig);
    ~CAHitNtupletAlpaka() override = default;
    void produce(device::Event& iEvent, const device::EventSetup& es) override;
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> tokenField_;
    device::EDGetToken<HitsOnDevice> tokenHit_;
    device::EDPutToken<TkSoADevice> tokenTrack_;

    Algo deviceAlgo_;
  };

  template <typename TrackerTraits>
  CAHitNtupletAlpaka<TrackerTraits>::CAHitNtupletAlpaka(const edm::ParameterSet& iConfig)
      : tokenField_(esConsumes()), deviceAlgo_(iConfig, consumesCollector()) {
    tokenHit_ = consumes(iConfig.getParameter<edm::InputTag>("pixelRecHitSrc"));
    tokenTrack_ = produces();
  }

  template <typename TrackerTraits>
  void CAHitNtupletAlpaka<TrackerTraits>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;

    desc.add<edm::InputTag>("pixelRecHitSrc", edm::InputTag("siPixelRecHitsPreSplittingCUDA"));

    Algo::fillDescriptions(desc);
    descriptions.addWithDefaultLabel(desc);
  }

  template <typename TrackerTraits>
  void CAHitNtupletAlpaka<TrackerTraits>::produce(device::Event& iEvent, const device::EventSetup& es) {
    auto bf = 1. / es.getData(tokenField_).inverseBzAtOriginInGeV();

    auto const& hits = iEvent.get(tokenHit_);

    iEvent.emplace(tokenTrack_, deviceAlgo_.makeTuplesAsync(hits, bf, iEvent.queue()));
  }

  using CAHitNtupletAlpakaPhase1 = CAHitNtupletAlpaka<pixelTopology::Phase1>;
  using CAHitNtupletAlpakaPhase2 = CAHitNtupletAlpaka<pixelTopology::Phase2>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"

DEFINE_FWK_ALPAKA_MODULE(CAHitNtupletAlpakaPhase1);
DEFINE_FWK_ALPAKA_MODULE(CAHitNtupletAlpakaPhase2);
