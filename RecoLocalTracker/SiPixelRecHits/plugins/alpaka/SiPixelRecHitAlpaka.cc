#include "DataFormats/BeamSpotSoA/interface/alpaka/BeamSpotDevice.h"
#include "DataFormats/SiPixelClusterSoA/interface/alpaka/SiPixelClustersCollection.h"
#include "DataFormats/SiPixelClusterSoA/interface/SiPixelClustersDevice.h"
#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigisDevice.h"
#include "DataFormats/SiPixelDigiSoA/interface/alpaka/SiPixelDigisCollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitSoADevice.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/TrackingRecHitSoACollection.h"
#include "DataFormats/Common/interface/Handle.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEFastParams.h"

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/global/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "RecoLocalTracker/Records/interface/TkPixelCPERecord.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEBase.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEFastAlpaka.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEFastParams.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/pixelCPEforDevice.h"

#include "PixelRecHitGPUKernel.h"
namespace ALPAKA_ACCELERATOR_NAMESPACE {
  template <typename TrackerTraits>
  class SiPixelRecHitAlpaka : public global::EDProducer<> {
  public:
    explicit SiPixelRecHitAlpaka(const edm::ParameterSet& iConfig);
    ~SiPixelRecHitAlpaka() override = default;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

    using ParamsOnGPU = pixelCPEforDevice::ParamsOnDeviceT<TrackerTraits>;

  private:
    void produce(edm::StreamID streamID, device::Event& iEvent, const device::EventSetup& iSetup) const override;

    const device::ESGetToken<pixelCPEforDevice::PixelCPEFastParams<Device, TrackerTraits>, TkPixelCPERecord> cpeToken_;
    const device::EDGetToken<BeamSpotDevice> tBeamSpot;
    const device::EDGetToken<SiPixelClustersSoA> tokenClusters_;
    const device::EDGetToken<SiPixelDigisSoA> tokenDigi_;
    const device::EDPutToken<TrackingRecHitAlpakaCollection<TrackerTraits>> tokenHit_;

    const pixelgpudetails::PixelRecHitGPUKernel<TrackerTraits> gpuAlgo_;
  };

  template <typename TrackerTraits>
  SiPixelRecHitAlpaka<TrackerTraits>::SiPixelRecHitAlpaka(const edm::ParameterSet& iConfig)
      : cpeToken_(esConsumes(edm::ESInputTag("", iConfig.getParameter<std::string>("CPE")))),
        tBeamSpot(consumes(iConfig.getParameter<edm::InputTag>("beamSpot"))),
        tokenClusters_(consumes(iConfig.getParameter<edm::InputTag>("src"))),
        tokenDigi_(consumes(iConfig.getParameter<edm::InputTag>("src"))),
        tokenHit_(produces()) {}

  template <typename TrackerTraits>
  void SiPixelRecHitAlpaka<TrackerTraits>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;

    desc.add<edm::InputTag>("beamSpot", edm::InputTag("offlineBeamSpotDevice"));
    desc.add<edm::InputTag>("src", edm::InputTag("siPixelClustersPreSplittingAlpaka"));

    std::string cpe = "PixelCPEFast";
    cpe += TrackerTraits::nameModifier;
    desc.add<std::string>("CPE", cpe);

    descriptions.addWithDefaultLabel(desc);
  }

  template <typename TrackerTraits>
  void SiPixelRecHitAlpaka<TrackerTraits>::produce(edm::StreamID streamID,
                                                   device::Event& iEvent,
                                                   const device::EventSetup& es) const {
    auto& fcpe = es.getData(cpeToken_);
    if (not fcpe.data()) {
      throw cms::Exception("Configuration") << "SiPixelRecHitAlpaka can only use a CPE of type PixelCPEFast";
    }

    auto const& clusters = iEvent.get(tokenClusters_);

    auto const& digis = iEvent.get(tokenDigi_);

    auto const& bs = iEvent.get(tBeamSpot);

    iEvent.emplace(tokenHit_, gpuAlgo_.makeHitsAsync(digis, clusters, bs, fcpe.data(), iEvent.queue()));
  }
  using SiPixelRecHitAlpakaPhase1 = SiPixelRecHitAlpaka<pixelTopology::Phase1>;
  using SiPixelRecHitAlpakaPhase2 = SiPixelRecHitAlpaka<pixelTopology::Phase2>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(SiPixelRecHitAlpakaPhase1);
DEFINE_FWK_ALPAKA_MODULE(SiPixelRecHitAlpakaPhase2);
