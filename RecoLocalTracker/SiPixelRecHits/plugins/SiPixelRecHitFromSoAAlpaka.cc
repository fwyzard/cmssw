#include <alpaka/alpaka.hpp>

#include <fmt/printf.h>

#include "DataFormats/Portable/interface/HostProductAlpaka.h"
#include "DataFormats/Portable/interface/Product.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitSoAHost.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsLayout.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

#include "RecoLocalTracker/SiPixelRecHits/interface/pixelCPEforDevice.h"

template <typename TrackerTraits>
class SiPixelRecHitFromSoAAlpaka : public edm::global::EDProducer<> {
  using HitModuleStartArray = typename TrackingRecHitAlpakaSoA<TrackerTraits>::HitModuleStartArray;
  using hindex_type = typename TrackerTraits::hindex_type;

public:
  explicit SiPixelRecHitFromSoAAlpaka(const edm::ParameterSet& iConfig);
  ~SiPixelRecHitFromSoAAlpaka() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  using HMSstorage = HostProductAlpaka<uint32_t[]>;

  // Data has been implicitly copied from Device to Host by the framework
  using HitsOnHost = TrackingRecHitAlpakaHost<TrackerTraits>;

private:
  void produce(edm::StreamID streamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  const edm::EDGetTokenT<HitsOnHost> hitsToken_;                      // Alpaka hits
  const edm::EDGetTokenT<SiPixelClusterCollectionNew> clusterToken_;  // legacy clusters
  const edm::EDPutTokenT<SiPixelRecHitCollection> rechitsPutToken_;   // legacy rechits
  const edm::EDPutTokenT<HMSstorage> hostPutToken_;
};

template <typename TrackerTraits>
SiPixelRecHitFromSoAAlpaka<TrackerTraits>::SiPixelRecHitFromSoAAlpaka(const edm::ParameterSet& iConfig)
    : geomToken_(esConsumes()),
      hitsToken_(consumes(iConfig.getParameter<edm::InputTag>("pixelRecHitSrc"))),
      clusterToken_(consumes(iConfig.getParameter<edm::InputTag>("src"))),
      rechitsPutToken_(produces()),
      hostPutToken_(produces()) {}

template <typename TrackerTraits>
void SiPixelRecHitFromSoAAlpaka<TrackerTraits>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("pixelRecHitSrc", edm::InputTag("siPixelRecHitsPreSplittingAlpaka"));
  desc.add<edm::InputTag>("src", edm::InputTag("siPixelClustersPreSplitting"));

  descriptions.addWithDefaultLabel(desc);
}

template <typename TrackerTraits>
void SiPixelRecHitFromSoAAlpaka<TrackerTraits>::produce(edm::StreamID streamID,
                                                        edm::Event& iEvent,
                                                        const edm::EventSetup& iSetup) const {
  auto& hits_h_ = iEvent.get(hitsToken_);
  auto nHits = hits_h_.view().nHits();
  LogDebug("SiPixelRecHitFromSoAAlpaka") << "converting " << nHits << " Hits";

  // allocate a buffer for the indices of the clusters
  constexpr auto nMaxModules = TrackerTraits::numberOfModules;
  auto hmsp = cms::alpakatools::make_host_buffer<hindex_type[]>(nMaxModules + 1);

  SiPixelRecHitCollection output;
  output.reserve(nMaxModules, nHits);

  if (0 == nHits) {
    iEvent.emplace(rechitsPutToken_, std::move(output));
    iEvent.emplace(hostPutToken_, std::move(hmsp));
    return;
  }
  output.reserve(nMaxModules, nHits);

  // Could not make a view on hitsModuleStart
  //alpaka::memcpy(iEvent.queue(), hmsp, hitsModuleStartView);

  // std::copy not really working, cannot d
  //std::copy(hits_h_.view().hitsModuleStart(), &hits_h_.view().hitsModuleStart() + nMaxModules + 1, hmsp.data());
  std::memcpy((void*)hmsp.data(), (void*)&hits_h_.view().hitsModuleStart(), sizeof(HitModuleStartArray));

  // wrap the buffer in a HostProduct, and move it to the Event, without reallocating the buffer or affecting hitsModuleStart
  iEvent.emplace(hostPutToken_, std::move(hmsp));

  auto xl = hits_h_.view().xLocal();
  auto yl = hits_h_.view().yLocal();
  auto xe = hits_h_.view().xerrLocal();
  auto ye = hits_h_.view().yerrLocal();

  const TrackerGeometry* geom = &iSetup.getData(geomToken_);

  edm::Handle<SiPixelClusterCollectionNew> hclusters = iEvent.getHandle(clusterToken_);
  auto const& input = *hclusters;

  constexpr uint32_t maxHitsInModule = pixelClustering::maxHitsInModule();

  int numberOfDetUnits = 0;
  int numberOfClusters = 0;
  for (auto const& dsv : input) {
    numberOfDetUnits++;
    unsigned int detid = dsv.detId();
    DetId detIdObject(detid);
    const GeomDetUnit* genericDet = geom->idToDetUnit(detIdObject);
    auto gind = genericDet->index();
    const PixelGeomDetUnit* pixDet = dynamic_cast<const PixelGeomDetUnit*>(genericDet);
    assert(pixDet);
    SiPixelRecHitCollection::FastFiller recHitsOnDetUnit(output, detid);
    auto fc = hits_h_.view().hitsModuleStart()[gind];
    auto lc = hits_h_.view().hitsModuleStart()[gind + 1];
    auto nhits = lc - fc;

    assert(lc > fc);
    LogDebug("SiPixelRecHitFromSoAAlpaka") << "in det " << gind << ": conv " << nhits << " hits from " << dsv.size()
                                           << " legacy clusters" << ' ' << fc << ',' << lc << "\n";
    if (nhits > maxHitsInModule)
      edm::LogWarning("SiPixelRecHitFromSoAAlpaka") << fmt::sprintf(
          "Too many clusters %d in module %d. Only the first %d hits will be converted", nhits, gind, maxHitsInModule);
    nhits = std::min(nhits, maxHitsInModule);

    LogDebug("SiPixelRecHitFromSoAAlpaka") << "in det " << gind << "conv " << nhits << " hits from " << dsv.size()
                                           << " legacy clusters" << ' ' << lc << ',' << fc;

    if (0 == nhits)
      continue;
    auto jnd = [&](int k) { return fc + k; };
    assert(nhits <= dsv.size());
    if (nhits != dsv.size()) {
      edm::LogWarning("GPUHits2CPU") << "nhits!= nclus " << nhits << ' ' << dsv.size();
    }
    for (auto const& clust : dsv) {
      assert(clust.originalId() >= 0);
      assert(clust.originalId() < dsv.size());
      if (clust.originalId() >= nhits)
        continue;
      auto ij = jnd(clust.originalId());
      LocalPoint lp(xl[ij], yl[ij]);
      LocalError le(xe[ij], 0, ye[ij]);
      SiPixelRecHitQuality::QualWordType rqw = 0;

      numberOfClusters++;

      /* cpu version....  (for reference)
    std::tuple<LocalPoint, LocalError, SiPixelRecHitQuality::QualWordType> tuple = cpe_->getParameters( clust, *genericDet );
    LocalPoint lp( std::get<0>(tuple) );
    LocalError le( std::get<1>(tuple) );
    SiPixelRecHitQuality::QualWordType rqw( std::get<2>(tuple) );
    */

      // Create a persistent edm::Ref to the cluster
      edm::Ref<edmNew::DetSetVector<SiPixelCluster>, SiPixelCluster> cluster = edmNew::makeRefTo(hclusters, &clust);
      // Make a RecHit and add it to the DetSet
      recHitsOnDetUnit.emplace_back(lp, le, rqw, *genericDet, cluster);
      // =============================

      LogDebug("SiPixelRecHitFromSoAAlpaka") << "cluster " << numberOfClusters << " at " << lp << ' ' << le;

    }  //  <-- End loop on Clusters

    //  LogDebug("SiPixelRecHitGPU")
    LogDebug("SiPixelRecHitFromSoAAlpaka") << "found " << recHitsOnDetUnit.size() << " RecHits on " << detid;

  }  //    <-- End loop on DetUnits

  LogDebug("SiPixelRecHitFromSoAAlpaka") << "found " << numberOfDetUnits << " dets, " << numberOfClusters
                                         << " clusters";

  iEvent.emplace(rechitsPutToken_, std::move(output));
}

using SiPixelRecHitFromSoAAlpakaPhase1 = SiPixelRecHitFromSoAAlpaka<pixelTopology::Phase1>;
DEFINE_FWK_MODULE(SiPixelRecHitFromSoAAlpakaPhase1);

using SiPixelRecHitFromSoAAlpakaPhase2 = SiPixelRecHitFromSoAAlpaka<pixelTopology::Phase2>;
DEFINE_FWK_MODULE(SiPixelRecHitFromSoAAlpakaPhase2);
