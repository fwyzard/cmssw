#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// Tracking Particles
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"

// RecHits
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

// DigiSimLinks
#include "SimDataFormats/TrackerDigiSimLink/interface/PixelDigiSimLink.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/StripDigiSimLink.h"

#include <vector>
#include <map>
#include <iostream>

class SimToRecoHitProducer : public edm::one::EDProducer<> {
public:
    explicit SimToRecoHitProducer(const edm::ParameterSet& cfg)
    {
        tpToken_ = consumes<TrackingParticleCollection>(cfg.getParameter<edm::InputTag>("trackingParticles"));
        pixelSimLinkToken_ = consumes<edm::DetSetVector<PixelDigiSimLink>>(cfg.getParameter<edm::InputTag>("pixelDigiSimLink"));
        stripSimLinkToken_ = consumes<edm::DetSetVector<StripDigiSimLink>>(cfg.getParameter<edm::InputTag>("stripDigiSimLink"));
        recHitToken_ = consumes<TrackingRecHitCollection>(cfg.getParameter<edm::InputTag>("recoHits"));

        produces<edm::ValueMap<std::vector<edm::Ref<TrackingRecHitCollection>>>>();
    }

    ~SimToRecoHitProducer() override {}

    void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override
    {
        // ===== Input Handles =====
        edm::Handle<TrackingParticleCollection> tpH;
        iEvent.getByToken(tpToken_, tpH);

        edm::Handle<edm::DetSetVector<PixelDigiSimLink>> pixelLinksH;
        iEvent.getByToken(pixelSimLinkToken_, pixelLinksH);

        edm::Handle<edm::DetSetVector<StripDigiSimLink>> stripLinksH;
        iEvent.getByToken(stripSimLinkToken_, stripLinksH);

        edm::Handle<TrackingRecHitCollection> recHitH;
        iEvent.getByToken(recHitToken_, recHitH);

        // ===== Indicizziamo tutte le RecoHits per DetId =====
        std::multimap<uint32_t, size_t> recHitMap;  // detid -> indice nella collection
        for (size_t i = 0; i < recHitH->size(); ++i) {
            uint32_t detid = (*recHitH)[i].geographicalId().rawId();
            recHitMap.insert({detid, i});
        }

        // ===== Output: vettore per ogni TP =====
        std::vector<std::vector<edm::Ref<TrackingRecHitCollection>>> tp2hits(tpH->size());

        size_t tpIndex = 0;

        // =====================================================
        //                  LOOP SULLE TP
        // =====================================================
        for (auto const& tp : *tpH) {

            auto& hitsForTP = tp2hits[tpIndex];

            for (auto const& g4Track : tp.g4Tracks()) {
                unsigned int simTrackId = static_cast<unsigned int>(g4Track.trackId());

                // ---------- PIXEL ----------
                for (auto const& detset : *pixelLinksH) {
                    for (auto const& link : detset) {
                        if (link.SimTrackId() == simTrackId) {
                            uint32_t detid = detset.detId();
                            auto range = recHitMap.equal_range(detid);
                            for (auto it = range.first; it != range.second; ++it) {
                                edm::Ref<TrackingRecHitCollection> hitRef(recHitH, it->second);
                                hitsForTP.push_back(hitRef);
                            }
                        }
                    }
                }

                // ---------- STRIP ----------
                for (auto const& detset : *stripLinksH) {
                    for (auto const& link : detset) {
                        if (link.SimTrackId() == simTrackId) {
                            uint32_t detid = detset.detId();
                            auto range = recHitMap.equal_range(detid);
                            for (auto it = range.first; it != range.second; ++it) {
                                edm::Ref<TrackingRecHitCollection> hitRef(recHitH, it->second);
                                hitsForTP.push_back(hitRef);
                            }
                        }
                    }
                }

            }

            // Debug printing
            std::cout << "TP #" << tpIndex << " → " << hitsForTP.size() << " recoHits" << std::endl;

            for (auto hit : hitsForTP) {
                if (hit->isValid()) {
                    GlobalPoint gp = hit->globalPosition();
                    std::cout << "  Hit at (r=" << gp.perp() << ", z=" << gp.z() << ")" << std::endl;
                }
            }

            tpIndex++;
        }

        // ===== Riempimento ValueMap =====
        auto outputMap = std::make_unique<edm::ValueMap<std::vector<edm::Ref<TrackingRecHitCollection>>>>();
        edm::ValueMap<std::vector<edm::Ref<TrackingRecHitCollection>>>::Filler filler(*outputMap);
        filler.insert(tpH, tp2hits.begin(), tp2hits.end());
        filler.fill();

        iEvent.put(std::move(outputMap));
    }

private:
    edm::EDGetTokenT<TrackingParticleCollection> tpToken_;
    edm::EDGetTokenT<edm::DetSetVector<PixelDigiSimLink>> pixelSimLinkToken_;
    edm::EDGetTokenT<edm::DetSetVector<StripDigiSimLink>> stripSimLinkToken_;
    edm::EDGetTokenT<TrackingRecHitCollection> recHitToken_;
};

DEFINE_FWK_MODULE(SimToRecoHitProducer);

