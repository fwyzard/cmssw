#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/TauReco/interface/PFTau.h"                        // reco::PFTau
#include "DataFormats/TauReco/interface/PFTauFwd.h"                     // reco::PFTauRef, reco::PFTauCollection
#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"           // reco::PFTauDiscriminator
#include "DataFormats/Phase2HLTPFTaus/interface/PFTauPair.h"            // reco::PFTauPair
#include "DataFormats/Phase2HLTPFTaus/interface/PFTauPairCollection.h"  // reco::PFTauPairCollection
#include "DataFormats/Common/interface/Handle.h"                        // edm::Handle
#include "DataFormats/Math/interface/deltaR.h"                          // reco::deltaR
#include <cmath>                                                        // std::fabs
#include <algorithm>                                                    // std::sort

class PFTauPairProducer : public edm::EDProducer {
public:
  explicit PFTauPairProducer(const edm::ParameterSet& cfg);
  ~PFTauPairProducer() = default;

private:
  void produce(edm::Event& evt, const edm::EventSetup& es);

  std::string moduleLabel_;

  edm::InputTag srcPFTaus_;
  edm::EDGetTokenT<reco::PFTauCollection> tokenPFTaus_;
  edm::InputTag srcPFTauSumChargedIso_;
  edm::EDGetTokenT<reco::PFTauDiscriminator> tokenPFTauSumChargedIso_;

  double min_pfTau_pt_;
  double max_pfTau_pt_;
  double min_pfTau_absEta_;
  double max_pfTau_absEta_;
  double min_pfTau_sumChargedIso_;
  double max_pfTau_sumChargedIso_;
  double min_dR_;
  double max_dR_;
  double min_dz_;
  double max_dz_;
};

PFTauPairProducer::PFTauPairProducer(const edm::ParameterSet& cfg)
    : moduleLabel_(cfg.getParameter<std::string>("@module_label")),
      min_pfTau_pt_(cfg.getParameter<double>("min_pfTau_pt")),
      max_pfTau_pt_(cfg.getParameter<double>("max_pfTau_pt")),
      min_pfTau_absEta_(cfg.getParameter<double>("min_pfTau_absEta")),
      max_pfTau_absEta_(cfg.getParameter<double>("max_pfTau_absEta")),
      min_pfTau_sumChargedIso_(cfg.getParameter<double>("min_pfTau_sumChargedIso")),
      max_pfTau_sumChargedIso_(cfg.getParameter<double>("max_pfTau_sumChargedIso")),
      min_dR_(cfg.getParameter<double>("min_dR")),
      max_dR_(cfg.getParameter<double>("max_dR")),
      min_dz_(cfg.getParameter<double>("min_dz")),
      max_dz_(cfg.getParameter<double>("max_dz")) {
  srcPFTaus_ = cfg.getParameter<edm::InputTag>("srcPFTaus");
  tokenPFTaus_ = consumes<reco::PFTauCollection>(srcPFTaus_);
  srcPFTauSumChargedIso_ = cfg.getParameter<edm::InputTag>("srcPFTauSumChargedIso");
  if (srcPFTauSumChargedIso_.label() != "") {
    tokenPFTauSumChargedIso_ = consumes<reco::PFTauDiscriminator>(srcPFTauSumChargedIso_);
  }

  produces<reco::PFTauPairCollection>();
}

namespace {
  bool isHigherPt(const std::pair<reco::PFTauRef, double>& pfTau_wChargedIso1,
                  const std::pair<reco::PFTauRef, double>& pfTau_wChargedIso2) {
    return pfTau_wChargedIso1.first->pt() > pfTau_wChargedIso2.first->pt();
  }
}  // namespace

void PFTauPairProducer::produce(edm::Event& evt, const edm::EventSetup& es) {
  auto pfTauPairs = std::make_unique<reco::PFTauPairCollection>();

  edm::Handle<reco::PFTauCollection> pfTaus;
  evt.getByToken(tokenPFTaus_, pfTaus);
  edm::Handle<reco::PFTauDiscriminator> pfTauSumChargedIso;
  if (srcPFTauSumChargedIso_.label() != "") {
    evt.getByToken(tokenPFTauSumChargedIso_, pfTauSumChargedIso);
  }

  std::vector<std::pair<reco::PFTauRef, double>> pfTaus_wChargedIso_sorted;
  size_t numPFTaus = pfTaus->size();
  for (size_t idxPFTau = 0; idxPFTau < numPFTaus; ++idxPFTau) {
    reco::PFTauRef pfTauRef(pfTaus, idxPFTau);
    double sumChargedIso = 0.;
    if (srcPFTauSumChargedIso_.label() != "") {
      sumChargedIso = (*pfTauSumChargedIso)[pfTauRef];
    }
    pfTaus_wChargedIso_sorted.push_back(std::pair<reco::PFTauRef, double>(pfTauRef, sumChargedIso));
  }
  std::sort(pfTaus_wChargedIso_sorted.begin(), pfTaus_wChargedIso_sorted.end(), isHigherPt);

  for (std::vector<std::pair<reco::PFTauRef, double>>::const_iterator leadPFTau_wChargedIso =
           pfTaus_wChargedIso_sorted.begin();
       leadPFTau_wChargedIso != pfTaus_wChargedIso_sorted.end();
       ++leadPFTau_wChargedIso) {
    if (!((min_pfTau_pt_ == -1. || leadPFTau_wChargedIso->first->pt() >= min_pfTau_pt_) &&
          (max_pfTau_pt_ == -1. || leadPFTau_wChargedIso->first->pt() <= max_pfTau_pt_) &&
          (min_pfTau_absEta_ == -1. || std::fabs(leadPFTau_wChargedIso->first->eta()) >= min_pfTau_absEta_) &&
          (max_pfTau_absEta_ == -1. || std::fabs(leadPFTau_wChargedIso->first->eta()) <= max_pfTau_absEta_) &&
          (min_pfTau_sumChargedIso_ == -1. || leadPFTau_wChargedIso->second >= min_pfTau_sumChargedIso_) &&
          (max_pfTau_sumChargedIso_ == -1. || leadPFTau_wChargedIso->second <= max_pfTau_sumChargedIso_)))
      continue;
    for (std::vector<std::pair<reco::PFTauRef, double>>::const_iterator subleadPFTau_wChargedIso =
             leadPFTau_wChargedIso + 1;
         subleadPFTau_wChargedIso != pfTaus_wChargedIso_sorted.end();
         ++subleadPFTau_wChargedIso) {
      if (!((min_pfTau_pt_ == -1. || subleadPFTau_wChargedIso->first->pt() >= min_pfTau_pt_) &&
            (max_pfTau_pt_ == -1. || subleadPFTau_wChargedIso->first->pt() <= max_pfTau_pt_) &&
            (min_pfTau_absEta_ == -1. || std::fabs(subleadPFTau_wChargedIso->first->eta()) >= min_pfTau_absEta_) &&
            (max_pfTau_absEta_ == -1. || std::fabs(subleadPFTau_wChargedIso->first->eta()) <= max_pfTau_absEta_) &&
            (min_pfTau_sumChargedIso_ == -1. || subleadPFTau_wChargedIso->second >= min_pfTau_sumChargedIso_) &&
            (max_pfTau_sumChargedIso_ == -1. || subleadPFTau_wChargedIso->second <= max_pfTau_sumChargedIso_)))
        continue;
      double dR = reco::deltaR(leadPFTau_wChargedIso->first->p4(), subleadPFTau_wChargedIso->first->p4());
      if (!((min_dR_ == -1. || dR >= min_dR_) && (max_dR_ == -1. || dR <= max_dR_)))
        continue;
      const reco::Track* leadPFTau_leadTrack = leadPFTau_wChargedIso->first->leadPFChargedHadrCand()->bestTrack();
      const reco::Track* subleadPFTau_leadTrack = subleadPFTau_wChargedIso->first->leadPFChargedHadrCand()->bestTrack();
      if (!(leadPFTau_leadTrack && subleadPFTau_leadTrack))
        continue;
      double dz = std::fabs(leadPFTau_leadTrack->vertex().z() - subleadPFTau_leadTrack->vertex().z());
      if (!((min_dz_ == -1. || dz >= min_dz_) && (max_dz_ == -1. || dz <= max_dz_)))
        continue;
      pfTauPairs->push_back(reco::PFTauPair(leadPFTau_wChargedIso->first,
                                            leadPFTau_wChargedIso->second,
                                            subleadPFTau_wChargedIso->first,
                                            subleadPFTau_wChargedIso->second,
                                            dz));
    }
  }

  evt.put(std::move(pfTauPairs));
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PFTauPairProducer);
