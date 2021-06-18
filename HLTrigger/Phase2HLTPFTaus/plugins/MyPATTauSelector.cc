#include "HLTrigger/Phase2HLTPFTaus/plugins/MyPATTauSelector.h"

#include "FWCore/Utilities/interface/InputTag.h"

MyPATTauSelector::MyPATTauSelector(const edm::ParameterSet& cfg) 
{
  src_ = cfg.getParameter<edm::InputTag>("src");
  token_ = consumes<pat::TauCollection>(src_);

  min_pt_ = cfg.getParameter<double>("min_pt");
  max_pt_ = cfg.getParameter<double>("max_pt");
  min_absEta_ = cfg.getParameter<double>("min_absEta");
  max_absEta_ = cfg.getParameter<double>("max_absEta");
  decayModes_ = cfg.getParameter<std::vector<int>>("decayModes");
  min_leadTrackPt_ = cfg.getParameter<double>("min_leadTrackPt");
  max_leadTrackPt_ = cfg.getParameter<double>("max_leadTrackPt");
  tauID_relChargedIso_ = cfg.getParameter<std::string>("tauID_relChargedIso");
  min_relChargedIso_ = cfg.getParameter<double>("min_relChargedIso");
  max_relChargedIso_ = cfg.getParameter<double>("max_relChargedIso");
  min_absChargedIso_ = cfg.getParameter<double>("min_absChargedIso");
  max_absChargedIso_ = cfg.getParameter<double>("max_absChargedIso");

  invert_ = cfg.getParameter<bool>("invert");

  produces<pat::TauCollection>("");
}

MyPATTauSelector::~MyPATTauSelector()
{}

void MyPATTauSelector::produce(edm::Event& evt, const edm::EventSetup& es)
{
  std::unique_ptr<pat::TauCollection> pfTaus_selected(new pat::TauCollection());

  edm::Handle<pat::TauCollection> pfTaus;
  evt.getByToken(token_, pfTaus);
  
  size_t numPFTaus = pfTaus->size();
  for ( size_t idxPFTau = 0; idxPFTau < numPFTaus; ++idxPFTau ) 
  {  
    const pat::Tau& pfTau = pfTaus->at(idxPFTau);
    double pfTau_absEta = std::fabs(pfTau.eta());
    double sumChargedIso = pfTau.tauID(tauID_relChargedIso_);
    bool isSelected = false;
    if ( (min_pt_            < 0. || pfTau.pt()                          >=  min_pt_                       ) &&
         (max_pt_            < 0. || pfTau.pt()                          <=  max_pt_                       ) &&
         (min_absEta_        < 0. || pfTau_absEta                        >=  min_absEta_                   ) &&
         (max_absEta_        < 0. || pfTau_absEta                        <=  max_absEta_                   ) &&
         (                           pfTau.leadPFChargedHadrCand().isNonnull()                             && 
                                     pfTau.leadPFChargedHadrCand()->bestTrack()                            ) &&
         (min_leadTrackPt_   < 0. || pfTau.leadPFChargedHadrCand()->pt() >=  min_leadTrackPt_              ) &&
         (max_leadTrackPt_   < 0. || pfTau.leadPFChargedHadrCand()->pt() <=  max_leadTrackPt_              ) &&
         (min_relChargedIso_ < 0. || sumChargedIso                       >= (min_relChargedIso_*pfTau.pt())) &&
         (max_relChargedIso_ < 0. || sumChargedIso                       <= (max_relChargedIso_*pfTau.pt())) &&
         (min_absChargedIso_ < 0. || sumChargedIso                       >=  min_absChargedIso_            ) &&
	 (max_absChargedIso_ < 0. || sumChargedIso                       <=  max_absChargedIso_            ) )
    {
      if ( decayModes_.size() >= 1 )
      {
        for ( std::vector<int>::const_iterator decayMode = decayModes_.begin();
              decayMode != decayModes_.end(); ++decayMode ) {
          if ( pfTau.decayMode() == (*decayMode) ) 
          {
            isSelected = true;
            break;
          }
        }
      }
      else 
      {
        isSelected = true;
      }
    }
    if ( (isSelected && !invert_) || (!isSelected && invert_) ) 
    {
      pfTaus_selected->push_back(pfTau);
    }
  }

  evt.put(std::move(pfTaus_selected));
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(MyPATTauSelector);
