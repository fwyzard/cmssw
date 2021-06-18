#ifndef L1Trigger_Phase2HLTPFTaus_MyPATTauSelector_h
#define L1Trigger_Phase2HLTPFTaus_MyPATTauSelector_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/PatCandidates/interface/Tau.h" // pat::Tau, pat::TauCollection

#include <vector>
#include <string>

class MyPATTauSelector : public edm::EDProducer 
{
 public:
  explicit MyPATTauSelector(const edm::ParameterSet& cfg);
  ~MyPATTauSelector();

 private:
  void produce(edm::Event& evt, const edm::EventSetup& es);

  edm::InputTag src_;
  edm::EDGetTokenT<pat::TauCollection> token_;

  double min_pt_;
  double max_pt_;
  double min_absEta_;
  double max_absEta_;
  std::vector<int> decayModes_;
  double min_leadTrackPt_;
  double max_leadTrackPt_;
  std::string tauID_relChargedIso_;
  double min_relChargedIso_;
  double max_relChargedIso_;
  double min_absChargedIso_;
  double max_absChargedIso_;

  bool invert_;
};

#endif
