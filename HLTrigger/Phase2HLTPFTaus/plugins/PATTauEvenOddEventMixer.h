#ifndef HLTrigger_Phase2HLTPFTaus_GenVertexProducer_h
#define HLTrigger_Phase2HLTPFTaus_GenVertexProducer_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/PatCandidates/interface/Tau.h"

#include <vector>

class PATTauEvenOddEventMixer : public edm::EDProducer 
{
 public:
  explicit PATTauEvenOddEventMixer(const edm::ParameterSet& cfg);
  ~PATTauEvenOddEventMixer();

 private:
  void produce(edm::Event& evt, const edm::EventSetup& es);

  std::string moduleLabel_;

  edm::InputTag srcEvenEvents_;
  edm::EDGetTokenT<pat::TauCollection> tokenEvenEvents_;
  edm::InputTag srcOddEvents_;
  edm::EDGetTokenT<pat::TauCollection> tokenOddEvents_;

  long numEvenEvents_;
  long numEvenTaus_;
  long numOddEvents_;
  long numOddTaus_;
};

#endif
