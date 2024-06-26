#ifndef HLTJetL1MatchProducer_h
#define HLTJetL1MatchProducer_h

#include <string>
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/JetReco/interface/TrackJetCollection.h"
#include "DataFormats/JetReco/interface/BasicJetCollection.h"

template <typename T>
class HLTJetL1MatchProducer : public edm::stream::EDProducer<> {
public:
  explicit HLTJetL1MatchProducer(const edm::ParameterSet &);
  ~HLTJetL1MatchProducer() override;
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);
  virtual void beginJob();
  void produce(edm::Event &, const edm::EventSetup &) override;

private:
  edm::EDGetTokenT<std::vector<T>> m_theJetToken;
  edm::EDGetTokenT<l1extra::L1JetParticleCollection> m_theL1TauJetToken;
  edm::EDGetTokenT<l1extra::L1JetParticleCollection> m_theL1CenJetToken;
  edm::EDGetTokenT<l1extra::L1JetParticleCollection> m_theL1ForJetToken;
  edm::InputTag jetsInput_;
  edm::InputTag L1TauJets_;
  edm::InputTag L1CenJets_;
  edm::InputTag L1ForJets_;
  double DeltaR2_;  // DeltaR2(HLT,L1) with sign
};

#endif
