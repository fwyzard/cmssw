#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigisHost.h"
#include "DataFormats/SiPixelDigi/interface/SiPixelDigisSoA.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

class SiPixelDigisSoAFromAlpaka : public edm::stream::EDProducer<> {
public:
  explicit SiPixelDigisSoAFromAlpaka(const edm::ParameterSet& iConfig);
  ~SiPixelDigisSoAFromAlpaka() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  edm::EDGetTokenT<SiPixelDigisHost> digiGetToken_;
  edm::EDPutTokenT<SiPixelDigisSoA> digiPutToken_;

  int nDigis_;
};

SiPixelDigisSoAFromAlpaka::SiPixelDigisSoAFromAlpaka(const edm::ParameterSet& iConfig)
    : digiGetToken_(consumes<SiPixelDigisHost>(iConfig.getParameter<edm::InputTag>("src"))),
      digiPutToken_(produces<SiPixelDigisSoA>()) {}

void SiPixelDigisSoAFromAlpaka::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("siPixelClustersCUDA"));
  descriptions.addWithDefaultLabel(desc);
}

void SiPixelDigisSoAFromAlpaka::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  const auto& digis_h = iEvent.get(digiGetToken_);
  iEvent.emplace(digiPutToken_,
                 digis_h.nDigis(),
                 digis_h.view().pdigi(),
                 digis_h.view().rawIdArr(),
                 digis_h.view().adc(),
                 digis_h.view().clus());
}

// define as framework plugin
DEFINE_FWK_MODULE(SiPixelDigisSoAFromAlpaka);