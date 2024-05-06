#include <iostream>
#include <string>

#include "DataFormats/HcalRecHit/interface/HcalRecHitHostCollection.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDPutToken.h"

class HcalRecHitSoAToLegacy : public edm::stream::EDProducer<> {
public:
  explicit HcalRecHitSoAToLegacy(edm::ParameterSet const& ps);
  ~HcalRecHitSoAToLegacy() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void produce(edm::Event&, edm::EventSetup const&) override;

private:
  const bool produceLegacy_;

  using IProductType = hcal::RecHitHostCollection;
  const edm::EDGetTokenT<IProductType> recHitsM0TokenIn_;

  const edm::EDPutTokenT<HBHERecHitCollection> recHitsLegacyTokenOut_;
};

void HcalRecHitSoAToLegacy::fillDescriptions(edm::ConfigurationDescriptions& confDesc) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("recHitsM0LabelIn", edm::InputTag{"hbheRecHitProducerPortable"});
  desc.add<std::string>("recHitsLegacyLabelOut", "");
  desc.add<bool>("produceLegacy", true);

  confDesc.addWithDefaultLabel(desc);
}

HcalRecHitSoAToLegacy::HcalRecHitSoAToLegacy(const edm::ParameterSet& ps)
    : produceLegacy_{ps.getParameter<bool>("produceLegacy")},
      recHitsM0TokenIn_{consumes<IProductType>(ps.getParameter<edm::InputTag>("recHitsM0LabelIn"))},
      recHitsLegacyTokenOut_{produceLegacy_
                                 ? produces<HBHERecHitCollection>(ps.getParameter<std::string>("recHitsLegacyLabelOut"))
                                 : edm::EDPutTokenT<HBHERecHitCollection>{}}  // empty token if disabled
{}

void HcalRecHitSoAToLegacy::produce(edm::Event& event, edm::EventSetup const& setup) {
  if (produceLegacy_) {
    // populate the legacy collection
    auto recHitsLegacy = std::make_unique<HBHERecHitCollection>();

    // get input from host SoA
    auto const& hcalRechitSoAView = event.get(recHitsM0TokenIn_).const_view();

    recHitsLegacy->reserve(hcalRechitSoAView.metadata().size());

    for (auto i = 0; i < hcalRechitSoAView.metadata().size(); i++) {
      // skip bad channels
      if (hcalRechitSoAView.chi2()[i] < 0)
        continue;

      // build a legacy rechit with the computed detid and MAHI energy
      recHitsLegacy->emplace_back(HcalDetId{hcalRechitSoAView.did()[i]},
                                  hcalRechitSoAView.energy()[i],
                                  0  // timeRising
      );
      // update the legacy rechit with the Chi2 and M0 values
      recHitsLegacy->back().setChiSquared(hcalRechitSoAView.chi2()[i]);
      recHitsLegacy->back().setRawEnergy(hcalRechitSoAView.energyM0()[i]);
    }

    // put the legacy collection
    event.put(recHitsLegacyTokenOut_, std::move(recHitsLegacy));
  }

  // clear the temporary collection for the next event
  //tmpRecHits_.resize(0);
}

DEFINE_FWK_MODULE(HcalRecHitSoAToLegacy);
