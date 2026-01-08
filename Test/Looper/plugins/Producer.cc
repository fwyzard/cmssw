#include <iostream>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

class Producer : public edm::global::EDProducer<> {
public:
  explicit Producer(const edm::ParameterSet& config) : value_{config.getParameter<int32_t>("value")} {}
  ~Producer() override = default;

  void produce(edm::StreamID sid, edm::Event&, const edm::EventSetup&) const override {
    std::cerr << "Producer::produce(" << sid << ", event, setup) --> " << value_ << '\n';
  }

private:
  int32_t value_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(Producer);
