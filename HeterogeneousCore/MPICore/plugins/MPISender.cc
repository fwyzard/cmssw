#include <string>

#include "DataFormats/TestObjects/interface/ThingCollection.h"
using namespace edmtest;

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "HeterogeneousCore/MPICore/interface/MPIToken.h"

#include "api.h"

class MPISender : public edm::global::EDProducer<> {
public:
  MPISender(edm::ParameterSet const& config)
      : mpiPrev_(consumes<MPIToken>(config.getParameter<edm::InputTag>("channel"))),
        mpiNext_(produces<MPIToken>()),
        data_(consumes<ThingCollection>(config.getParameter<edm::InputTag>("data"))),
        tag_(config.getParameter<int32_t>("tag")) {}

  void produce(edm::StreamID sid, edm::Event& event, edm::EventSetup const&) const override {
    // read the MPIToken used to establish the communication channel
    MPIToken token = event.get(mpiPrev_);

    // read the data to be sent over the MPI channel
    auto data = event.get(data_);

    // send the data over MPI
    // note: currently this uses a blocking send
    token.channel()->sendSerializedProduct(sid, data);

    // write a shallow copy of the channel to the output, so other modules can consume it
    // to indicate that they should run after this
    event.emplace(mpiNext_, token);
  }

private:
  edm::EDGetTokenT<MPIToken> const mpiPrev_;  // MPIToken used to establish the communication channel
  edm::EDPutTokenT<MPIToken> const mpiNext_;  // copy of the MPIToken that may be used to implement an ordering relation
  edm::EDGetTokenT<ThingCollection> const data_;  // data to be read from the Event and sent over the MPI channel
  int32_t const tag_;                             // MPI tag used to identify the destination
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(MPISender);
