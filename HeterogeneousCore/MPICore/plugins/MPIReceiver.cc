#include <string>

#include <TBufferFile.h>
#include <TClass.h>

#include "DataFormats/TestObjects/interface/ThingCollection.h"
using namespace edmtest;

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "HeterogeneousCore/MPICore/interface/MPIToken.h"

#include "api.h"

class MPIReceiver : public edm::global::EDProducer<> {
public:
  MPIReceiver(edm::ParameterSet const& config)
      : mpiPrev_(consumes<MPIToken>(config.getParameter<edm::InputTag>("channel"))),
        mpiNext_(produces<MPIToken>()),
        data_(produces<ThingCollection>()),
        tag_(config.getParameter<int32_t>("tag")) {}

  void produce(edm::StreamID sid, edm::Event& event, edm::EventSetup const&) const override {
    // read the MPIToken used to establish the communication channel
    MPIToken token = event.get(mpiPrev_);

    // receive the data to sent over the MPI channel
    // note: currently this uses a blocking probe/recv
    MPI_Message message;
    MPI_Status status;
    int source = token.channel()->rank();
    auto comm = token.channel()->comm();
    ThingCollection data;

    MPI_Mprobe(source, MPI_ANY_TAG, comm, &message, &status);
    {
      // receive the product
      assert(EDM_MPI_SendSerializedProduct == status.MPI_TAG);
      int size;
      MPI_Get_count(&status, MPI_BYTE, &size);
      TBufferFile buffer{TBuffer::kRead, size};
      MPI_Mrecv(buffer.Buffer(), size, MPI_BYTE, &message, &status);

      static TClass* type = TClass::GetClass<ThingCollection>();
      type->ReadBuffer(buffer, &data);
    }
    event.emplace(data_, data);

    // write a shallow copy of the channel to the output, so other modules can consume it
    // to indicate that they should run after this
    event.emplace(mpiNext_, token);
  }

private:
  edm::EDGetTokenT<MPIToken> const mpiPrev_;  // MPIToken used to establish the communication channel
  edm::EDPutTokenT<MPIToken> const mpiNext_;  // copy of the MPIToken that may be used to implement an ordering relation
  edm::EDPutTokenT<ThingCollection> const data_;  // data to be read over the MPI channel and put into the Event
  int32_t const tag_;                             // MPI tag used to identify the destination
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(MPIReceiver);
