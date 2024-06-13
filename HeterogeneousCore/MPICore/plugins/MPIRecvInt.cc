// -*- C++ -*-
//
// Package:    HeterogeneousCore/MPICore
// Class:      MPIRecvInt
//
/**\class MPIRecvInt MPIRecvInt.cc HeterogeneousCore/MPICore/plugins/MPIRecvInt.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Fawaz M Sh Kh W Alazemi
//         Created:  Tue, 28 May 2024 11:13:32 GMT
//
//

// system include files
#include <memory>
#include <mpi.h>
#include <future>
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "HeterogeneousCore/MPICore/interface/MPICommunicator.h"

//
// class declaration
//

class MPIRecvInt : public edm::stream::EDProducer<edm::ExternalWork> {
public:
  explicit MPIRecvInt(const edm::ParameterSet&);
  ~MPIRecvInt() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  void acquire(edm::Event const&, edm::EventSetup const&, edm::WaitingTaskWithArenaHolder) override;

private:
  void beginStream(edm::StreamID) override;
  void produce(edm::Event&, const edm::EventSetup&) override;
  void endStream() override;
  void asyncTask(edm::WaitingTaskWithArenaHolder, MPI_Comm, int);
  //void beginRun(edm::Run const&, edm::EventSetup const&) override;
  //void endRun(edm::Run const&, edm::EventSetup const&) override;
  //void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
  //void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

  // ----------member data ---------------------------
  edm::EDGetTokenT<MPIToken> communicatorToken_;
  edm::EDGetTokenT<int> tagIDToken_;
  int userTagID_;
  edm::EDPutTokenT<int> outIntData_;
  edm::StreamID sid_ = edm::StreamID::invalidStreamID();
  int recvIntData_;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
MPIRecvInt::MPIRecvInt(const edm::ParameterSet& iConfig)
    : communicatorToken_{consumes(iConfig.getParameter<edm::InputTag>("communicator"))},
      tagIDToken_{consumes(iConfig.getParameter<edm::InputTag>("tagID"))},
      userTagID_{iConfig.getUntrackedParameter<int>("userTagID")},
      outIntData_{produces()} {
  //register your products
  /* Examples
  produces<ExampleData2>();

  //if do put with a label
  produces<ExampleData2>("label");
 
  //if you want to put into the Run
  produces<ExampleData2,InRun>();
  */
  //now do what ever other initialization is needed
}

MPIRecvInt::~MPIRecvInt() {
  // do anything here that needs to be done at destruction time
  // (e.g. close files, deallocate resources etc.)
  //
  // please remove this method altogether if it would be left empty
}

//
// member functions
//

void MPIRecvInt::asyncTask(edm::WaitingTaskWithArenaHolder holder, MPI_Comm dataComm_, int tagID) {
  MPI_Recv(&recvIntData_, 1, MPI_INT, MPI_ANY_SOURCE, tagID, dataComm_, MPI_STATUS_IGNORE);
  holder.doneWaiting(nullptr);
}

void MPIRecvInt::acquire(edm::Event const& iEvent,
                         edm::EventSetup const& iSetup,
                         edm::WaitingTaskWithArenaHolder holder) {
  MPIToken tokenData = iEvent.get(communicatorToken_);

  const MPICommunicator* MPICommPTR = tokenData.token_;
  MPI_Comm dataComm_ = MPICommPTR->Communicator();
  int tagID = iEvent.get(tagIDToken_) + userTagID_;

  std::future<void> fut =
      std::async(std::launch::async, &MPIRecvInt::asyncTask, this, std::move(holder), dataComm_, tagID);
}

// ------------ method called to produce the data  ------------
void MPIRecvInt::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  /* This is an event example
  //Read 'ExampleData' from the Event
  ExampleData const& in = iEvent.get(inToken_);

  //Use the ExampleData to create an ExampleData2 which 
  // is put into the Event
  iEvent.put(std::make_unique<ExampleData2>(in));
  */

  /* this is an EventSetup example
  //Read SetupData from the SetupRecord in the EventSetup
  SetupData& setup = iSetup.getData(setupToken_);
  */

  iEvent.emplace(outIntData_, recvIntData_);
}

// ------------ method called once each stream before processing any runs, lumis or events  ------------
void MPIRecvInt::beginStream(edm::StreamID stream) {
  // please remove this method if not needed
  sid_ = stream;
  //std::cout<<"Stream ID = "<<sid_.value()<<"\n";
}

// ------------ method called once each stream after processing all runs, lumis and events  ------------
void MPIRecvInt::endStream() {
  // please remove this method if not needed
}

// ------------ method called when starting to processes a run  ------------
/*
void
MPIRecvInt::beginRun(edm::Run const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a run  ------------
/*
void
MPIRecvInt::endRun(edm::Run const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when starting to processes a luminosity block  ------------
/*
void
MPIRecvInt::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a luminosity block  ------------
/*
void
MPIRecvInt::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void MPIRecvInt::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(MPIRecvInt);
