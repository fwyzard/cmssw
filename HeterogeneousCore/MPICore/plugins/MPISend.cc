// -*- C++ -*-
//
// Package:    HeterogeneousCore/MPISend
// Class:      MPISend
//
/**\class MPISend MPISend.cc HeterogeneousCore/MPICore/plugins/MPISend.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Fawaz M Sh Kh W Alazemi
//         Created:  Thu, 17 Nov 2022 09:50:39 GMT
//
//

// system include files
#include <memory>
#include <iostream>
#include <mpi.h>
#include <vector>
#include <sstream>

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
//
//

class MPISend : public edm::stream::EDProducer<> {
public:
  explicit MPISend(const edm::ParameterSet&);
  ~MPISend() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  //void acquire(edm::Event const&, edm::EventSetup const&, edm::WaitingTaskWithArenaHolder) override;
private:
  void beginStream(edm::StreamID) override;
  void produce(edm::Event&, const edm::EventSetup&) override;
  //void endStream() override;

  //void beginRun(edm::Run const&, edm::EventSetup const&) override;
  //void endRun(edm::Run const&, edm::EventSetup const&) override;
  //void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
  //void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

  // ----------member data ---------------------------
  edm::EDGetTokenT<MPIToken> communicatorToken_;
  edm::EDGetTokenT<std::vector<int>> inData_;
  edm::EDPutTokenT<std::vector<int>> outData_;
  edm::EDPutTokenT<MPIToken> comOut_;
  edm::StreamID sid_ = edm::StreamID::invalidStreamID();
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
MPISend::MPISend(const edm::ParameterSet& iConfig)
    : communicatorToken_{consumes(iConfig.getParameter<edm::InputTag>("communicator"))},
      inData_{consumes(iConfig.getParameter<edm::InputTag>("incomingData"))},
      comOut_{produces()} {
  //register your products
  /* Examples
  produces<ExampleData2>();

  //if do put with a label
  produces<ExampleData2>("label");
 
  //if you want to put into the Run
  produces<ExampleData2,InRun>();
  */
  //now do what ever other initialization is needed

  edm::LogAbsolute log("MPI");

  log << "MPISend::MPISend is up.";
}

MPISend::~MPISend() {
  // do anything here that needs to be done at destruction time
  // (e.g. close files, deallocate resources etc.)
  //
  // please remove this method altogether if it would be left empty
}

//
// member functions
//
/*
 void MPISend::acquire(edm::Event const& iEvent, edm::EventSetup const& iSetup, edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
	 
  std::async([this, holder = std::move(waitingTaskHolder)]{
    MPI_Send("123", 3, MPI_BYTE, 0, 0, MPI_COMM_WORLD);
    holder.doneWaiting(nullptr);
  });

}
*/

// ------------ method called to produce the data  ------------
void MPISend::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  using namespace std;
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
  edm::LogAbsolute log("MPI");

  MPIToken tokenData = iEvent.get(communicatorToken_);

  const MPICommunicator* MPICommPTR = tokenData.token_;  //replace with a better name
  MPI_Comm dataComm_ = MPICommPTR->dataCommunicator();

  int tagID = tokenData.tagID_;  //replace with a better name
  int dest = tokenData.source_;   //replace with a better name

  std::vector<int> data = iEvent.get(inData_);

  MPI_Send(&data[0], (int)data.size(), MPI_INT, dest, tagID, dataComm_);

  iEvent.emplace(comOut_, tokenData);
}

// ------------ method called once each stream before processing any runs, lumis or events  ------------
void MPISend::beginStream(edm::StreamID stream) {
  // please remove this method if not needed
  sid_ = stream;
  edm::LogAbsolute("MPI") << "MPISend::beginStream (Stream = " << sid_.value() << ").";
}

// ------------ method called once each stream after processing all runs, lumis and events  ------------
/*
void MPISend::endStream() {
  // please remove this method if not needed
}
*/
// ------------ method called when starting to processes a run  ------------
/*
void
MPISend::beginRun(edm::Run const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a run  ------------
/*
void
MPISend::endRun(edm::Run const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when starting to processes a luminosity block  ------------
/*
void
MPISend::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a luminosity block  ------------
/*
void
MPISend::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void MPISend::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(MPISend);
