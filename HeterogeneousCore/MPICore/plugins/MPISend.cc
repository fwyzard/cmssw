// -*- C++ -*-
//
// Package:    HeterogeneousCore/MPICore
// Class:      MPISend
//
/**\class MPISend MPISend.cc HeterogeneousCore/MPICore/plugins/MPISend.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Fawaz M Sh Kh W Alazemi
//         Created:  Tue, 28 May 2024 11:13:11 GMT
//
//

// system include files
#include <memory>
#include <mpi.h>

#include <future>
#include <thread>
#include <chrono>
#include "TBufferFile.h"
#include "TClass.h"
#include "TString.h"

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
class MPISend : public edm::stream::EDProducer<> {
public:
  explicit MPISend(const edm::ParameterSet&);
  ~MPISend() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginStream(edm::StreamID) override;
  void produce(edm::Event&, const edm::EventSetup&) override;
  void endStream() override;

  //void beginRun(edm::Run const&, edm::EventSetup const&) override;
  //void endRun(edm::Run const&, edm::EventSetup const&) override;
  //void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
  //void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

  // ----------member data ---------------------------
  edm::EDGetTokenT<MPIToken> communicatorToken_;
  edm::EDGetTokenT<int> tagIDToken_;
  edm::EDGetTokenT<int> intToken_;
  edm::EDGetTokenT<std::vector<int>> vectorToken_;
  int userTagID_;
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
      tagIDToken_{consumes(iConfig.getParameter<edm::InputTag>("tagID"))},
      intToken_{consumes(iConfig.getParameter<edm::InputTag>("intData"))},
      vectorToken_{consumes(iConfig.getParameter<edm::InputTag>("vectorData"))},
      userTagID_{iConfig.getUntrackedParameter<int>("userTagID")} {
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
MPISend::~MPISend() {
  // do anything here that needs to be done at destruction time
  // (e.g. close files, deallocate resources etc.)
  //
  // please remove this method altogether if it would be left empty
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void MPISend::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
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

  edm::LogAbsolute log("MPI");

  MPIToken mpiCommToken = iEvent.get(communicatorToken_);
  const MPICommunicator* MPICommPTR = mpiCommToken.token_;  //replace with a better name
  MPI_Comm dataComm_ = MPICommPTR->Communicator();

  int tagID = iEvent.get(tagIDToken_) + userTagID_;

  int intData = iEvent.get(intToken_);
  MPI_Send(&intData, 1, MPI_INT, 0, tagID, dataComm_);

  std::vector<int> vectorData = iEvent.get(vectorToken_);
  vectorData.shrink_to_fit();

  TBufferFile send_buffer(TBuffer::kWrite);
  TClass* type = TClass::GetClass(typeid(std::vector<int>));
  send_buffer.WriteObjectAny((void*)&vectorData, type, false);
  MPI_Send(send_buffer.Buffer(), send_buffer.Length(), MPI_BYTE, 0, tagID + 1, dataComm_);
}

// ------------ method called once each stream before processing any runs, lumis or events  ------------
void MPISend::beginStream(edm::StreamID stream) {
  sid_ = stream;
  // please remove this method if not needed
}

// ------------ method called once each stream after processing all runs, lumis and events  ------------
void MPISend::endStream() {
  // please remove this method if not needed
}

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
