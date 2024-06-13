// -*- C++ -*-
//
// Package:    HeterogeneousCore/MPICore
// Class:      MPIDriver
//
/**\class MPIDriver MPIDriver.cc HeterogeneousCore/MPICore/plugins/MPIDriver.cc

Description: [one line class summary]

Implementation:
[Notes on implementation]
*/
//
// Original Author:  Fawaz M Sh Kh W Alazemi
//         Created:  Tue, 28 May 2024 11:15:25 GMT
//
//

// system include files
#include <memory>
#include <mpi.h>
#include "TBufferFile.h"
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "HeterogeneousCore/MPIServices/interface/MPIService.h"
#include "HeterogeneousCore/MPICore/interface/MPICommunicator.h"

//
// class declaration
//

class MPIDriver : public edm::stream::EDProducer<edm::GlobalCache<MPICommunicator>> {
public:
  explicit MPIDriver(const edm::ParameterSet&, MPICommunicator const*);
  ~MPIDriver() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  static std::unique_ptr<MPICommunicator> initializeGlobalCache(edm::ParameterSet const&);
  static void globalEndJob(MPICommunicator*);

private:
  void beginStream(edm::StreamID) override;
  void produce(edm::Event&, const edm::EventSetup&) override;
  void endStream() override;

  //void beginRun(edm::Run const&, edm::EventSetup const&) override;
  //void endRun(edm::Run const&, edm::EventSetup const&) override;
  //void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
  //void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

  // ----------member data ---------------------------

  edm::EDPutTokenT<MPIToken> token_;
  edm::EDPutTokenT<int> tagToken_;
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
MPIDriver::MPIDriver(const edm::ParameterSet& iConfig, MPICommunicator const* MPICommPTR)
    : token_{produces()}, tagToken_{produces()} {
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

MPIDriver::~MPIDriver() {
  // do anything here that needs to be done at destruction time
  // (e.g. close files, deallocate resources etc.)
  //
  // please remove this method altogether if it would be left empty
  //  communicator_->disconnect();
}

//
// member functions
//

std::unique_ptr<MPICommunicator> MPIDriver::initializeGlobalCache(edm::ParameterSet const& iConfig) {
  //MPI_init is executed in MPIServices.cc

  MPIService::required();

  std::unique_ptr<MPICommunicator> handler =
      std::make_unique<MPICommunicator>(iConfig.getUntrackedParameter<std::string>("service"));

  handler->connect();

  return handler;
}

void MPIDriver::globalEndJob(MPICommunicator* handler) {
  std::cout << "Closing Connection " << std::endl;
  handler->disconnect();
}

// ------------ method called to produce the data  ------------
void MPIDriver::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
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
  MPICommunicator* handler = const_cast<MPICommunicator*>(globalCache());

  int tagID;
  MPI_Recv(&tagID, 1, MPI_UNSIGNED, 0, 0, handler->Communicator(), MPI_STATUS_IGNORE);
  iEvent.emplace(tagToken_, tagID);
  iEvent.emplace(token_, handler);
}

// ------------ method called once each stream before processing any runs, lumis or events  ------------
void MPIDriver::beginStream(edm::StreamID) {
  // please remove this method if not needed
}

// ------------ method called once each stream after processing all runs, lumis and events  ------------
void MPIDriver::endStream() {
  // please remove this method if not needed
}

// ------------ method called when starting to processes a run  ------------
/*
   void
   MPIDriver::beginRun(edm::Run const&, edm::EventSetup const&)
   {
   }
   */

// ------------ method called when ending the processing of a run  ------------
/*
   void
   MPIDriver::endRun(edm::Run const&, edm::EventSetup const&)
   {
   }
   */

// ------------ method called when starting to processes a luminosity block  ------------
/*
   void
   MPIDriver::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
   {
   }
   */

// ------------ method called when ending the processing of a luminosity block  ------------
/*
   void
   MPIDriver::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
   {
   }
   */

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void MPIDriver::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(MPIDriver);
