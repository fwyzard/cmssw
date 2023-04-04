// -*- C++ -*-
//
// Package:    HeterogeneousCore/MPIController
// Class:      MPIController
//
/**\class MPIController MPIController.cc HeterogeneousCore/MPIController/plugins/MPIController.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Fawaz M Sh Kh W Alazemi
//         Created:  Fri, 24 Mar 2023 21:03:08 GMT
//
//

#include <memory>
#include <iostream>
#include <string>
#include <tuple>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/MPIServices/interface/MPIService.h"
#include "HeterogeneousCore/MPICore/interface/MPICommunicator.h"

#include "api.h"
#include "messages.h"
#include <atomic>


struct CountRun {
   CountRun():value(0) {}
   //Using mutable since we want to update the value.
   mutable std::atomic<unsigned int> value;
};

struct MPICommTools{
	MPICommunicator main; 
	MPIControlLink link;
	MPI_Comm const dataComm(){ return main.dataCommunicator(); } 
};


class MPIController : public edm::stream::EDProducer<edm::GlobalCache<MPICommunicator>, edm::RunCache<CountRun>> {
public:
  explicit MPIController(const edm::ParameterSet&, MPICommunicator const*);
  ~MPIController() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  static std::unique_ptr<MPICommunicator> initializeGlobalCache(edm::ParameterSet const&);
  static void globalEndJob(MPICommunicator const* iMPICommunicator);

static std::shared_ptr<CountRun> globalBeginRun(edm::Run const&, edm::EventSetup const&, GlobalCache const*) {
       return std::shared_ptr<CountRun>(new CountRun());
   }
static void globalEndRun(edm::Run const& iRun, edm::EventSetup const&, RunContext const* iContext) {
      std::cout <<"Number of events seen in Run "<<" = "<<iContext->run()->value<<std::endl;
   }
private:
  void beginStream(edm::StreamID) override;
  void produce(edm::Event&, const edm::EventSetup&) override;
  void endStream() override;

  void beginRun(edm::Run const&, edm::EventSetup const&) override;
  void endRun(edm::Run const&, edm::EventSetup const&) override;
  void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
  void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

  // ----------member data ---------------------------
  edm::StreamID sid_ = edm::StreamID::invalidStreamID();
  edm::EDPutTokenT<MPIToken> token_;
  static MPIControlLink link; //One instance is required among all objects
};

MPIControlLink MPIController::link; 


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
MPIController::MPIController(const edm::ParameterSet& iConfig, MPICommunicator const* MPICommPTR) : token_{produces()} {
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

  log << "MPIController::MPIController is up. Link to MPISource is Set.";
}

MPIController::~MPIController() {
  // do anything here that needs to be done at destruction time
  // (e.g. close files, deallocate resources etc.)
  //
  // please remove this method altogether if it would be left empty
  edm::LogAbsolute log("MPI");
  std::cout<<"~MPIController";
}

//
// member functions
//

std::unique_ptr<MPICommunicator> MPIController::initializeGlobalCache(edm::ParameterSet const& iConfig) {
  edm::LogAbsolute log("MPI");

  EDM_MPI_build_types();  //Move to MPIServices
  edm::Service<MPIService> service;
  service->required();
  std::unique_ptr<MPICommunicator> MPICommPTR =
      std::make_unique<MPICommunicator>(iConfig.getUntrackedParameter<std::string>("service"));

  MPICommPTR->connect();
  MPICommPTR->splitCommunicator();
  

  link.setCommunicator(MPICommPTR->controlCommunicator());
  return MPICommPTR;
}

void MPIController::globalEndJob(MPICommunicator const* MPICommPTR) {
	edm::LogAbsolute log("MPI");
	std::cout<<"MPIController::globalEndJob";
	link.sendDisconnect(0); 	
}

// ------------ method called to produce the data  ------------
void MPIController::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
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

  MPICommunicator const* MPICommPTR = globalCache();
  //FIXME: tagID = Stream id is temp solution.
  int tagID = (int)(iEvent.id().event());

  //Event will be sent through Control communicator
  link.sendEvent(tagID, iEvent.eventAuxiliary());

  log << "MPIController::produce. Starting Event " << iEvent.id().event() << " on Stream " << sid_.value();
  iEvent.emplace(token_, MPICommPTR, tagID, 0); 

}

// ------------ method called once each stream before processing any runs, lumis or events  ------------
void MPIController::beginStream(edm::StreamID sid) {
  // please remove this method if not needed
  sid_ = sid;
  // link.sendConnect(sid_);
  std::cout<<"beginStream? ";
  //std::cin.ignore();

  link.sendBeginStream(sid_.value());
  edm::LogAbsolute("MPI") << "MPIController::beginStream, stream = " << sid_.value() << ".";
}

// ------------ method called once each stream after processing all runs, lumis and events  ------------
void MPIController::endStream() {
  // signal the end stream

  link.sendEndStream(sid_.value());
  edm::LogAbsolute("MPI") << "MPIController::endStream, stream = " << sid_.value() << ".";
}

// ------------ method called when starting to processes a run  ------------

void MPIController::beginRun(edm::Run const& run, edm::EventSetup const& setup) {
  

  MPICommunicator const* MPICommPTR = globalCache(); 
  

  edm::LogAbsolute log("MPI");
  auto aux = run.runAuxiliary();
  aux.setProcessHistoryID(run.processHistory().id());
  int runID = aux.run();
  
  
  std::cout<<"Run signal sent "<<runID<<"\n";   
  link.sendRunReadySignal(runID);
  
  //wait for ack
  MPI_Recv(NULL, 0, MPI_CHAR, 0, runID, MPICommPTR->controlCommunicator(), MPI_STATUS_IGNORE);  
  std::cout<<"Run Ack Received "<<runID<<"\n";
  
  
  
  link.sendBeginRun(sid_.value(), aux);
  log << "MPIController::beginRun. SentBeginRun,  ";
  //transmit the ProcessHistory

  link.sendSerializedProduct(sid_.value(), run.processHistory());
  log<<" sendSerializedProduct Sent (stream = " << sid_.value() << ").";
}

// ------------ method called when ending the processing of a run  ------------

void MPIController::endRun(edm::Run const& run, edm::EventSetup const& setup) {
  auto aux = run.runAuxiliary();
  aux.setProcessHistoryID(run.processHistory().id());
  link.sendEndRun(sid_.value(), aux);
  edm::LogAbsolute("MPI") << "MPIController::endRun, stream = " << sid_.value() << ".";
}

// ------------ method called when starting to processes a luminosity block  ------------

void MPIController::beginLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& setup) {
  auto aux = lumi.luminosityBlockAuxiliary();
  aux.setProcessHistoryID(lumi.processHistory().id());
  link.sendBeginLuminosityBlock(sid_.value(), aux);
  edm::LogAbsolute("MPI") << "MPIController::beginLuminosityBlock. stream = " << sid_.value() << ".";
}

// ------------ method called when ending the processing of a luminosity block  ------------

void MPIController::endLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& setup) {
  auto aux = lumi.luminosityBlockAuxiliary();
  aux.setProcessHistoryID(lumi.processHistory().id());
  link.sendEndLuminosityBlock(sid_.value(), aux);
  edm::LogAbsolute("MPI") << "MPIController::endLuminosityBlock, stream = " << sid_.value() << ".";
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void MPIController::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(MPIController);
