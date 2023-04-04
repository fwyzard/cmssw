// -*- C++ -*-
//
// Package:    HeterogeneousCore/MPIEmptySource
// Class:      MPIEmptySource
//
/**\class MPIEmptySource MPIEmptySource.cc HeterogeneousCore/MPIEmptySource/plugins/MPIEmptySource.cc

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
#include "conversion.h"

class MPIEmptySource : public edm::stream::EDProducer<edm::GlobalCache<MPICommunicator>> {
public:
  explicit MPIEmptySource(const edm::ParameterSet&, MPICommunicator const*);
  ~MPIEmptySource() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  static std::unique_ptr<MPICommunicator> initializeGlobalCache(edm::ParameterSet const&);
  static void globalEndJob(MPICommunicator const* iMPICommunicator);

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
  MPIControlLink link;
  edm::EDPutTokenT<MPIToken> token_;
  int MPISourceRank_;
  MPI_Comm controlComm_ = MPI_COMM_NULL;
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
MPIEmptySource::MPIEmptySource(const edm::ParameterSet& iConfig, MPICommunicator const* MPICommPTR)
    : token_{produces()} {
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
  controlComm_ = MPICommPTR->controlCommunicator();
  link = MPIControlLink(MPICommPTR->controlCommunicator(), MPISourceRank_);
  log << "MPIEmptySource::MPIEmptySource is up. Link to MPISource " << MPISourceRank_ << " is Set.";
}

MPIEmptySource::~MPIEmptySource() {
  // do anything here that needs to be done at destruction time
  // (e.g. close files, deallocate resources etc.)
  //
  // please remove this method altogether if it would be left empty
  edm::LogAbsolute log("MPI");
  log << "MPIEmptySource::~MPIEmptySource().";
}

//
// member functions
//

std::unique_ptr<MPICommunicator> MPIEmptySource::initializeGlobalCache(edm::ParameterSet const& iConfig) {
  edm::LogAbsolute log("MPI");

  EDM_MPI_build_types();  //Move to MPIServices
  edm::Service<MPIService> service;
  service->required();
  std::unique_ptr<MPICommunicator> MPICommPTR =
      std::make_unique<MPICommunicator>(iConfig.getUntrackedParameter<std::string>("service"));
  MPICommPTR->publish_and_listen();
  MPICommPTR->splitCommunicator();

  auto [mainRank, mainSize] = MPICommPTR->rankAndSize(MPICommPTR->mainCommunicator());
  auto [contRank, contSize] = MPICommPTR->rankAndSize(MPICommPTR->controlCommunicator());
  auto [dataRank, dataSize] = MPICommPTR->rankAndSize(MPICommPTR->dataCommunicator());

  log << "MPIEmptySource::initializeGlobalCache. Connected to MPISource (Main rank=" << mainRank
      << ", size=" << mainSize << ", Controller rank=" << contRank << ", size=" << contSize
      << ", Data rank=" << dataRank << ", size=" << dataSize << ")";
  //FIXME: set MPI Source and Rank here;
  return MPICommPTR;
}

void MPIEmptySource::globalEndJob(MPICommunicator const* MPICommPTR) {
  edm::LogAbsolute("MPI") << "MPIEmptySource::globalEndJob";
}

// ------------ method called to produce the data  ------------
void MPIEmptySource::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
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

  //Event will be sent through Control communicator

  //MPI_Status status;
  MPI_Message message;
  MPI_Mprobe(MPI_ANY_SOURCE, EDM_MPI_ProcessEvent, controlComm_, &message, MPI_STATUS_IGNORE);
  edm::EventAuxiliary eventAuxiliary;
  auto [status, stream] = link.receiveEvent(eventAuxiliary, message);
  int tagID = stream;
  log << "EDM_MPI_ProcessEvent (stream = " << iEvent.id().event() << " Stream = " << sid_.value()
      << ", source = " << status.MPI_SOURCE << ", received Tag = " << tagID << ").";
  iEvent.emplace(token_, MPICommPTR, tagID, 0);
}

// ------------ method called once each stream before processing any runs, lumis or events  ------------
void MPIEmptySource::beginStream(edm::StreamID sid) {
  // please remove this method if not needed
  edm::LogAbsolute log("MPI");
  sid_ = sid;

  MPI_Status status;
  MPI_Message message;
  MPI_Mprobe(MPI_ANY_SOURCE, EDM_MPI_BeginStream, controlComm_, &message, &status);

  EDM_MPI_Empty_t buffer;
  MPI_Mrecv(&buffer, 1, EDM_MPI_Empty, &message, &status);
  log << "EDM_MPI_BeginStream (stream = " << buffer.stream << ", source = " << status.MPI_SOURCE << ").";
}

// ------------ method called once each stream after processing all runs, lumis and events  ------------
void MPIEmptySource::endStream() {
  edm::LogAbsolute log("MPI");

  MPI_Status status;
  MPI_Message message;
  MPI_Mprobe(MPI_ANY_SOURCE, EDM_MPI_EndStream, controlComm_, &message, &status);

  EDM_MPI_Empty_t buffer;
  MPI_Mrecv(&buffer, 1, EDM_MPI_Empty, &message, &status);

  log << "EDM_MPI_EndStream (stream = " << buffer.stream << ", source = " << status.MPI_SOURCE << ").";
}

// ------------ method called when starting to processes a run  ------------

void MPIEmptySource::beginRun(edm::Run const& run, edm::EventSetup const& setup) {
  edm::LogAbsolute log("MPI");

  MPI_Status status;
  MPI_Message message;
  MPI_Mprobe(MPI_ANY_SOURCE, EDM_MPI_BeginRun, controlComm_, &message, &status);
  EDM_MPI_RunAuxiliary_t buffer;
  MPI_Mrecv(&buffer, 1, EDM_MPI_RunAuxiliary, &message, &status);
  log << "EDM_MPI_BeginRun (stream = " << buffer.stream << ", source = " << status.MPI_SOURCE << ").";

  // receive the ProcessHistory
  //MPI_Mprobe(status.MPI_SOURCE, EDM_MPI_SendSerializedProduct, controlComm_, &message, &status);
  MPI_Mprobe(MPI_ANY_SOURCE, EDM_MPI_SendSerializedProduct, controlComm_, &message, &status);
  int size;
  MPI_Get_count(&status, MPI_BYTE, &size);
  char* b = new char[size];
  MPI_Mrecv(b, size, MPI_BYTE, &message, &status);
  log << "EDM_MPI_SendSerializedProduct (stream = " << buffer.stream << ", source = " << status.MPI_SOURCE << ").";
}

// ------------ method called when ending the processing of a run  ------------

void MPIEmptySource::endRun(edm::Run const& run, edm::EventSetup const& setup) {
  edm::LogAbsolute log("MPI");
  MPI_Status status;
  MPI_Message message;
  MPI_Mprobe(MPI_ANY_SOURCE, EDM_MPI_EndRun, controlComm_, &message, &status);

  EDM_MPI_RunAuxiliary_t buffer;
  MPI_Mrecv(&buffer, 1, EDM_MPI_RunAuxiliary, &message, &status);

  log << "EDM_MPI_EndRun (stream = " << buffer.stream << ", source = " << status.MPI_SOURCE << ").";
}

// ------------ method called when starting to processes a luminosity block  ------------

void MPIEmptySource::beginLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& setup) {
  edm::LogAbsolute log("MPI");
  MPI_Status status;
  MPI_Message message;
  MPI_Mprobe(MPI_ANY_SOURCE, EDM_MPI_BeginLuminosityBlock, controlComm_, &message, &status);
  // receive the LuminosityBlockAuxiliary
  EDM_MPI_LuminosityBlockAuxiliary_t buffer;
  MPI_Mrecv(&buffer, 1, EDM_MPI_LuminosityBlockAuxiliary, &message, &status);

  log << "EDM_MPI_BeginLuminosityBlock (stream = " << buffer.stream << ", source = " << status.MPI_SOURCE << ").";
  // signal a new lumisection
}

// ------------ method called when ending the processing of a luminosity block  ------------

void MPIEmptySource::endLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& setup) {
  edm::LogAbsolute log("MPI");
  MPI_Status status;
  MPI_Message message;
  MPI_Mprobe(MPI_ANY_SOURCE, EDM_MPI_EndLuminosityBlock, controlComm_, &message, &status);

  // receive the LuminosityBlockAuxiliary
  EDM_MPI_LuminosityBlockAuxiliary_t buffer;
  MPI_Mrecv(&buffer, 1, EDM_MPI_LuminosityBlockAuxiliary, &message, &status);
  log << "EDM_MPI_EndLuminosityBlock (stream = " << buffer.stream << ", source = " << status.MPI_SOURCE << ").";
  // nothing else to do
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void MPIEmptySource::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(MPIEmptySource);
