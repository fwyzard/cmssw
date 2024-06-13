// standard include files
#include <memory>

// MPI include files
#include <mpi.h>

// CMSSW include files
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "HeterogeneousCore/MPICore/interface/MPICommunicator.h"
#include "HeterogeneousCore/MPIServices/interface/MPIService.h"

//
// class declaration
//

struct MPICommHandler {
  mutable std::atomic<unsigned int> tagIDCounter_;
  MPICommunicator comm_;
  MPICommHandler(std::string serverName) : tagIDCounter_(1), comm_{serverName} {};
  //Using mutable since we want to update the value.
  int getNewTagID() {
    tagIDCounter_++;
    if (tagIDCounter_ == 0)
      tagIDCounter_ = 1;
    return tagIDCounter_;
  }
};

class MPISource : public edm::stream::EDProducer<edm::GlobalCache<MPICommHandler>> {
public:
  explicit MPISource(const edm::ParameterSet&, MPICommHandler const*);
  ~MPISource() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  static std::unique_ptr<MPICommHandler> initializeGlobalCache(edm::ParameterSet const&);
  static void globalEndJob(MPICommHandler*);

private:
  void beginStream(edm::StreamID) override;
  void produce(edm::Event&, const edm::EventSetup&) override;
  void endStream() override;

  //void beginRun(edm::Run const&, edm::EventSetup const&) override;
  //void endRun(edm::Run const&, edm::EventSetup const&) override;
  //void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
  //void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

  // ----------member data ---------------------------

  MPI_Comm comm_ = MPI_COMM_NULL;
  //std::optional<MPICommunicator const*> communicator_;
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
MPISource::MPISource(const edm::ParameterSet& iConfig, MPICommHandler const* hanlder)
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

  /*
  //MPI_init is executed in MPIServices.cc
   
  MPIService::required();
  communicator_.emplace("mpi_server");  //config.getUntrackedParameter<std::string>("service"));
  communicator_->publish_and_listen();
  */
  // communicator_ = MPICommPTR;
}

MPISource::~MPISource() {
  // do anything here that needs to be done at destruction time
  // (e.g. close files, deallocate resources etc.)
  //
  // please remove this method altogether if it would be left empty
}

//
// member functions
//

std::unique_ptr<MPICommHandler> MPISource::initializeGlobalCache(edm::ParameterSet const& iConfig) {
  //MPI_init is executed in MPIServices.cc

  MPIService::required();

  std::unique_ptr<MPICommHandler> handler =
      std::make_unique<MPICommHandler>(iConfig.getUntrackedParameter<std::string>("service"));

  (handler->comm_).publish_and_listen();

  return handler;
}

void MPISource::globalEndJob(MPICommHandler* handler) {
  std::cout << "Closing connection: number of tags used = " << handler->tagIDCounter_ << std::endl;
  (handler->comm_).disconnect();
}

// ------------ method called to produce the data  ------------

void MPISource::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
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
  MPICommHandler* handler = const_cast<MPICommHandler*>(globalCache());
  int tagID = 10000 * handler->getNewTagID();

  MPI_Send(&tagID, 1, MPI_UNSIGNED, 0, 0, (handler->comm_).Communicator());

  iEvent.emplace(tagToken_, tagID);
  iEvent.emplace(token_, &handler->comm_);
}

// ------------ method called once each stream before processing any runs, lumis or events  ------------
void MPISource::beginStream(edm::StreamID) {
  // please remove this method if not needed
}

// ------------ method called once each stream after processing all runs, lumis and events  ------------
void MPISource::endStream() {
  // please remove this method if not needed
}

// ------------ method called when starting to processes a run  ------------
/*
void
MPISource::beginRun(edm::Run const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a run  ------------
/*
void
MPISource::endRun(edm::Run const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when starting to processes a luminosity block  ------------
/*
void
MPISource::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a luminosity block  ------------
/*
void
MPISource::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void MPISource::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(MPISource);
