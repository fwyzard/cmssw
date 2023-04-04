#include <deque>
#include <memory>
#include <string>
#include "mpi.h"

#include <TBuffer.h>
#include <TBufferFile.h>
#include <TClass.h>

#include "DataFormats/Provenance/interface/BranchListIndex.h"
#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "DataFormats/Provenance/interface/EventSelectionID.h"
#include "DataFormats/Provenance/interface/EventToProcessBlockIndexes.h"
#include "DataFormats/Provenance/interface/LuminosityBlockAuxiliary.h"
#include "DataFormats/Provenance/interface/ProcessHistory.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "DataFormats/Provenance/interface/RunAuxiliary.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/InputSource.h"
#include "FWCore/Framework/interface/InputSourceDescription.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/ProductProvenanceRetriever.h"
#include "FWCore/MessageLogger/interface/ErrorObj.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescriptionFiller.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Sources/interface/PuttableSourceBase.h"
#include "HeterogeneousCore/MPIServices/interface/MPIService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/MPICore/interface/MPICommunicator.h"

#include "api.h"
#include "conversion.h"
#include "messages.h"

class MPISource : public edm::PuttableSourceBase {
public:
  explicit MPISource(edm::ParameterSet const& config, edm::InputSourceDescription const& desc);
  ~MPISource() override;
  using InputSource::processHistoryRegistryForUpdate;
  using InputSource::productRegistryUpdate;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  ItemType getNextItemType() override;
  std::shared_ptr<edm::RunAuxiliary> readRunAuxiliary_() override;
  std::shared_ptr<edm::LuminosityBlockAuxiliary> readLuminosityBlockAuxiliary_() override;
  void readEvent_(edm::EventPrincipal& eventPrincipal) override;

  char port_[MPI_MAX_PORT_NAME];
  MPI_Comm controlComm_ = MPI_COMM_NULL;
  MPIControlLink link; //no need for static MPIControlLink as we have only one instance of MPISource

  edm::ProcessHistory history_;
  /* FIXME replace with a data product that keeps track of the MPIDriver origin
  edm::BranchDescription originBranchDescription_;
  edm::ProductProvenance originProvenance_;
  */
  std::optional<MPICommunicator> communicator_;
  edm::EDPutTokenT<MPIToken> token_;
  std::shared_ptr<edm::RunAuxiliary> runAuxiliary_;
  std::shared_ptr<edm::LuminosityBlockAuxiliary> luminosityBlockAuxiliary_;
  //std::deque<edm::EventAuxiliary> eventAuxiliaries_;
  struct EventData {
    edm::EventAuxiliary eventAuxiliary;
    int tagID;
    int source;
  };
  std::optional<EventData> event_;
  int receivedEvents_ = 0; 
  std::deque<int> nextRunID; 
  bool runIsActive_ = false; 
};

MPISource::MPISource(edm::ParameterSet const& config, edm::InputSourceDescription const& desc)
    : edm::PuttableSourceBase(config, desc), token_{produces()} {
  edm::LogAbsolute log("MPI");

  // FIXME move into the MPIService ?
  // make sure the EDM MPI types are available
  EDM_MPI_build_types();

  edm::Service<MPIService> service;
  service->required();

  communicator_.emplace("mpi_server");  //config.getUntrackedParameter<std::string>("service"));
  communicator_->publish_and_listen();
  communicator_->splitCommunicator();

  controlComm_ = communicator_->controlCommunicator();
  link = MPIControlLink(controlComm_, 0);

  /* FIXME move MPIRecv
  // receive the branch descriptions
  MPI_Message message;
  int source = status.MPI_SOURCE;
  while (true) {
    MPI_Mprobe(source, MPI_ANY_TAG, controlComm_, &message, &status);
    if (status.MPI_TAG == EDM_MPI_SendComplete) {
      // all branches have been received
      MPI_Mrecv(&buffer, 1, EDM_MPI_Empty, &message, &status);
      edm::LogAbsolute("MPI") << "all BranchDescription received";
      break;
    } else {
      // receive the branch description for the next event product
      assert(EDM_MPI_SendSerializedProduct == status.MPI_TAG);
      int size;
      MPI_Get_count(&status, MPI_BYTE, &size);
      TBufferFile blob{TBuffer::kRead, size};
      MPI_Mrecv(blob.Buffer(), size, MPI_BYTE, &message, &status);
      edm::BranchDescription bd;
      TClass::GetClass(typeid(edm::BranchDescription))->ReadBuffer(blob, &bd);
      bd.setDropped(false);
      bd.setProduced(false);
      bd.setOnDemand(false);
      bd.setIsProvenanceSetOnRead(true);
      bd.init();
      productRegistryUpdate().copyProduct(bd);
    }
  }
  edm::LogAbsolute("MPI") << "registered branchess:\n";
  for (auto& keyval : productRegistry()->productList()) {
    edm::LogAbsolute("MPI") << "  - " << keyval.first;
  }
  edm::LogAbsolute("MPI") << '\n';
  */
  log << "MPISource::MPISource is up.\n";
}

MPISource::~MPISource() {
  edm::LogAbsolute log("MPI");
  log << "MPISource::~MPISource()\n";
}

MPISource::ItemType MPISource::getNextItemType() {
  edm::LogAbsolute log("MPI");

  MPI_Status status;
  MPI_Message message;
  int flag; 
  std::cout<<"Waiting to receive a message.\n"; 
  /*MPI_Improbe(MPI_ANY_SOURCE, MPI_ANY_TAG, controlComm_, &flag, &message, &status);
  if( ! flag) {
	  return IsSynchronize; 
  }*/
  MPI_Mprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, controlComm_, &message, &status);
  std::cout<<"Received a message\n"; 
  //std::cin.ignore(); 
  switch (status.MPI_TAG) {
    // Connect message
    case EDM_MPI_Connect: {
      // receive the message header
      EDM_MPI_Empty_t buffer;
      MPI_Mrecv(&buffer, 1, EDM_MPI_Empty, &message, &status);

      log << "EDM_MPI_Connect (stream = " << buffer.stream << ", source = " << status.MPI_SOURCE << ").";
      // the Connect message is unexpected here (see above)
      return IsInvalid;
    }

    // Disconnect message
    case EDM_MPI_Disconnect: {
      // receive the message header
      EDM_MPI_Empty_t buffer;
      MPI_Mrecv(&buffer, 1, EDM_MPI_Empty, &message, &status);
      log<<"EDM_MPI_Disconnect (stream = " << buffer.stream << ", source = " << status.MPI_SOURCE << ").\n";
      // signal the end of the input data
      return IsStop;
    }

    // BeginStream message
    case EDM_MPI_BeginStream: {
      // receive the message header
      EDM_MPI_Empty_t buffer;
      MPI_Mrecv(&buffer, 1, EDM_MPI_Empty, &message, &status);
      log<<"EDM_MPI_BeginStream (stream = " << buffer.stream << ", source = " << status.MPI_SOURCE << ").\n";
      // nothing else to do
      return IsFile; //getNextItemType();
    }

    // EndStream message
    case EDM_MPI_EndStream: {
      // receive the message header
      EDM_MPI_Empty_t buffer;
      MPI_Mrecv(&buffer, 1, EDM_MPI_Empty, &message, &status);

      log<<"EDM_MPI_EndStream (stream = " << buffer.stream << ", source = " << status.MPI_SOURCE << ").\n";
      // nothing else to do
      return IsSynchronize; //getNextItemType();
    }



    // RunReadySignal message
    case EDM_MPI_RunReadySignal: {
	EDM_MPI_RunAuxiliary_t buffer;
      MPI_Mrecv(&buffer, 1, EDM_MPI_Empty, &message, &status);
      int runID = buffer.stream ; 
      if(runIsActive_){
	nextRunID.push_back(runID); 
      }else{
	runIsActive_ = true; 
	MPI_Ssend(NULL, 0, MPI_CHAR, 0, runID, controlComm_); 
      }

	return IsSynchronize; 

    }
    // BeginRun message
    case EDM_MPI_BeginRun: {
      // receive the RunAuxiliary
      EDM_MPI_RunAuxiliary_t buffer;
      MPI_Mrecv(&buffer, 1, EDM_MPI_RunAuxiliary, &message, &status);

      std::cout << "EDM_MPI_BeginRun (stream = " << buffer.stream << ", source = " << status.MPI_SOURCE << ").\n";

      runAuxiliary_ = std::make_shared<edm::RunAuxiliary>();
      edmFromBuffer(buffer, *runAuxiliary_);

      // receive the ProcessHistory
      MPI_Mprobe(status.MPI_SOURCE, EDM_MPI_SendSerializedProduct, controlComm_, &message, &status);
      //MPI_Mprobe(MPI_ANY_SOURCE, EDM_MPI_SendSerializedProduct, controlComm_, &message, &status);
      int size;
      MPI_Get_count(&status, MPI_BYTE, &size);
      TBufferFile blob{TBuffer::kRead, size};
      MPI_Mrecv(blob.Buffer(), size, MPI_BYTE, &message, &status);
      log<<"EDM_MPI_SendSerializedProduct (stream = " << buffer.stream << ", source = " << status.MPI_SOURCE << ").\n";
      history_.clear();
      TClass::GetClass(typeid(edm::ProcessHistory))->ReadBuffer(blob, &history_);
      history_.initializeTransients();
      if (processHistoryRegistryForUpdate().registerProcessHistory(history_)) {
        edm::LogAbsolute("MPI") << "new ProcessHistory registered: " << history_;
      }

      // signal a new run
      return IsRun;
    }

    // EndRun message
    case EDM_MPI_EndRun: {
      // receive the RunAuxiliary message
      EDM_MPI_RunAuxiliary_t buffer;
      MPI_Mrecv(&buffer, 1, EDM_MPI_RunAuxiliary, &message, &status);

      log << "EDM_MPI_EndRun (stream = " << buffer.stream << ", source = " << status.MPI_SOURCE << ").\n";     // nothing else to do

      runIsActive_ = false ; 


      if( ! nextRunID.empty()){
	int runID = nextRunID.front(); 
       	nextRunID.pop_front(); 
	runIsActive_ = true; 
//	link.sendRunReadyAck(runID); //FIXME: Should use RunID as a tag. 
	MPI_Ssend(NULL, 0, MPI_CHAR, 0, runID, controlComm_); 

      }

      return IsSynchronize; //getNextItemType();
    }

    // BeginLuminosityBlock message
    case EDM_MPI_BeginLuminosityBlock: {
      // receive the LuminosityBlockAuxiliary
      EDM_MPI_LuminosityBlockAuxiliary_t buffer;
      MPI_Mrecv(&buffer, 1, EDM_MPI_LuminosityBlockAuxiliary, &message, &status);

      luminosityBlockAuxiliary_ = std::make_shared<edm::LuminosityBlockAuxiliary>();
      edmFromBuffer(buffer, *luminosityBlockAuxiliary_);

      log<<"EDM_MPI_BeginLuminosityBlock (stream = " << buffer.stream << ", source = " << status.MPI_SOURCE << ").\n";
      // signal a new lumisection
      return IsLumi;
    }

    // EndLuminosityBlock message
    case EDM_MPI_EndLuminosityBlock: {
      // receive the LuminosityBlockAuxiliary
      EDM_MPI_LuminosityBlockAuxiliary_t buffer;
      MPI_Mrecv(&buffer, 1, EDM_MPI_LuminosityBlockAuxiliary, &message, &status);
      log<<"EDM_MPI_EndLuminosityBlock (stream = " << buffer.stream << ", source = " << status.MPI_SOURCE << ").\n";
      // nothing else to do
      return IsSynchronize; //getNextItemType();
    }

    // ProcessEvent message
    case EDM_MPI_ProcessEvent: {
      // allocate a new event
      
      //auto& event = events_.emplace_back();
      //event.eventProducts.reserve(productRegistryUpdate().size());
      //auto& event = eventAuxiliaries_.emplace_back();

      // receive the EventAuxiliary
      // FIXME: create a new function for link.receiveEvent that returns status, stream, aux. 
      EventData currentEvent; 
      auto [status, stream] = link.receiveEvent(currentEvent.eventAuxiliary, message);
      currentEvent.tagID = stream ; 
      currentEvent.source = status.MPI_SOURCE;       
      //auto [status, stream] = link.receiveEvent(event, message);
      //int source = status.MPI_SOURCE;
      //event.source = status.MPI_SOURCE;
      //event.stream = stream;
      log <<"EDM_MPI_ProcessEvent (tagID = " << currentEvent.tagID << ", source = " << currentEvent.source << ").\n";

      event_ = std::make_optional<EventData>(currentEvent); 
      /* FIXME move MPIRecv
      //
      MPI_Message message;
      while (true) {
        MPI_Mprobe(source, MPI_ANY_TAG, controlComm_, &message, &status);
        if (EDM_MPI_SendComplete == status.MPI_TAG) {
          // all products have been received
          EDM_MPI_Empty_t buffer;
          MPI_Mrecv(&buffer, 1, EDM_MPI_Empty, &message, &status);
          edm::LogAbsolute("MPI") << "all Products received";
          break;
        } else {
          edm::BranchKey key;
          edm::ProductProvenance provenance;
          edm::ProductID pid;
          edm::WrapperBase* wrapper;
          {
            // receive the BranchKey
            assert(EDM_MPI_SendSerializedProduct == status.MPI_TAG);
            int size;
            MPI_Get_count(&status, MPI_BYTE, &size);
            TBufferFile buffer{TBuffer::kRead, size};
            MPI_Mrecv(buffer.Buffer(), size, MPI_BYTE, &message, &status);
            TClass::GetClass(typeid(edm::BranchKey))->ReadBuffer(buffer, &key);
          }

          edm::BranchDescription const& branch = productRegistry()->productList().at(key);

          MPI_Mprobe(source, MPI_ANY_TAG, controlComm_, &message, &status);
          {
            // receive the ProductProvenance
            assert(EDM_MPI_SendSerializedProduct == status.MPI_TAG);
            int size;
            MPI_Get_count(&status, MPI_BYTE, &size);
            TBufferFile buffer{TBuffer::kRead, size};
            MPI_Mrecv(buffer.Buffer(), size, MPI_BYTE, &message, &status);
            TClass::GetClass(typeid(edm::ProductProvenance))->ReadBuffer(buffer, &provenance);
          }
          MPI_Mprobe(source, MPI_ANY_TAG, controlComm_, &message, &status);
          {
            // receive the ProductID
            assert(EDM_MPI_SendSerializedProduct == status.MPI_TAG);
            int size;
            MPI_Get_count(&status, MPI_BYTE, &size);
            TBufferFile buffer{TBuffer::kRead, size};
            MPI_Mrecv(buffer.Buffer(), size, MPI_BYTE, &message, &status);
            TClass::GetClass(typeid(edm::ProductID))->ReadBuffer(buffer, &pid);
          }
          MPI_Mprobe(source, MPI_ANY_TAG, controlComm_, &message, &status);
          {
            // receive the product
            assert(EDM_MPI_SendSerializedProduct == status.MPI_TAG);
            int size;
            MPI_Get_count(&status, MPI_BYTE, &size);
            TBufferFile buffer{TBuffer::kRead, size};
            MPI_Mrecv(buffer.Buffer(), size, MPI_BYTE, &message, &status);
            // construct an edm::Wrapper<T> and fill it with the received product
            // TODO this would be much simpler if the MPIDriver could sent the Wrapper<T> instead of T
            edm::TypeWithDict const& type = branch.wrappedType();
            edm::ObjectWithDict object = type.construct();
            *reinterpret_cast<bool*>(reinterpret_cast<char*>(object.address()) +
                                     type.dataMemberByName("present").offset()) = true;
            branch.unwrappedType().getClass()->ReadBuffer(
                buffer, reinterpret_cast<char*>(object.address()) + type.dataMemberByName("obj").offset());
            wrapper = reinterpret_cast<edm::WrapperBase*>(object.address());
          }
          edm::LogAbsolute("MPI") << "received object for branch " << key;
          //edm::LogAbsolute("MPI") << "received object of type " << branch.unwrappedType();

          // store the received product
          event.eventProducts.emplace_back(std::unique_ptr<edm::WrapperBase>(wrapper), &branch, provenance);
        }
      }
      */

      // signal a new event
      receivedEvents_ ++; 
      return IsEvent;
    }

    // unexpected message
    default: {
      log << "MPISource::getNextItemType(): invalid tag.";
      return IsInvalid;
    }
  }
}

std::shared_ptr<edm::RunAuxiliary> MPISource::readRunAuxiliary_() { return runAuxiliary_; }

std::shared_ptr<edm::LuminosityBlockAuxiliary> MPISource::readLuminosityBlockAuxiliary_() {
  return luminosityBlockAuxiliary_;
}

void MPISource::readEvent_(edm::EventPrincipal& eventPrincipal) { 
  
  assert(event_.has_value());  
  edm::LogAbsolute("MPI") << "number of received events: " << receivedEvents_;
  EventData& currentEvent = event_.value(); 
  auto& aux = currentEvent.eventAuxiliary;
  edm::ProductProvenanceRetriever prov(eventPrincipal.transitionIndex(), *productRegistry());
  eventPrincipal.fillEventPrincipal(aux,
                                    &history_,
                                    edm::EventSelectionIDVector{},
                                    edm::BranchListIndexes{},
                                    edm::EventToProcessBlockIndexes{},
                                    prov,
                                    nullptr,
                                    false);

  edm::Event event(eventPrincipal, moduleDescription(), nullptr);
  event.setProducer(this, nullptr);
  event.emplace(token_, &communicator_.value(), currentEvent.tagID, currentEvent.source);
  commit_(event);

  /* FIXME move MPIRecv ?
  for (auto& product : event.eventProducts) {
    //edm::LogAbsolute("MPI") << "putting object for branch " << *product.branchDescription;
    eventPrincipal.put(*product.branchDescription, std::move(product.product), product.provenance);
  }
  */

  //eventAuxiliaries_.pop_front();
}

void MPISource::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setComment("Comunicate with another cmsRun process over MPI.");
  edm::InputSource::fillDescription(desc);
  descriptions.add("source", desc);
}

#include "FWCore/Framework/interface/InputSourceMacros.h"
DEFINE_FWK_INPUT_SOURCE(MPISource);
