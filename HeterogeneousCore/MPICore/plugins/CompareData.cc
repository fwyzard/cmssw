// -*- C++ -*-
//
// Package:    HeterogeneousCore/CompareData
// Class:      CompareData
//
/**\class CompareData CompareData.cc HeterogeneousCore/MPICore/plugins/CompareData.cc

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
#include <vector>
#include <sstream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

//
// class declaration
//
//
//

class CompareData : public edm::stream::EDProducer<> {
public:
  explicit CompareData(const edm::ParameterSet&);
  ~CompareData() override;

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
  //MPICommunicator* x;
  edm::EDGetTokenT<std::vector<int>> sourceData_;
  edm::EDGetTokenT<std::vector<int>> controllerData_;
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
CompareData::CompareData(const edm::ParameterSet& iConfig)
    : sourceData_{consumes(iConfig.getParameter<edm::InputTag>("sourceData"))},
      controllerData_{consumes(iConfig.getParameter<edm::InputTag>("controllerData"))} {
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

CompareData::~CompareData() {
  // do anything here that needs to be done at destruction time
  // (e.g. close files, deallocate resources etc.)
  //
  // please remove this method altogether if it would be left empty
}

// ------------ method called to produce the data  ------------
void CompareData::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
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
  //****************** MPI Begin *****************
  std::vector<int> s_data = iEvent.get(sourceData_);
  std::vector<int> c_data = iEvent.get(controllerData_);
  std::sort(c_data.begin(), c_data.end());

  assert(c_data.size() == s_data.size());
  for (std::size_t i = 0; i < c_data.size(); i++) {
    assert(c_data[i] == s_data[i]);
  }
  edm::LogAbsolute("MPI") << "(CompareData::produce, (Stream = " << sid_.value() << ", Event = " << iEvent.id().event()
                          << ") Data are Equal";

  //****************** MPI END ******************
}

// ------------ method called once each stream before processing any runs, lumis or events  ------------
void CompareData::beginStream(edm::StreamID stream) {
  // please remove this method if not needed
  sid_ = stream;
  edm::LogAbsolute("MPI") << "CompareData::beginStream (Stream = " << sid_.value() << ".";
}

// ------------ method called once each stream after processing all runs, lumis and events  ------------
/*void CompareData::endStream() {
	
  // please remove this method if not needed
}*/

// ------------ method called when starting to processes a run  ------------
/*
void
CompareData::beginRun(edm::Run const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a run  ------------
/*
void
CompareData::endRun(edm::Run const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when starting to processes a luminosity block  ------------
/*
void
CompareData::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a luminosity block  ------------
/*
void
CompareData::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void CompareData::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(CompareData);
