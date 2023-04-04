// -*- C++ -*-
//
// Package:    HeterogeneousCore/GenerateDummyData
// Class:      GenerateDummyData
//
/**\class GenerateDummyData GenerateDummyData.cc HeterogeneousCore/MPICore/plugins/GenerateDummyData.cc

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
#include <ctime>
#include <cstdlib>
#include <vector>
#include <sstream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

class GenerateDummyData : public edm::stream::EDProducer<> {
public:
  explicit GenerateDummyData(const edm::ParameterSet&);
  ~GenerateDummyData() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginStream(edm::StreamID) override;
  void produce(edm::Event&, const edm::EventSetup&) override;
  //void endStream() override;

  //void beginRun(edm::Run const&, edm::EventSetup const&) override;
  //void endRun(edm::Run const&, edm::EventSetup const&) override;
  //void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
  //void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

  // ----------member data ---------------------------
  edm::EDPutTokenT<std::vector<int>> dataToken_;
  edm::StreamID sid_ = edm::StreamID::invalidStreamID();
  std::vector<int> data_;
};

//
// constants, enums and typedefs
//
/******************
 * MPI Begin
 ******************/

/*********** MPI END ***********/

//
// static data member definitions
//

//
// constructors and destructor
//
GenerateDummyData::GenerateDummyData(const edm::ParameterSet& iConfig) : dataToken_{produces()} {
  //register your products
  /* Examples
  produces<ExampleData2>();

  //if do put with a label
  produces<ExampleData2>("label");
 
  //if you want to put into the Run
  produces<ExampleData2,InRun>();
  */
  //now do what ever other initialization is needed
  edm::LogAbsolute("MPI") << "GenerateDummyData::GenerateDummyData().";
}

GenerateDummyData::~GenerateDummyData() {
  // do anything here that needs to be done at destruction time
  // (e.g. close files, deallocate resources etc.)
  //
  // please remove this method altogether if it would be left empty
  edm::LogAbsolute("MPI") << "GenerateDummyData::~GenerateDummyData().";
}
//
// member functions
//

// ------------ method called to produce the data  ------------
void GenerateDummyData::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
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
  std::srand((int)iEvent.id().event());  //std::time(0));
  data_.clear();
  for (int i = 0; i < 10; i++) {
    data_.push_back(std::rand() % 10000);
  }
  std::stringstream ss;
  for (int i : data_) {
    ss << i << " ";
  }
  log << "GenerateDummyData::produce (sid_ = " << sid_.value() << ", Event = " << iEvent.id().event()
      << ") Data = " << ss.str();

  iEvent.emplace(dataToken_, data_);
}

// ------------ method called once each stream before processing any runs, lumis or events  ------------
void GenerateDummyData::beginStream(edm::StreamID stream) {
  // please remove this method if not needed
  sid_ = stream;
  edm::LogAbsolute("MPI") << "GenerateDummyData::beginStream: Stream = " << sid_.value();
}

// ------------ method called once each stream after processing all runs, lumis and events  ------------
/*
void GenerateDummyData::endStream() {
  // please remove this method if not needed
}
*/
// ------------ method called when starting to processes a run  ------------
/*
void
GenerateDummyData::beginRun(edm::Run const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a run  ------------
/*
void
GenerateDummyData::endRun(edm::Run const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when starting to processes a luminosity block  ------------
/*
void
GenerateDummyData::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a luminosity block  ------------
/*
void
GenerateDummyData::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void GenerateDummyData::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(GenerateDummyData);
