// -*- C++ -*-
//
// Package:    HeterogeneousCore/MPICore
// Class:      DataValidator
//
/**\class DataValidator DataValidator.cc HeterogeneousCore/MPICore/plugins/DataValidator.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Fawaz M Sh Kh W Alazemi
//         Created:  Mon, 10 Jun 2024 07:51:23 GMT
//
//

// system include files
#include <memory>
#include <iostream>
#include <vector>
#include <algorithm>

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

class DataValidator : public edm::stream::EDProducer<> {
public:
  explicit DataValidator(const edm::ParameterSet&);
  ~DataValidator() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginStream(edm::StreamID) override;
  void produce(edm::Event&, const edm::EventSetup&) override;
  void endStream() override;

  edm::EDGetTokenT<int> originalIntToken_;
  edm::EDGetTokenT<int> receivedIntToken_;

  edm::EDGetTokenT<std::vector<int>> originalVectorToken_;
  edm::EDGetTokenT<std::vector<int>> receivedVectorToken_;

  //void beginRun(edm::Run const&, edm::EventSetup const&) override;
  //void endRun(edm::Run const&, edm::EventSetup const&) override;
  //void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
  //void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

  // ----------member data ---------------------------
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
DataValidator::DataValidator(const edm::ParameterSet& iConfig)
    : originalIntToken_{consumes(iConfig.getParameter<edm::InputTag>("originalInt"))},
      receivedIntToken_{consumes(iConfig.getParameter<edm::InputTag>("receivedInt"))},
      originalVectorToken_{consumes(iConfig.getParameter<edm::InputTag>("originalVector"))},
      receivedVectorToken_{consumes(iConfig.getParameter<edm::InputTag>("receivedVector"))} {
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

DataValidator::~DataValidator() {
  // do anything here that needs to be done at destruction time
  // (e.g. close files, deallocate resources etc.)
  //
  // please remove this method altogether if it would be left empty
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void DataValidator::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
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

  int eventID = iEvent.id().event();

  int originalData = iEvent.get(originalIntToken_);
  int receivedData = iEvent.get(receivedIntToken_);

  bool flag = true;
  if (originalData != -1 * receivedData) {
    std::cout << "eventID " << eventID << " -->  failed int mismatched (" << originalData << ", " << receivedData
              << ")\n";
    flag = false;
  }

  std::vector<int> originalV = iEvent.get(originalVectorToken_);
  std::vector<int> receivedV = iEvent.get(receivedVectorToken_);

  if (originalV.size() != receivedV.size()) {
    std::cout << "EventID " << eventID << " --> failed v.size (" << originalV.size() << ", " << receivedV.size()
              << ")\n";
    flag = false;
  }

  std::sort(originalV.begin(), originalV.end());
  for (auto i = 0; i < int(originalV.size()); i++) {
    if (receivedV[i] != originalV[i]) {
      std::cout << "eventID " << eventID << " -->  failed v Content Mismatch\n";
      flag = false;
    }
  }
  if (flag) {
    std::cout << "EventID " << eventID << " = Passed\n";
  }
}

// ------------ method called once each stream before processing any runs, lumis or events  ------------
void DataValidator::beginStream(edm::StreamID) {
  // please remove this method if not needed
}

// ------------ method called once each stream after processing all runs, lumis and events  ------------
void DataValidator::endStream() {
  // please remove this method if not needed
}

// ------------ method called when starting to processes a run  ------------
/*
void
DataValidator::beginRun(edm::Run const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a run  ------------
/*
void
DataValidator::endRun(edm::Run const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when starting to processes a luminosity block  ------------
/*
void
DataValidator::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a luminosity block  ------------
/*
void
DataValidator::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void DataValidator::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(DataValidator);
