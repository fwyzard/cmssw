/// ######################################################################################
/// # Plugin used to convert the CMSSW detector elements in ACTS detector elements.      #
/// # The output is a vector of Acts::CMSDetectorElement which can be used to            #
/// # build the Tracking Geometry.                                                       #
/// # NOTE: Plugins list:                                                                #  
/// # (1) Converts the CMSSW detElements into ACTS ones and builds the Tracking Geometry # <- DONE (I) <->(III) flags to map material
/// # (2) Produces a JSON material file starting from material tracks                    # <- DONE (II)
/// # (3) Takes as input the tracking geometry from (1) and the json file from (3) and   # <- DONE (IV) validation only
/// #     decorates the tracking geometry with material. It performs material Validation #  
/// ######################################################################################
// git cms-addpkg
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/ACTSTrackerGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "DataFormats/GeometrySurface/interface/RectangularPlaneBounds.h"
#include "DataFormats/GeometrySurface/interface/TrapezoidalPlaneBounds.h"

#include "DataFormats/TrackerCommon/interface/PixelBarrelName.h"
#include "DataFormats/TrackerCommon/interface/PixelEndcapName.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DetectorDescription/Core/interface/DDRotationMatrix.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "Math/RotationZ.h"
#include "FWCore/Framework/interface/ESConsumesCollector.h"

#include "Math/Rotation3D.h"
#include "Math/AxisAngle.h"

#include "Acts/Definitions/Algebra.hpp"
#include "Acts/Detector/KdtSurfacesProvider.hpp"
#include "Acts/Geometry/Polyhedron.hpp"
#include "Acts/Geometry/GeometryContext.hpp"
#include "Acts/Surfaces/PlaneSurface.hpp"
#include "Acts/Surfaces/RectangleBounds.hpp"
#include "Acts/Surfaces/TrapezoidBounds.hpp"
#include "Acts/Visualization/GeometryView3D.hpp"
#include "Acts/Visualization/ObjVisualization3D.hpp"
#include "ActsPlugins/Json/JsonDetectorElement.hpp"
#include "Acts/Visualization/ViewConfig.hpp"

#include "Acts/Geometry/Blueprint.hpp"
#include "Acts/Geometry/ContainerBlueprintNode.hpp" 
#include "Acts/Geometry/LayerBlueprintNode.hpp" 
#include "Acts/Geometry/MaterialDesignatorBlueprintNode.hpp" 
#include "Acts/Geometry/SurfaceArrayCreator.hpp"
#include "Acts/Detector/KdtSurfacesProvider.hpp"
#include "Acts/Detector/ProtoDetector.hpp"
#include "Acts/Detector/detail/ReferenceGenerators.hpp"
#include "Acts/Detector/interface/ISurfacesProvider.hpp"
#include "Acts/Detector/LayerStructureBuilder.hpp"
#include "Acts/Detector/detail/BlueprintHelper.hpp"
#include "Acts/Detector/detail/BlueprintDrawer.hpp"
#include "Acts/Definitions/Units.hpp"
#include "Acts/Navigation/SurfaceArrayNavigationPolicy.hpp"
#include "Acts/Navigation/TryAllNavigationPolicy.hpp"
#include "ActsPlugins/ActSVG/DetectorSvgConverter.hpp"
#include "ActsPlugins/Json/JsonSurfacesReader.hpp"
#include "ActsPlugins/Json/JsonMaterialDecorator.hpp"
#include "ActsPlugins/ActSVG/TrackingGeometrySvgConverter.hpp"
#include "ActsPlugins/ActSVG/SurfaceArraySvgConverter.hpp"
#include "ActsPlugins/Root/RootMaterialTrackIo.hpp"

#include "Acts/Material/PropagatorMaterialAssigner.hpp"
#include "Acts/Material/BinnedSurfaceMaterialAccumulater.hpp"
#include "Acts/Material/BinnedSurfaceMaterial.hpp"
#include "Acts/Material/AccumulatedMaterialSlab.hpp"
#include "Acts/Material/HomogeneousSurfaceMaterial.hpp"
#include "Acts/Material/IntersectionMaterialAssigner.hpp"
#include "Acts/Material/MaterialValidater.hpp"
#include "Acts/Material/MaterialMapper.hpp"

#include <fstream>
#include <iomanip>
#include <random>
#include <nlohmann/json.hpp> 
#include "TChain.h"
#include "TFile.h"

#include "ACTSinCMSSW/GeometryDataFormat/interface/CMSDetectorElement.h"
#include "ACTSinCMSSW/GeometryDataFormat/interface/JsonMaterialWriter.hpp"

using json = nlohmann::json;
using KdtSurfacesDim2Bin100 = Acts::Experimental::KdtSurfaces<2u, 100u>;
const std::array<Acts::AxisDirection, 2UL> casts{Acts::AxisDirection::AxisZ, Acts::AxisDirection::AxisR};

using DetElVect = std::vector<std::shared_ptr<Acts::CMSDetectorElement>>;

struct TrackingGeometryWithDetEls {
    DetElVect detElements;
    std::shared_ptr<Acts::TrackingGeometry> trackingGeometry;
};
// ############################## MATERIAL METHODS ##############################

struct MaterialConfig {
  /// material collection to read
  std::string outputMaterialTracks = "material-tracks";
  /// name of the output tree
  std::string treeName = "material-tracks";
  /// List of input files
  std::vector<std::string> fileList;

  // Read surface information for the root file
  bool readCachedSurfaceInformation = false;
};

template <typename element_t, typename index_t>
void stableSort(index_t numElements, const element_t* elements,
                index_t* sortedIndices, Bool_t sortDescending) {
  for (index_t i = 0; i < numElements; i++) {
    sortedIndices[i] = i;
  }

  if (sortDescending) {
    std::stable_sort(sortedIndices, sortedIndices + numElements,
                     CompareDesc<const element_t*>(elements));
  } else {
    std::stable_sort(sortedIndices, sortedIndices + numElements,
                     CompareAsc<const element_t*>(elements));
  }
}

class MyMaterialEvReader {
public:
    MyMaterialEvReader(const MaterialConfig& config)
      : m_cfg(config){

      ActsPlugins::RootMaterialTrackIo::Config ioCfg;
      m_payload = std::make_unique<ActsPlugins::RootMaterialTrackIo>(ioCfg);
      
      m_inputChain = std::make_unique<TChain>(m_cfg.treeName.c_str());

      // loop over the input files
      for (const auto& inputFile : m_cfg.fileList) {
        // add file to the input chain
        m_inputChain->Add(inputFile.c_str());
        //std::cout << "Adding File " << inputFile << " to tree '" << m_cfg.treeName << "'." << std::endl;
      }

      // Connect the branches
      m_payload->connectForRead(*m_inputChain);

      // get the number of entries, which also loads the tree
      std::size_t nentries = m_inputChain->GetEntries();
      m_events = static_cast<std::size_t>(m_inputChain->GetMaximum("event_id") + 1);
      m_batchSize = nentries / m_events;
      //std::cout << "The full chain has "
      //           << nentries << " entries for " << m_events
      //           << " events this corresponds to a batch size of: " << m_batchSize << std::endl;

      // Sort the entry numbers of the events
      {
        // necessary to guarantee that m_inputChain->GetV1() is valid for the
        // entire range
        m_inputChain->SetEstimate(nentries + 1);

        m_entryNumbers.resize(nentries);
        m_inputChain->Draw("event_id", "", "goff");
        stableSort(m_inputChain->GetEntries(), m_inputChain->GetV1(),
                                m_entryNumbers.data(), false);
      }
    }

    std::unordered_map<std::size_t, Acts::RecordedMaterialTrack> read(std::size_t eventNumber) {

      mtrackCollection.clear();
      
      // Loop over the entries for this event
      for (std::size_t ib = 0; ib < m_batchSize; ++ib) {
        // Read the correct entry: startEntry + ib
        auto entry = m_batchSize * eventNumber + ib;
        entry = m_entryNumbers.at(entry);
        //std::cout << "Reading event: " << eventNumber
        //                               << " with stored entry: " << entry << std::endl; 
        m_inputChain->GetEntry(entry);

        Acts::RecordedMaterialTrack rmTrack = m_payload->read();

        //std::cout << "Track vertex:  " << rmTrack.first.first << std::endl;
        //std::cout << "Track momentum:" << rmTrack.first.second << std::endl;

        mtrackCollection[ib] = (std::move(rmTrack));
      }

      return mtrackCollection;
    }

private:
    MaterialConfig m_cfg;
    std::unique_ptr<TChain> m_inputChain;
    std::unique_ptr<ActsPlugins::RootMaterialTrackIo> m_payload;
    
    std::vector<long long> m_entryNumbers = {};

    std::size_t m_events = 0;
    std::size_t m_batchSize = 0;

    std::unordered_map<std::size_t, Acts::RecordedMaterialTrack> mtrackCollection;
};

struct MaterialSurfaceSelector {
  std::vector<const Acts::Surface*> surfaces = {};

  /// @param surface is the test surface
  void operator()(const Acts::Surface* surface) {
    if (surface->surfaceMaterial() != nullptr && !rangeContainsValue(surfaces, surface)) {
      surfaces.push_back(surface);
    }
  }
};

// ##########################################################################################


class JsonMaterialMapsProducer : public edm::one::EDProducer<> {
//class JsonMaterialMapsProducer : public edm::ESProducer {
public:
  explicit JsonMaterialMapsProducer(const edm::ParameterSet& ps);
  ~JsonMaterialMapsProducer() override = default;
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;
  //void produce(const ACTSTrackerGeometryRecord& iRecord);

private:
  edm::ESGetToken<TrackingGeometryWithDetEls, ACTSTrackerGeometryRecord> ACTStrkGeomInfoToken_;  
  std::string inputFile_, outputFile_;
  int Nevents_;
};

JsonMaterialMapsProducer::JsonMaterialMapsProducer(const edm::ParameterSet& ps)
    : ACTStrkGeomInfoToken_(esConsumes<TrackingGeometryWithDetEls, ACTSTrackerGeometryRecord>()),
      inputFile_(ps.getUntrackedParameter<std::string>("G4InputFile")),
      outputFile_(ps.getUntrackedParameter<std::string>("OutputFile")),
      Nevents_(ps.getUntrackedParameter<int>("Nevents")){}

void JsonMaterialMapsProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) { 
//void JsonMaterialMapsProducer::produce(const ACTSTrackerGeometryRecord& iRecord) { 
  // Get the Tracking Geometry and check if it's valid
  const auto& trkGeo_and_DetEls = iSetup.getData(ACTStrkGeomInfoToken_);
  DetElVect detEls = trkGeo_and_DetEls.detElements;
  std::shared_ptr<Acts::TrackingGeometry> trackingGeometry = trkGeo_and_DetEls.trackingGeometry;
  if (!trackingGeometry) {
      edm::LogError("ACTSRefitTracksProducer") << "ACTS TrackerGeometry is nullptr!";
      return;  
  }
  // const auto& trackingGeometry = iSetup.getData(trackerGeomToken_); // <- old
  //const TrackerGeometry& trackingGeometry = iRecord.get(trackerGeomToken_);

  std::cout << ">>>>> Create Material Maps <<<<<" << std::endl;

  // ===== Collect Material Tracks =====
  Acts::GeometryContext gCtx;
  Acts::MagneticFieldContext mfCtx;

  MaterialConfig test_reader_cfg;
  test_reader_cfg.treeName = "material-tracks";
  test_reader_cfg.fileList = {inputFile_}; // "/eos/user/l/ldamenti/G4MaterialFiles/MaterialFile_ExtendedGeo2023_10000_100_test_TrkOnly.root"
  MyMaterialEvReader my_reader(test_reader_cfg);

  // Collect the material surfaces from the trackingGeometry:
  MaterialSurfaceSelector selector;
  trackingGeometry->visitSurfaces(selector, false);
  std::vector<const Acts::Surface*> map_surf = selector.surfaces;

  // Define the Material mapper -> (1) Accumulater and (2) Assigner
  Acts::BinnedSurfaceMaterialAccumulater::Config Mat_Acc_cfg;
  Mat_Acc_cfg.geoContext = gCtx;
  Mat_Acc_cfg.materialSurfaces = map_surf;
  std::unique_ptr<const Acts::Logger> AccLogger = Acts::getDefaultLogger("BinnedSurfaceMaterialAccumulater", Acts::Logging::INFO);
  Acts::IntersectionMaterialAssigner::Config Mat_Assigner_cfg;
  Mat_Assigner_cfg.surfaces = map_surf;
  std::unique_ptr<const Acts::Logger> AssigLogger = Acts::getDefaultLogger("IntersectionMaterialAssigner", Acts::Logging::INFO);

  Acts::MaterialMapper::Config MatMap_cfg;
  MatMap_cfg.assignmentFinder = std::make_shared<Acts::IntersectionMaterialAssigner>(Mat_Assigner_cfg, std::move(AssigLogger));        // (1)
  MatMap_cfg.surfaceMaterialAccumulater = std::make_shared<Acts::BinnedSurfaceMaterialAccumulater>(Mat_Acc_cfg, std::move(AccLogger)); // (2)
  std::unique_ptr<const Acts::Logger> mlogger_MM = Acts::getDefaultLogger("Test", Acts::Logging::DEBUG);
  Acts::MaterialMapper MatMapper(MatMap_cfg, std::move(mlogger_MM));

  // Preparation before the loop: definition of the state, the WhiteBoard and the number of total event: 
  auto state = MatMapper.createState();
  double ref_index = 0;
  double Tot_events = Nevents_;
  double tracks_count = 0;

  // Loop over the total number of events:
  std::unordered_map<std::size_t, Acts::RecordedMaterialTrack> MappedTracks;
  for(int i = 0; i < Tot_events; i++){
    std::cout << "\rAnalysing event " << i+1 << " out of " << Tot_events << std::flush;
    ref_index+=1;

    //ActsExamples::AlgorithmContext AlgCont(0, i, WB);
    //reader.read(AlgCont);
    //std::unordered_map<std::size_t, Acts::RecordedMaterialTrack> this_event = reader.getMaterialTracks();
    std::unordered_map<std::size_t, Acts::RecordedMaterialTrack> this_event = my_reader.read(i);

    // Loop over all the tracks of this event: 
    for (const auto& [index, track] : this_event){
      auto [mappedTrk, unmappedTrk] = MatMapper.mapMaterial(*state, gCtx, mfCtx, track);

      //MatTrackWriter.writeT(gCtx, tracks_count, mappedTrk);
      //MappedTracks[tracks_count] = mappedTrk;
      tracks_count+=1;  
    }
  }

  // ===== Save the material maps into a json file =====

  Acts::TrackingGeometryMaterial trkGeoMat_maps = MatMapper.finalizeMaps(*state);

  JsonMaterialWriter::Config json_cfg;
  json_cfg.fileName = outputFile_;
  JsonMaterialWriter json_writer(json_cfg, Acts::Logging::Level::INFO);
  json_writer.writeMaterial(trkGeoMat_maps);




}

//DEFINE_FWK_EVENTSETUP_MODULE(JsonMaterialMapsProducer);
DEFINE_FWK_MODULE(JsonMaterialMapsProducer);


