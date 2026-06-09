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

// ##################### Decorator Methods #################################################    

struct MaterialSurfaceSelector {
  std::vector<const Acts::Surface*> surfaces = {};

  /// @param surface is the test surface
  void operator()(const Acts::Surface* surface) {
    if (surface->surfaceMaterial() != nullptr && !rangeContainsValue(surfaces, surface)) {
      surfaces.push_back(surface);
    }
  }
};


struct MaterialValidation_cfg{
  /// Number of tracks per event
  std::size_t ntracks = 1000;
  /// Start position for the scan
  Acts::Vector3 startPosition = Acts::Vector3(0., 0., 0.);
  /// Start direction for the scan: phi
  std::pair<double, double> phiRange = {-std::numbers::pi, std::numbers::pi};
  /// Start direction for the scan: eta
  std::pair<double, double> etaRange = {-4., 4.};
  // The validater
  std::shared_ptr<Acts::MaterialValidater> materialValidater = nullptr;
  /// Output collection name
  std::string outputMaterialTracks = "material_tracks";

};

using RandomEngine = std::mt19937;
using RandomSeed = uint32_t;

struct myContext{
  /// Magnetic and Geometry contrext
  Acts::GeometryContext geoContext;
  Acts::MagneticFieldContext magFieldContext;
  /// Number of event
  std::size_t EvNumber; 
};

class MaterialValidator {
public:
  MaterialValidator(const MaterialValidation_cfg& cfg)
  : m_cfg(cfg){
    if(m_cfg.materialValidater == nullptr){
      throw std::invalid_argument("Missing material validater.");
    }
  }

  std::unordered_map<std::size_t, Acts::RecordedMaterialTrack> execute(const myContext& context, bool debug = false){
    // Create a random number generator
    RandomSeed seed = 1234567890u;
    RandomEngine rng(seed + context.EvNumber);

    // Setup random number distributions for some quantities
    std::uniform_real_distribution<double> phiDist(m_cfg.phiRange.first, m_cfg.phiRange.second);
    std::uniform_real_distribution<double> etaDist(m_cfg.etaRange.first, m_cfg.etaRange.second);

    // Loop over the number of tracks
    for (std::size_t iTrack = 0; iTrack < m_cfg.ntracks; ++iTrack) {
      if(debug) std::cout << "\rEv: " << context.EvNumber << ", Track " << iTrack + 1 << " out of " << m_cfg.ntracks << std::flush;
      // Generate a random phi and eta
      double phi = phiDist(rng);
      double eta = etaDist(rng);
      double theta = 2 * std::atan(std::exp(-eta));
      Acts::Vector3 direction(std::cos(phi) * std::sin(theta), std::sin(phi) * std::sin(theta), std::cos(theta));

      // Record the material
      auto rMaterial = m_cfg.materialValidater->recordMaterial(
          context.geoContext, context.magFieldContext, m_cfg.startPosition,
          direction);

      recordedMaterialTracks[iTrack] = rMaterial;
    }

    return recordedMaterialTracks;
  }

private:
  MaterialValidation_cfg m_cfg;
  std::unordered_map<std::size_t, Acts::RecordedMaterialTrack> recordedMaterialTracks;

};

struct MyMaterialTrackWriter_cfg {
  /// material collection to write
  std::string inputMaterialTracks = "material-tracks";
  /// path of the output file
  std::string filePath = "";
  /// file access mode
  std::string fileMode = "RECREATE";
  /// name of the output tree
  std::string treeName = "material-tracks";

  /// Re-calculate total values from individual steps (for cross-checks)
  bool recalculateTotals = false;
  /// Write aut pre and post step (for G4), otherwise central step position
  bool prePostStep = false;
  /// Write the surface to which the material step correpond
  bool storeSurface = false;
  /// Write the volume to which the material step correpond
  bool storeVolume = false;
  /// Collapse consecutive interactions of a single surface into a single
  /// interaction
  bool collapseInteractions = false;
};

class MyMaterialTrackWriter {
 public:
      MyMaterialTrackWriter(const MyMaterialTrackWriter_cfg& config)
        : m_cfg(config){

      ActsPlugins::RootMaterialTrackIo::Config ioCfg;
      m_payload = std::make_unique<ActsPlugins::RootMaterialTrackIo>(ioCfg);

      // An input collection name and tree name must be specified
      if (m_cfg.inputMaterialTracks.empty()) {
        throw std::invalid_argument("Missing input collection");
      } else if (m_cfg.treeName.empty()) {
        throw std::invalid_argument("Missing tree name");
      }

      // Setup ROOT I/O
      m_outputFile = TFile::Open(m_cfg.filePath.c_str(), m_cfg.fileMode.c_str());
      if (m_outputFile == nullptr) {
        throw std::ios_base::failure("Could not open '" + m_cfg.filePath + "'");
      }

      m_outputFile->cd();
      m_outputTree = new TTree(m_cfg.treeName.c_str(), "TTree from RootMaterialTrackWriter");
      if (m_outputTree == nullptr) {
        throw std::bad_alloc();
      }
      // Connect the branches
      m_payload->connectForWrite(*m_outputTree);
      }

      ~MyMaterialTrackWriter() {
          m_outputTree = nullptr;
          m_outputFile = nullptr;
      }

      void finalize() {
        if (m_outputFile && m_outputTree) {
            m_outputFile->cd();
            m_outputTree->Write();
            m_outputFile->Close();
        }
        m_outputTree = nullptr;
        m_outputFile = nullptr;
      }

      void writeT(
          const myContext& ctx, 
          const std::unordered_map<std::size_t, Acts::RecordedMaterialTrack>& materialTracks) {

        // Loop over the material tracks and write them out
        for (auto& [idTrack, mtrack] : materialTracks) {
          m_payload->write(ctx.geoContext, ctx.EvNumber, mtrack);
          m_outputTree->Fill();
        }
      }

 private:
      /// The config class
      MyMaterialTrackWriter_cfg m_cfg;
      /// The output file name
      TFile* m_outputFile = nullptr;
      /// The output tree name
      TTree* m_outputTree = nullptr;
      /// The read - write payload
      std::unique_ptr<ActsPlugins::RootMaterialTrackIo> m_payload;
};
// ##########################################################################################

class MappedMaterialValidator : public edm::one::EDProducer<> {
//class MappedMaterialValidator : public edm::ESProducer {
public:
  explicit MappedMaterialValidator(const edm::ParameterSet& ps);
  ~MappedMaterialValidator() override = default;
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;
  //void produce(const ACTSTrackerGeometryRecord& iRecord);

private:
  edm::ESGetToken<TrackingGeometryWithDetEls, ACTSTrackerGeometryRecord> ACTStrkGeomInfoToken_;  
  int Nevents_, Tracks_perEv_;
};

MappedMaterialValidator::MappedMaterialValidator(const edm::ParameterSet& ps)
    : ACTStrkGeomInfoToken_(esConsumes<TrackingGeometryWithDetEls, ACTSTrackerGeometryRecord>()),
      Nevents_(ps.getUntrackedParameter<int>("Nevents")),
      Tracks_perEv_(ps.getUntrackedParameter<int>("Ntracks")) {}

void MappedMaterialValidator::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) { 
//void MappedMaterialValidator::produce(const ACTSTrackerGeometryRecord& iRecord) { 
  // Get the Tracking Geometry and check if it's valid
  const auto& trkGeo_and_DetEls = iSetup.getData(ACTStrkGeomInfoToken_);
  DetElVect detEls = trkGeo_and_DetEls.detElements;
  std::shared_ptr<Acts::TrackingGeometry> trackingGeometry = trkGeo_and_DetEls.trackingGeometry;
  if (!trackingGeometry) {
      edm::LogError("ACTSRefitTracksProducer") << "ACTS TrackerGeometry is nullptr!";
      return;  
  }
  //const auto& trackingGeometry = iSetup.getData(trackerGeomToken_); //<- Old
  //const TrackerGeometry& trackingGeometry = iRecord.get(trackerGeomToken_);

  // Validate the mapped material
  myContext myCtx;
  Acts::GeometryContext gCtx;
  Acts::MagneticFieldContext mfCtx;
  myCtx.geoContext = gCtx;
  myCtx.magFieldContext = mfCtx;

  // The MateriaValidator needs a config (MatVal_cfg) which needs a MaterialValidater which needs a IntersectionMaterialAssigner
  // NOTE: we define the IntersectionMaterialAssigner using the same cfg (Mat_Assigner_cfg) used for the mapping
  MaterialSurfaceSelector sel;
  trackingGeometry->visitSurfaces(sel, false);
  std::vector<const Acts::Surface*> map_surfaces = sel.surfaces;

  Acts::IntersectionMaterialAssigner::Config Mat_Assigner_cfg_forValidation;
  Mat_Assigner_cfg_forValidation.surfaces = map_surfaces;

  Acts::MaterialValidater::Config actsVal_cfg;
  actsVal_cfg.materialAssigner = std::make_shared<Acts::IntersectionMaterialAssigner>(Mat_Assigner_cfg_forValidation);  

  MaterialValidation_cfg MatVal_cfg;
  MatVal_cfg.ntracks = Tracks_perEv_; // Numer of tracks per event
  std::unique_ptr<const Acts::Logger> mlogger = Acts::getDefaultLogger("MaterialValidater", Acts::Logging::INFO);
  MatVal_cfg.materialValidater = std::make_shared<Acts::MaterialValidater>(actsVal_cfg, std::move(mlogger));

  MaterialValidator MatVal(MatVal_cfg);

  // Define the writer to save the MaterialTracks:
  MyMaterialTrackWriter_cfg MatTrackWriter_cfg;
  MatTrackWriter_cfg.filePath = "Acts_MatTracks_forValidation.root";
  MyMaterialTrackWriter MatTrackWriter(MatTrackWriter_cfg);

  std::cout << ">>>>> Material Mapper Validation (with " << Nevents_ << " events): " << std::endl; 
  for(int i = 0; i < Nevents_; i++){
    myCtx.EvNumber = i+1;
    std::unordered_map<std::size_t, Acts::RecordedMaterialTrack> tracks = MatVal.execute(myCtx, true);
    //Write the tracks:
    MatTrackWriter.writeT(myCtx, tracks);
  }
  std::cout << std::endl;

  // Close the file:  
  MatTrackWriter.finalize();

}

//DEFINE_FWK_EVENTSETUP_MODULE(MappedMaterialValidator);
DEFINE_FWK_MODULE(MappedMaterialValidator);


