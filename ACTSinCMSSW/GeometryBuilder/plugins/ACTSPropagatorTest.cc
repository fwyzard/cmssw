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
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

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
#include "Acts/Propagator/Navigator.hpp"
#include "Acts/Propagator/RiddersPropagator.hpp"
#include "Acts/Propagator/detail/SteppingLogger.hpp"
#include "Acts/Propagator/EigenStepper.hpp"
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

#include "Acts/Propagator/EigenStepper.hpp"
#include "Acts/MagneticField/NullBField.hpp"

#include <fstream>
#include <iomanip>
#include <random>
#include <nlohmann/json.hpp> 
#include "TChain.h"
#include "TFile.h"

#include "ACTSinCMSSW/GeometryDataFormat/interface/CMSDetectorElement.h"
#include "ACTSinCMSSW/GeometryDataFormat/interface/JsonMaterialWriter.hpp"
#include "ACTSinCMSSW/GeometryDataFormat/interface/CMSMagneticFieldProvider.hpp"
#include "ACTSinCMSSW/GeometryDataFormat/interface/RootPropagationStepsWriter.hpp"

using PropagationOutput = std::pair<PropagationSummary, Acts::RecordedMaterial>;
using PropagationSummaries = std::vector<PropagationSummary>; 
using RandomEngine = std::mt19937;
using RandomSeed = uint32_t;
using DetElVect = std::vector<std::shared_ptr<Acts::CMSDetectorElement>>;

struct TrackingGeometryWithDetEls {
    DetElVect detElements;
    std::shared_ptr<Acts::TrackingGeometry> trackingGeometry;
};

struct myContext{
  /// Magnetic and Geometry contrext
  Acts::GeometryContext geoContext;
  Acts::MagneticFieldContext magFieldContext;
  /// Number of event
  std::size_t EvNumber; 
};

struct PropagationAlgorithm_Config {
  /// Switch the logger to sterile - for timing measurements
  bool sterileLogger = false;
  /// Modify the behavior of the material interaction: energy loss
  bool energyLoss = true;
  /// Modify the behavior of the material interaction: scattering
  bool multipleScattering = true;
  /// Modify the behavior of the material interaction: record
  bool recordMaterialInteractions = true;
  /// looper protection
  double ptLoopers = 500 * Acts::UnitConstants::MeV;
  /// Max step size steering
  double maxStepSize = 5 * Acts::UnitConstants::m;
};

template <typename propagator_t>
class MyConcretePropagator {
 public:
  explicit MyConcretePropagator(propagator_t propagator)
      : m_propagator{std::move(propagator)} {}

  Acts::Result<PropagationOutput> execute(
      const myContext& context, const PropagationAlgorithm_Config& cfg,
      const Acts::Logger& logger,
      const Acts::BoundTrackParameters& startParameters) const {
    ACTS_DEBUG("Test propagation/extrapolation starts");

    PropagationSummary summary(startParameters);
    Acts::RecordedMaterial recordedMaterial; 

    // The step length logger for testing & end of world aborter
    using MaterialInteractor = Acts::MaterialInteractor;
    using SteppingLogger = Acts::detail::SteppingLogger;
    using EndOfWorld = Acts::EndOfWorldReached;

    // Actor list
    using ActorList =
        Acts::ActorList<SteppingLogger, MaterialInteractor, EndOfWorld>;
    using PropagatorOptions =
        typename propagator_t::template Options<ActorList>;

    PropagatorOptions options(context.geoContext, context.magFieldContext);
    // Activate loop protection at some pt value
    options.loopProtection =
        startParameters.transverseMomentum() < cfg.ptLoopers;

    // Switch the material interaction on/off & eventually into logging mode
    auto& mInteractor = options.actorList.template get<MaterialInteractor>();
    mInteractor.multipleScattering = cfg.multipleScattering;
    mInteractor.energyLoss = cfg.energyLoss;
    mInteractor.recordInteractions = cfg.recordMaterialInteractions;

    // Switch the logger to sterile, e.g. for timing checks
    auto& sLogger = options.actorList.template get<SteppingLogger>();
    sLogger.sterile = cfg.sterileLogger;
    // Set a maximum step size
    options.stepping.maxStepSize = cfg.maxStepSize;

    auto state = m_propagator.makeState(options);

    auto resultInit = m_propagator.initialize(state, startParameters);
    if (!resultInit.ok()) {
      return resultInit.error();
    }

    // Propagate using the propagator
    auto resultTmp = m_propagator.propagate(state);
    if (!resultTmp.ok()) {
      return resultTmp.error();
    }

    // Collect internal stepping information
    summary.nStepTrials = state.stepping.nStepTrials;

    auto result =
        m_propagator.makeResult(std::move(state), resultTmp, options, true);
    if (!result.ok()) {
      return result.error();
    }
    auto& resultValue = result.value();

    // Collect general summary information
    summary.nSteps = resultValue.steps;
    summary.pathLength = resultValue.pathLength;

    // Collect the steps
    auto& steppingResults =
        resultValue.template get<SteppingLogger::result_type>();
    summary.steps = std::move(steppingResults.steps);

    summary.statistics = resultValue.statistics;

    // Also set the material recording result - if configured
    if (cfg.recordMaterialInteractions) {
      auto materialResult = resultValue.template get<MaterialInteractor::result_type>();
      recordedMaterial = std::move(materialResult);
    }

    return std::pair{std::move(summary), std::move(recordedMaterial)};
  }

 private:
  propagator_t m_propagator;
};

struct Ranges {
  std::pair<double, double> phiRange;
  std::pair<double, double> etaRange;
  std::pair<double, double> qPRange;
  bool Is_qPFixed = false;
  double qPFixed;
};

std::vector<Acts::BoundTrackParameters> ParticlesGun(double Npart, double EvID, Ranges rngs, Acts::Vector4 pos_init, Acts::ParticleHypothesis p_Type){
  // Create a random number generator:
  RandomSeed seed = 123456789u;
  RandomEngine randomEng(seed + EvID); //PartID is use to increase the "randomness"

  // Setup random distributions:
  std::uniform_real_distribution<double> phiDist(rngs.phiRange.first, rngs.phiRange.second);
  std::uniform_real_distribution<double> etaDist(rngs.etaRange.first, rngs.etaRange.second);
  std::uniform_real_distribution<double> qPDist(rngs.qPRange.first, rngs.qPRange.second);

  // Define the output:
  std::vector<Acts::BoundTrackParameters> TrackParam_vec;

  for(int i = 0; i < Npart; i++){
    // Generate random parameters
    double phi = phiDist(randomEng);
    double eta = etaDist(randomEng);
    double theta = 2 * std::atan(std::exp(-eta));
    Acts::Vector3 direction(std::cos(phi) * std::sin(theta), std::sin(phi) * std::sin(theta), std::cos(theta));
    
    double qP;
    if(rngs.Is_qPFixed) qP = rngs.qPFixed;
    else qP = qPDist(randomEng);

    // Initial parameters: Track position/time four-vector, Track direction three-vector, Charge over Momentum, Curvilinear bound parameters covariance matrix, Particle hypothesis
    Acts::BoundTrackParameters startCurv = Acts::BoundTrackParameters::createCurvilinear(pos_init, direction, qP, std::nullopt, p_Type);
    TrackParam_vec.push_back(startCurv);
  }
  
  return TrackParam_vec;
};

class ACTSPropagatorTest : public edm::one::EDProducer<> {
public:
  explicit ACTSPropagatorTest(const edm::ParameterSet& ps);
  ~ACTSPropagatorTest() override = default;
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

private:
  edm::ESGetToken<TrackingGeometryWithDetEls, ACTSTrackerGeometryRecord> trkGeomInfoToken_;
  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magFieldToken_;

};

ACTSPropagatorTest::ACTSPropagatorTest(const edm::ParameterSet& ps)
    : trkGeomInfoToken_(esConsumes<TrackingGeometryWithDetEls, ACTSTrackerGeometryRecord>()),
      magFieldToken_(esConsumes<MagneticField, IdealMagneticFieldRecord>()) {}

void ACTSPropagatorTest::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) { 

  const MagneticField& magField = iSetup.getData(magFieldToken_);
  const auto& trkGeo_and_DetEls = iSetup.getData(trkGeomInfoToken_);

  DetElVect detEls = trkGeo_and_DetEls.detElements;
  std::shared_ptr<Acts::TrackingGeometry> trackingGeometry = trkGeo_and_DetEls.trackingGeometry;

  myContext myCtx;
  myCtx.geoContext = Acts::GeometryContext{};
  myCtx.magFieldContext = Acts::MagneticFieldContext{};


  // ===== Definition of EigenStepper =====
  auto magFieldPtr = std::make_shared<const CMSMagneticFieldProvider>(magField);
  auto BField_0T = std::make_shared<Acts::NullBField>(); // nullBfield
  Acts::EigenStepper<> es(magFieldPtr);


  GlobalPoint point(0, 0, 0); // cm
  GlobalVector field = magField.inTesla(point);
  std::cout << "B CMSSW -> " << field << std::endl;

  Acts::MagneticFieldProvider::Cache cache;
  auto B_res = magFieldPtr->getField(Acts::Vector3{0,0,0}, cache);
  if(B_res.ok()){
    std::cout << "B ACTS -> " << B_res.value().transpose() / Acts::UnitConstants::T << std::endl;
  }

  // *** NAVIGATOR ***
  Acts::Navigator::Config navi_cfg;
  navi_cfg.trackingGeometry = trackingGeometry;
  std::shared_ptr<const Acts::Logger> navi_logger = Acts::getDefaultLogger("Navigator", Acts::Logging::Level::INFO);
  Acts::Navigator navi(navi_cfg, std::move(navi_logger));
  std::shared_ptr<const Acts::Logger> prop_logger = Acts::getDefaultLogger("Propagator", Acts::Logging::Level::INFO);
  // *** PROPAGATOR ***
  Acts::Propagator prop(es, navi, std::move(prop_logger));

  // Instantiate my concrete propagator:
  MyConcretePropagator my_prop(prop);
  PropagationAlgorithm_Config PropAlg_cfg;
  std::shared_ptr<const Acts::Logger> Myprop_logger = Acts::getDefaultLogger("Concrete Propagator", Acts::Logging::Level::INFO);

  Ranges rngs;
  rngs.phiRange = std::make_pair(-std::numbers::pi, std::numbers::pi);
  rngs.etaRange = std::make_pair(-4, 4);
  rngs.qPRange =  std::make_pair(1. / 1*Acts::UnitConstants::GeV, 1. / 10*Acts::UnitConstants::GeV);
  rngs.Is_qPFixed = true;
  rngs.qPFixed = 1. / 10*Acts::UnitConstants::GeV;
  double N_particle = 10000;
  std::vector<Acts::BoundTrackParameters> Tracks_par = ParticlesGun(N_particle, 1, rngs, Acts::Vector4::Zero(), Acts::ParticleHypothesis::muon());

  PropagationSummaries PropSum_vec;
  std::unordered_map<std::size_t, Acts::RecordedMaterialTrack> RecMat_vec;


  std::cout << ">>>>> Propagator test:" << std::endl; 
  std::size_t P_count = 0;
  for(const auto& this_particle : Tracks_par){
    std::cout << "\rParticle: " << P_count + 1 << " out of " << N_particle << std::flush;
    auto result = my_prop.execute(myCtx, PropAlg_cfg, *Myprop_logger, this_particle);

    PropagationSummary This_PropSum = result.value().first;

    Acts::RecordedMaterial This_RecMat = result.value().second;


    Acts::RecordedMaterialTrack This_RecMatTrack; 

    This_RecMatTrack.first.first = Acts::Vector3::Zero(); // The starting position is 0 for all the particles
    This_RecMatTrack.first.second = this_particle.absoluteMomentum()*this_particle.direction();
    This_RecMatTrack.second = This_RecMat;


    PropSum_vec.push_back(This_PropSum);
    RecMat_vec[P_count] = This_RecMatTrack;
    P_count+=1;

  }

  // Verify the propagarion:
  RootPropagationStepsWriter::Config PropStepWrite_cfg;
  PropStepWrite_cfg.filePath = "Propagator_test.root";
  RootPropagationStepsWriter PropStepWrite(PropStepWrite_cfg);
  PropStepWrite.writeT(1, PropSum_vec);
  PropStepWrite.finalize();

}

DEFINE_FWK_MODULE(ACTSPropagatorTest);


