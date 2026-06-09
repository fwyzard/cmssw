#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "Alignment/TrackerAlignment/interface/TrackerAlignment.h"
#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentRcd.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"

#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "DataFormats/GeometrySurface/interface/RectangularPlaneBounds.h"
#include "DataFormats/GeometrySurface/interface/TrapezoidalPlaneBounds.h"
#include "Geometry/CommonTopologies/interface/Topology.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/CommonTopologies/interface/RadialStripTopology.h"
#include "Geometry/CommonTopologies/interface/RectangularStripTopology.h"
#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"
#include "Geometry/CommonTopologies/interface/CSCRadialStripTopology.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateCombiner.h"

#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "SimDataFormats/Associations/interface/TrackToTrackingParticleAssociator.h"
#include "SimTracker/Common/interface/TrackingParticleSelector.h"

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"

#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/StripClusterParameterEstimator.h"
#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"

#include "Geometry/Records/interface/ACTSTrackerGeometryRecord.h"

#include "Acts/Geometry/TrackingGeometry.hpp"
#include "Acts/MagneticField/MagneticFieldContext.hpp"
#include "Acts/Utilities/CalibrationContext.hpp"
#include "Acts/EventData/TransformationHelpers.hpp"
#include "Acts/EventData/TrackContainer.hpp"
#include "Acts/EventData/VectorTrackContainer.hpp"
#include "Acts/EventData/TrackStateProxy.hpp"
#include "Acts/EventData/TrackProxy.hpp"
#include "Acts/TrackFitting/KalmanFitter.hpp"
#include "Acts/Propagator/RiddersPropagator.hpp"
#include "Acts/Propagator/Navigator.hpp"
#include "Acts/Propagator/detail/SteppingLogger.hpp"
#include "Acts/Propagator/EigenStepper.hpp"
#include "Acts/Surfaces/Surface.hpp"
#include "Acts/Surfaces/PerigeeSurface.hpp"
#include "Acts/Navigation/SurfaceArrayNavigationPolicy.hpp"
#include "Acts/Navigation/TryAllNavigationPolicy.hpp"
#include "Acts/Propagator/Navigator.hpp"
#include "Acts/Propagator/RiddersPropagator.hpp"
#include "Acts/Propagator/detail/SteppingLogger.hpp"
#include "Acts/Propagator/EigenStepper.hpp"
#include "Acts/Utilities/CalibrationContext.hpp"
#include "Acts/TrackFitting/GainMatrixUpdater.hpp"
#include "Acts/Seeding/EstimateTrackParamsFromSeed.hpp"
#include "Acts/TrackFitting/GainMatrixSmoother.hpp"

#include "ACTSinCMSSW/GeometryDataFormat/interface/CMSDetectorElement.h"
#include "ACTSinCMSSW/GeometryDataFormat/interface/CMSKFCalibrator.hpp"
#include "ACTSinCMSSW/GeometryDataFormat/interface/CMSMagneticFieldProvider.hpp"
#include "ACTSinCMSSW/GeometryDataFormat/interface/FromCMSSWtoACTS_hits.hpp"
#include "ACTSinCMSSW/GeometryDataFormat/interface/CMSDetectorSourceLink.hpp"
#include "ACTSinCMSSW/GeometryDataFormat/interface/SurfaceConverters.hpp"
#include "ACTSinCMSSW/GeometryDataFormat/interface/RootPropagationStepsWriter.hpp"

#include "Acts/Visualization/GeometryView3D.hpp"
#include "Acts/Visualization/ObjVisualization3D.hpp"
#include "Acts/Visualization/PlyVisualization3D.hpp"

#include "TFile.h"
#include "TTree.h"
#include "TVector3.h"
#include <Eigen/Core>
#include <algorithm> 

#include <vector>
#include <iostream>
#include <cmath>
#include <cstring> 
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>

// DEBUG: propagate true particle:
struct myContext{
  /// Magnetic and Geometry contrext
  Acts::GeometryContext geoContext;
  Acts::MagneticFieldContext magFieldContext;
  /// Number of event
  std::size_t EvNumber; 
};
using PropagationOutput = std::pair<PropagationSummary, Acts::RecordedMaterial>;
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

// END of DEBUG

using DetElVect = std::vector<std::shared_ptr<Acts::CMSDetectorElement>>;

template <typename T>
Eigen::Matrix<T, 3, 1> makeDirectionFromPhiTheta(T phi, T theta) {
  const T sinTheta{std::sin(theta)};
  return {
      std::cos(phi) * sinTheta,
      std::sin(phi) * sinTheta,
      std::cos(theta),
  };
}

struct TrackingGeometryWithDetEls {
    DetElVect detElements;
    std::shared_ptr<Acts::TrackingGeometry> trackingGeometry;
};

struct LocalHitPosition {    
    float x = -999.f;
    float y = -999.f;
    float x_err = -1.f;
    float y_err = -1.f;
    bool valid = false;
};

LocalHitPosition processHitDEBUG(const TrackingRecHit* hit,
                            const PixelClusterParameterEstimator* pixelCPE,
                            const StripClusterParameterEstimator* stripCPE,
                            const TrackerGeometry* trackerGeometry) {

    // if (dynamic_cast<const SiPixelRecHit*>(hit)) 
    //     std::cout << "[Looking 2 hit type]SiPixelRecHit\n";
    // else if (dynamic_cast<const SiStripRecHit1D*>(hit))
    //     std::cout << "[Looking 2 hit type]SiStripRecHit1D\n";
    // else if (dynamic_cast<const SiStripRecHit2D*>(hit))
    //     std::cout << "[Looking 2 hit type]SiStripRecHit2D\n";
    // else if (dynamic_cast<const SiStripMatchedRecHit2D*>(hit))
    //     std::cout << "[Looking 2 hit type]SiStripMatchedRecHit2D\n";
    // else if (dynamic_cast<const ProjectedSiStripRecHit2D*>(hit))
    //     std::cout << "[Looking 2 hit type]ProjectedSiStripRecHit2D\n";
    // else
    //     std::cout << "UNKNOWN HIT TYPE\n";

    LocalHitPosition result;

    if (!hit || !hit->isValid()) return result;

    const DetId detId = hit->geographicalId();
    const GeomDet* geomDet = trackerGeometry->idToDet(detId);
    if (!geomDet) return result;

    std::optional<std::tuple<LocalPoint, LocalError>> params;

    if (const auto* pixelHit = dynamic_cast<const SiPixelRecHit*>(hit)) {
        auto clusterRef = pixelHit->cluster();
        if (!clusterRef.isNonnull()) return result;

        auto fullParams = pixelCPE->getParameters(*clusterRef, *geomDet);
        params = std::make_tuple(std::get<0>(fullParams), std::get<1>(fullParams));

    } else if (const auto* stripHit1D = dynamic_cast<const SiStripRecHit1D*>(hit)) {
        auto clusterRef = stripHit1D->cluster();
        if (!clusterRef.isNonnull()) return result;

        params = stripCPE->localParameters(*clusterRef, *geomDet);

    } else if (const auto* stripHit2D = dynamic_cast<const SiStripRecHit2D*>(hit)) {
        auto clusterRef = stripHit2D->cluster();
        if (!clusterRef.isNonnull()) return result;

        params = stripCPE->localParameters(*clusterRef, *geomDet);

    } else if (const auto* projHit = dynamic_cast<const ProjectedSiStripRecHit2D*>(hit)) {
        const auto& originalHit = projHit->originalHit();
        auto clusterRef = originalHit.cluster();
        if (!clusterRef.isNonnull()) return result;

        params = stripCPE->localParameters(*clusterRef, *geomDet);
    }

    if (params.has_value()) {
        const LocalPoint& lp = std::get<0>(*params);
        const LocalError& le = std::get<1>(*params);
        result.x = lp.x();
        result.y = lp.y();
        result.x_err = std::sqrt(le.xx());
        result.y_err = std::sqrt(le.yy());
        result.valid = true;
    }

    return result;
}

// void processHitNew(const TrackingRecHit* hit) {

//     const GeomDetUnit& detUnit = *(hit->detUnit());

//     DetId detId = hit->geographicalId();

//     unsigned int IntSubDetID = (detId.subdetId());

//     if (IntSubDetID == StripSubdetector::TIB) {

//         const RectangularStripTopology& topol = dynamic_cast<const RectangularStripTopology&>(detUnit.type().topology());

//         std::cout << "Crossing a TIB layer, do I need to invert the local x? " << std::endl;   
//         std::cout << "DetId:" << detId.rawId() << std::endl;
//         bool isExternal = tTopo.tibIsExternalString(detId);
//         bool isInternal = tTopo.tibIsInternalString(detId);
//         std::cout << "External: " << isExternal << "; Internal: " << isInternal << std::endl;

//         bool isStereo = tTopo.tibIsStereo(detId);
//         bool isDoubleSide = tTopo.tibIsDoubleSide(detId);
//         bool isRPhi = tTopo.tibIsRPhi(detId);
//         std::cout << "is Stereo? " << isStereo << std::endl;
//         std::cout << "is double side? " << isDoubleSide << std::endl;
//         std::cout << "is RPhi? " << isRPhi << std::endl;

//         // if(isExternal) {
//         //     pos.x = -1 * pos.x;
//         // }
//     }

// }

std::ostream& operator<<(std::ostream& os, HitType t) {
    switch (t) {
        case HitType::Hit1D: return os << "Hit1D";
        case HitType::Hit2D: return os << "Hit2D";
        default:             return os << "Unknown";
    }
}
// Function used to save into a json file the CMSSW and ACTS surfaces crossed by a single track
// Global positions of the hit are also saved in a different json file:
// HERE

void SingleTrack3DVisualizer(std::vector<Acts::SourceLink> CMSSW_hits, const auto& reFittedTrack, const TrackerGeometry* TrkGeo) {

    Acts::GeometryContext gCtx;

    // ################
    // # CMSSW FILES: #
    // ################
    Acts::ObjVisualization3D Surf_obj_cmssw;
    Acts::PlyVisualization3D Hits_obj_cmssw;
    Acts::PlyVisualization3D Traj_obj_cmssw;

    // Vector of hits used to draw the trajectory
    std::vector<Acts::Vector3> hits_vec_cmssw;

    /// NOTE: CMSSW_hits is a vector of SourceLink obtaine converting CMSSW hits into ACTS hits
    
    // Loop on all the hits:
    for( auto hit : CMSSW_hits){
        // Collect the surface and the detector element:
        auto surf = CMSSurfaceAccessor(hit);
        auto detEl = static_cast<const Acts::CMSDetectorElement*>(surf->associatedDetectorElement());
        if(detEl) {
            // Save the surface and the global point into the files:
            auto surf = detEl->surface().getSharedPtr();
            Acts::GeometryView3D::drawSurface(Surf_obj_cmssw, *surf, Acts::GeometryContext{}, Acts::Transform3::Identity(), Acts::ViewConfig{});

            auto& cmsDetSL = hit.get<CMSDetectorSourceLink>();
            auto x = cmsDetSL.lPos[0];
            auto y = cmsDetSL.lPos[1];
            auto cov = cmsDetSL.lCov;
            std::cout << "[Hits after conversion] type = " << cmsDetSL.hitType << "; X = " << x << " +- " << cov(0,0) << "; Y = " << y << " +- " << cov(1,1) << std::endl;


            // auto globalPos = surf->localToGlobal(gCtx, cmsDetSL.lPos, Acts::Vector3(0,0,0));
            // auto globalPos = surf->localToGlobal(gCtx, Acts::Vector2{x, y}, Acts::Vector3(0,0,0));


            // TEST: try to obtain the global position from CMSSW localToGlobal conversion:
            SurfaceConverters conv(TrkGeo);
            auto surf_cmssw = conv.fromACTStoCMSSW(*surf);
            LocalPoint lPoint_cmssw(x / 10,y / 10); // from mm to cm 
            GlobalPoint gPoint_cmssw = surf_cmssw->toGlobal(lPoint_cmssw);

            hits_vec_cmssw.push_back(Acts::Vector3{gPoint_cmssw.x() * 10, gPoint_cmssw.y() * 10, gPoint_cmssw.z() * 10}); // from cm to mm
            //hits_vec_cmssw.push_back(globalPos);


            Hits_obj_cmssw.vertex(Acts::Vector3{gPoint_cmssw.x() * 10, gPoint_cmssw.y() * 10, gPoint_cmssw.z() * 10}, {255, 165, 0}); // Orange Color
            //Hits_obj_cmssw.vertex(globalPos, {255, 165, 0}); // Orange Color
        }
    }

    // Draw the CMSSW trajectory
    for(unsigned int i = 0; i < hits_vec_cmssw.size() - 1; i++) {
        Traj_obj_cmssw.line(hits_vec_cmssw[i], hits_vec_cmssw[i+1],  {255, 165, 0}); // Orange Color 
    }

    // Write and close all the files:
    Surf_obj_cmssw.write("ObjFiles/CMSSW_surfaces.obj");
    Surf_obj_cmssw.clear();
    Hits_obj_cmssw.write("ObjFiles/CMSSW_hits.ply");
    Hits_obj_cmssw.clear();
    Traj_obj_cmssw.write("ObjFiles/CMSSW_traj.ply");
    Traj_obj_cmssw.clear();



    // ###############
    // # ACTS FILES: #
    // ###############
    Acts::ObjVisualization3D Surf_obj;
    Acts::ObjVisualization3D Surf_obj_extra;
    Acts::PlyVisualization3D Hits_obj;
    Acts::PlyVisualization3D Traj_obj;

    // Vector of hits used to draw the trajectory
    std::vector<Acts::Vector3> hits_vec;

    // Loop on all the states of the given track
    for(auto trk_state : reFittedTrack.trackStatesReversed()){
        // Check wheter a reference surface is associated to this track state
        if(trk_state.hasReferenceSurface()){
            const Acts::Surface& this_surf = trk_state.referenceSurface();

            // Check wheter this surf is associated to t a detector element
            if(this_surf.associatedDetectorElement() != nullptr){
                //std::cout << "[SingleTrack3DVisualizer] >>>>> Found detElement associated to a surface, saving the surface in an obj file" << std::endl;
                // Save this surface in an obj file:
                Acts::GeometryView3D::drawSurface(Surf_obj, this_surf, Acts::GeometryContext{}, Acts::Transform3::Identity(), Acts::ViewConfig{});
            } 
            // else {
            //     //std::cout << "[SingleTrack3DVisualizer] detElement not found for this surface, moving to the next track state" << std::endl;
            //     Acts::GeometryView3D::drawSurface(Surf_obj_extra, this_surf, Acts::GeometryContext{}, Acts::Transform3::Identity(), Acts::ViewConfig{});
            // }

            // Check wheter this surf has a hit
            if(trk_state.hasCalibrated()){ 
                //std::cout << "[SingleTrack3DVisualizer] >>>>> This detElement has a calibrated hit, saving it into an obj file" << std::endl;

                // Get the local hit
                auto clusterPos = trk_state.template calibrated<2>(); 
                auto cluster_cov = trk_state.template calibratedCovariance<6>(); 
                double x_cluster = clusterPos[0];
                double y_cluster = clusterPos[1];
                std::cout << "[Hits from final track state] X = " << x_cluster << " +- " << cluster_cov(0,0) << "; Y = " << y_cluster << " +- " << cluster_cov(1,1) << std::endl;
                // Convert it into a global hit
                auto globalPos =  this_surf.localToGlobal(gCtx, Acts::Vector2(x_cluster, y_cluster), Acts::Vector3(0,0,0));
                // Save this point into an obj file
                Hits_obj.vertex(globalPos, {0, 255, 0}); // Green Color

                // Add this hit in the vector:
                hits_vec.push_back(globalPos);

            } 
            // else {
            //     std::cout << "[SingleTrack3DVisualizer] >>>>> This detElement has NOT a calibrated hit" << std::endl;
            // }
        } 
        // else {
        //     std::cout << "[SingleTrack3DVisualizer] ERROR: Tracks state not associated to a reference surface!" << std::endl;
        // }
    }

    // Draw the trajectory
    for(unsigned int i = 0; i < hits_vec.size(); i++) {
        Traj_obj.line(hits_vec[i], hits_vec[i+1], {0, 255, 0}); // Green Color
    }

    // Write and close all the files:
    Surf_obj.write("ObjFiles/ACTS_surfaces.obj");
    Surf_obj.clear();
    Surf_obj_extra.write("ObjFiles/ACTS_surfaces_Extra.obj");
    Surf_obj_extra.clear();
    Hits_obj.write("ObjFiles/ACTS_hits.ply");
    Hits_obj.clear();
    Traj_obj.write("ObjFiles/ACTS_traj.ply");
    Traj_obj.clear();
}

TkRotation<float> eigenToTkRotation(const Eigen::Matrix3d& mat) {
    return TkRotation<float>(
        static_cast<float>(mat(0,0)), static_cast<float>(mat(0,1)), static_cast<float>(mat(0,2)),
        static_cast<float>(mat(1,0)), static_cast<float>(mat(1,1)), static_cast<float>(mat(1,2)),
        static_cast<float>(mat(2,0)), static_cast<float>(mat(2,1)), static_cast<float>(mat(2,2))
    );
}

bool areEqual(const TkRotation<float>& a, const TkRotation<float>& b, float eps = 1e-6) {
    return std::fabs(a.xx() - b.xx()) < eps &&
           std::fabs(a.xy() - b.xy()) < eps &&
           std::fabs(a.xz() - b.xz()) < eps &&
           std::fabs(a.yx() - b.yx()) < eps &&
           std::fabs(a.yy() - b.yy()) < eps &&
           std::fabs(a.yz() - b.yz()) < eps &&
           std::fabs(a.zx() - b.zx()) < eps &&
           std::fabs(a.zy() - b.zy()) < eps &&
           std::fabs(a.zz() - b.zz()) < eps;
}


//class ACTSRefitTracksProducerDEBUG : public edm::one::EDAnalyzer<> {
class ACTSRefitTracksProducerDEBUG : public edm::one::EDProducer<> {
public:
  explicit ACTSRefitTracksProducerDEBUG(const edm::ParameterSet&);
  ~ACTSRefitTracksProducerDEBUG() override;

  //void analyze(const edm::Event&, const edm::EventSetup&) override;
  void produce(edm::Event&, const edm::EventSetup&) override;

private:
    edm::ESGetToken<TrackingGeometryWithDetEls, ACTSTrackerGeometryRecord> ACTStrkGeomInfoToken_;  
    edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> trackerGeomToken_;
    const edm::EDGetTokenT<reco::TrackToTrackingParticleAssociator> trackAssociatorToken_; 
    const edm::EDGetTokenT<TrackingParticleCollection> trackingParticleToken_;
    std::vector<edm::InputTag> trackLabels_;
    std::vector<edm::EDGetTokenT<edm::View<reco::Track>>> trackTokens_;

    edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster>> pixelClusterToken_;
    edm::EDGetTokenT<edmNew::DetSetVector<SiStripCluster>> stripClusterToken_;

    // Token to compute the hit position and error given the cluster
    edm::ESGetToken<PixelClusterParameterEstimator, TkPixelCPERecord> cpeToken_;
    edm::ESGetToken<StripClusterParameterEstimator, TkStripCPERecord> stripCpeToken_;

    // Token for the tracking topology
    edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> trackerTopoToken_;

    // Token for the trajectory
    edm::EDGetTokenT<TrajTrackAssociationCollection> trajTrackAssocToken_;

    edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magFieldToken_;

    TrackingParticleSelector tpSelector;

    std::vector<double> P_and_E;
    std::vector<double> vertex;
    double Eta_;
    double Phi_;
    double Mass_;
    double Charge_;
    double ParticleID_;
    double Quality_;

    std::vector<float> local_hit_x_, local_hit_y_, local_hit_x_err_, local_hit_y_err_;
    std::vector<uint32_t> geo_id_;
    std::vector<float> global_hit_x_, global_hit_y_, global_hit_z_;
    std::vector<float> true_mom_x_, true_mom_y_, true_mom_z_;
    std::vector<float> true_vert_x_, true_vert_y_, true_vert_z_;
    int trk_length_;

    std::vector<double> local_x_, local_y_, local_x_err_, local_y_err_;
    std::vector<int> hit_type_;
    std::vector<bool> isInverted_;
    int trk_len_;
    std::vector<uint32_t> mod_id_;
    std::vector<double> pos4_vec;
    std::vector<double> dir3_vec;
    double qoverP_vec;
    std::vector<double> covariance_vec;
    int pdgID;

    int hitIndex = 0;

    // Summary Parameters:
    double TotTrueParticles = 0;
    double TotCMSSW_Matched = 0;
    double TotACTS_reFitted = 0;
    double TotACTS_GoodOoutIndetID = 0;
};




ACTSRefitTracksProducerDEBUG::ACTSRefitTracksProducerDEBUG(const edm::ParameterSet& iConfig)
    : ACTStrkGeomInfoToken_(esConsumes<TrackingGeometryWithDetEls, ACTSTrackerGeometryRecord>()),
      trackerGeomToken_(esConsumes<TrackerGeometry, TrackerDigiGeometryRecord>()),
      trackAssociatorToken_(consumes<reco::TrackToTrackingParticleAssociator>(iConfig.getUntrackedParameter<edm::InputTag>("trackAssociator"))),  
      trackingParticleToken_(consumes<TrackingParticleCollection>(iConfig.getParameter<edm::InputTag>("trackingParticles"))),  
      trackLabels_(iConfig.getParameter<std::vector<edm::InputTag>>("trackLabels")),
      pixelClusterToken_(consumes<edmNew::DetSetVector<SiPixelCluster>>(edm::InputTag("displacedMuonReducedTrackExtras"))),
      stripClusterToken_(consumes<edmNew::DetSetVector<SiStripCluster>>(edm::InputTag("displacedMuonReducedTrackExtras"))),
      cpeToken_(esConsumes<PixelClusterParameterEstimator, TkPixelCPERecord>(edm::ESInputTag("", "PixelCPEGeneric"))),
      stripCpeToken_(esConsumes<StripClusterParameterEstimator, TkStripCPERecord>(edm::ESInputTag("", "StripCPEfromTrackAngle"))),
      trackerTopoToken_(esConsumes<TrackerTopology, TrackerTopologyRcd>()),
      // trajTrackAssocToken_(consumes<TrajTrackAssociationCollection>(edm::InputTag("trajBuilder"))),
      magFieldToken_(esConsumes<MagneticField, IdealMagneticFieldRecord>())
    
  {
    produces<reco::TrackCollection>("recoTracksCollACTS");
    produces<reco::TrackCollection>("recoTracksCollCMSSW");
    produces<reco::TrackExtraCollection>();
    produces<TrackingRecHitCollection>();

    for (const auto& tag : trackLabels_) {
        trackTokens_.push_back(consumes<edm::View<reco::Track>>(tag));
    }

    tpSelector = TrackingParticleSelector(iConfig.getParameter<double>("ptMinTP"),
                                        iConfig.getParameter<double>("ptMaxTP"),
                                        iConfig.getParameter<double>("minRapidityTP"),
                                        iConfig.getParameter<double>("maxRapidityTP"),
                                        iConfig.getParameter<double>("tipTP"),
                                        iConfig.getParameter<double>("lipTP"),
                                        iConfig.getParameter<int>("minHitTP"),
                                        iConfig.getParameter<bool>("signalOnlyTP"),
                                        iConfig.getParameter<bool>("intimeOnlyTP"),
                                        iConfig.getParameter<bool>("chargedOnlyTP"),
                                        iConfig.getParameter<bool>("stableOnlyTP"),
                                        iConfig.getParameter<std::vector<int>>("pdgIdTP"),
                                        iConfig.getParameter<bool>("invertRapidityCutTP"),
                                        iConfig.getParameter<double>("minPhiTP"),
                                        iConfig.getParameter<double>("maxPhiTP"));
}

using Covariance = const Eigen::Ref<const Eigen::Matrix<double,6,6>>; // Eigen::Ref to take both const and non-const matrices
reco::Track::CovarianceMatrix convertCovACTStoCMSSW(std::vector<double> actsParams, std::shared_ptr<const Acts::Surface> pSurface, Covariance cov_acts){

    // Convert the ACTS covariance:
    Eigen::Matrix<double,5,5> Cacts;
    Cacts.setZero();

    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            Cacts(i,j) = cov_acts(i,j);
        }
    }

    double phi    = actsParams[2];
    double theta  = actsParams[3];
    double lambda = M_PI/2. - theta;
    
    auto rot = (pSurface->transform(Acts::GeometryContext{})).rotation();
    Acts::Vector3 u = rot.col(0); // l0 direction
    Acts::Vector3 v = rot.col(1); // l1 direction

    // Define the Jacobian:
    Eigen::Matrix<double,5,5> J = Eigen::Matrix<double,5,5>::Zero();
    // q/p
    J(0, Acts::eBoundQOverP) = 1.0;
    // lambda = pi/2 - theta
    J(1, Acts::eBoundTheta) = -1.0;
    // phi
    J(2, Acts::eBoundPhi) = 1.0;
    // l0, l1 -> dxy
    J(3, Acts::eBoundLoc0) = -std::sin(phi)*u.x() + std::cos(phi)*u.y();
    J(3, Acts::eBoundLoc1) = -std::sin(phi)*v.x() + std::cos(phi)*v.y();
    // l0, l1 -> dsz
    J(4, Acts::eBoundLoc0) = -std::sin(lambda)*(std::cos(phi)*u.x() + std::sin(phi)*u.y()) + std::cos(lambda)*u.z();
    J(4, Acts::eBoundLoc1) = -std::sin(lambda)*(std::cos(phi)*v.x() + std::sin(phi)*v.y()) + std::cos(lambda)*v.z();

    // Apply the Jacobian:
    Eigen::Matrix<double,5,5> cmsCov = J * Cacts * J.transpose();

    // Convert the covariance matrix back:
    reco::Track::CovarianceMatrix cov_cmssw;
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j <= i; ++j) {
            cov_cmssw(i,j) = cmsCov(i,j);
        }
    }

    return cov_cmssw;
}


ACTSRefitTracksProducerDEBUG::~ACTSRefitTracksProducerDEBUG() {
    std::cout << "SUMMARY: " << std::endl;
    std::cout << "Total Number of true tracks: " << TotTrueParticles << std::endl;
    std::cout << "Total Number of matched tracks from CMSSW: " << TotCMSSW_Matched << " (efficiency = " << TotCMSSW_Matched / TotTrueParticles * 100 << "%)" << std::endl;
    std::cout << "Total Number of reFit from ACTS: " << TotACTS_reFitted << std::endl;
    std::cout << "Total Number of tracks with matching Outer and Inner detID: " << TotACTS_GoodOoutIndetID << " (" << TotACTS_GoodOoutIndetID / TotCMSSW_Matched * 100 << "%)" <<std::endl;
}

//void ACTSRefitTracksProducerDEBUG::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
void ACTSRefitTracksProducerDEBUG::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {

    // Get the topology from the iSetup:
    const TrackerTopology& tTopo = iSetup.getData(trackerTopoToken_);

    // Get the trajectory associator:
    // edm::Handle<TrajTrackAssociationCollection> trajTrackAssocHandle;
    // iEvent.getByToken(trajTrackAssocToken_, trajTrackAssocHandle);

    // DEBUG:
    std::string filename = "data.txt";

    std::ifstream infile(filename);
    bool itExists = infile.good();
    infile.close();

    std::ofstream outfile(filename, std::ios::app);

    if(!itExists){
        outfile << "nMeas\tnHoles\tO_match\tI_match\teta\tnHits_cmssw\n";
    }


    bool verbose = true;

    // ===== Define the global hits collection =====
    auto allHitsCollection = std::make_unique<TrackingRecHitCollection>();


    // #############################################################################################
    // # PART I: Find the association Reco <-> Sim track (i.e. Reco Tracks <-> Tracking particles) #
    // # To do so I use the quickTrackAssociatorByHits to make an association map                  #
    // #############################################################################################

    // Define the associator and the TP collection:
    auto const& associatorByHits = iEvent.get(trackAssociatorToken_);  
    auto TPCollectionH = iEvent.getHandle(trackingParticleToken_);  

    edm::Handle<edmNew::DetSetVector<SiPixelCluster>> pixelClusters;
    iEvent.getByToken(pixelClusterToken_, pixelClusters);
    if (!pixelClusters.isValid()) {
        edm::LogWarning("ACTSRefitTracksProducerDEBUG") << "Pixel clusters collection not found!";
    }

    edm::Handle<edmNew::DetSetVector<SiStripCluster>> stripClusters;
    iEvent.getByToken(stripClusterToken_, stripClusters);
    if (!stripClusters.isValid()) {
        edm::LogWarning("ACTSRefitTracksProducerDEBUG") << "Strip clusters collection not found!";
    }

    // Fill the tp collection filtering them using some rules (look at the constructor) 
    TrackingParticleRefVector tpCollection; 
    for (size_t i = 0, size = TPCollectionH->size(); i < size; ++i) {  
        auto tp = TrackingParticleRef(TPCollectionH, i);
        if (tpSelector(*tp)) {
        tpCollection.push_back(tp);
        }
    }

    trk_length_ = 0;
    trk_len_ = 0;

    // ===== Get the tracking geometries =====
    // CMSSW
    const TrackerGeometry* trackerGeometry = &iSetup.getData(trackerGeomToken_);
    if (!trackerGeometry) {
        edm::LogError("ACTSRefitTracksProducerDEBUG") << "CMSSW TrackerGeometry is nullptr!";
        return;  
    }
    // ACTS
    const auto& trkGeo_and_DetEls = iSetup.getData(ACTStrkGeomInfoToken_);
    DetElVect detEls = trkGeo_and_DetEls.detElements;
    std::shared_ptr<Acts::TrackingGeometry> ACTStrackingGeom = trkGeo_and_DetEls.trackingGeometry;
    if (!ACTStrackingGeom) {
        edm::LogError("ACTSRefitTracksProducerDEBUG") << "ACTS TrackerGeometry is nullptr!";
        return;  
    }
    // DEBUG: loop on detEl to get surface info:
    std::ofstream outF("DetEl_Info.txt");
    Acts::GeometryContext local_gCtx;
    for(auto detEl : detEls){
        auto surf = (*detEl).surface().getSharedPtr();
        auto globalPos = surf->localToGlobal(local_gCtx, Acts::Vector2(0, 0), Acts::Vector3(0,0,0));
        auto detId = (*detEl).detID();
        auto subDet = (*detEl).subDetector();
        outF << surf->geometryId() << "; " << detId << "; " << subDet << "; " << globalPos.transpose() << '\n';
    }
    outF.close();

    // DEBUG: Module rotation, traslation:
    std::ofstream outF2("rotationStudy.txt");
    std::ofstream outF3("rotationStudy_2.txt");
    // Loop on all CMS detector elements:
    std::unordered_map<unsigned int, TkRotation<float>> rotMap_cmssw;
    for (const auto& det : trackerGeometry->dets()) {
        auto cmssw_surf = det->surface();
        const auto& pos = cmssw_surf.position();
        const auto& rot = cmssw_surf.rotation();

        auto detID = det->geographicalId().rawId();
        outF2 << "[CMSSW] detId " <<  detID << "; pos: " << pos << "; rot: " << rot << '\n';
        // Filling the map
        rotMap_cmssw[detID] = rot;

    }
    // Loop on all ACTS detector elements:
    for(const auto& det : detEls) {
        auto t = (*det).transform(Acts::GeometryContext{});
        auto detID = (*det).detID();
        auto subDetector = (*det).subDetector();
        
        auto rot_acts = t.rotation();
        outF2 << "[ACTS] detId " <<  detID << "; pos: " << t.translation() << "; rot: " << t.rotation() << '\n';

        // Find the CMSSW rotation associated to this detElement:
        std::vector<int> isEqual;
        auto it = rotMap_cmssw.find(detID);
        if(it != rotMap_cmssw.end()){
            auto rot_cmssw = it->second;
            // for(int i = 0; i < 3; i++){
            //     for(int j = 0; j< 3; j++){
            //         if (std::fabs(rot_cmssw(i,j) - rot_acts(i,j)) > 1e-6) isEqual.push_back(1);
            //     }
            // }
            if (areEqual(rot_cmssw, eigenToTkRotation(rot_acts))){
                outF3 << "[ROTATION STUDY] Rotation matrices for " << detID << " (" << subDetector << ") are equal" << '\n';
            } 
            else {
                outF3 << "[ROTATION STUDY] ERROR: Rotation matrices for " << detID << " (" << subDetector << ") are NOT equal" << '\n';
            }
        } else {
            std::cout << "[ERROR] det id " << detID << " not found in association matrix" << std::endl;
        }

    }
    outF2.close();
    outF3.close();


    // ===== Define the propagator to be used in the KF =====

    const MagneticField& magField = iSetup.getData(magFieldToken_);

    // *** EigenStepper ***
    auto magFieldPtr = std::make_shared<const CMSMagneticFieldProvider>(magField);


    // Acts::MagneticFieldProvider::Cache cache;
    // auto B_res = magFieldPtr->getField(Acts::Vector3{0,0,0}, cache);
    // if(B_res.ok()){
    //     std::cout << "[DEBUG] B ACTS -> " << B_res.value().transpose() / Acts::UnitConstants::T << std::endl;
    // }

    Acts::EigenStepper<> es(magFieldPtr);
    // *** NAVIGATOR ***
    Acts::Navigator::Config navi_cfg;
    navi_cfg.trackingGeometry = ACTStrackingGeom;
    std::shared_ptr<const Acts::Logger> navi_logger = Acts::getDefaultLogger("Navigator", Acts::Logging::Level::INFO);
    Acts::Navigator navi(navi_cfg, std::move(navi_logger));
    std::shared_ptr<const Acts::Logger> prop_logger = Acts::getDefaultLogger("Propagator", Acts::Logging::Level::INFO);
    // *** PROPAGATOR ***
    Acts::Propagator prop(es, navi, std::move(prop_logger));

    // ===== Define the output =====
    /// NOTE: the Track collection for CMSSW is temporary (just to have the same number of entries of ACTS one)
    /// (related to the issue of the inner and outer detId match)
    auto TracksColl = std::make_unique<reco::TrackCollection>();
    auto TracksColl_CMSSW = std::make_unique<reco::TrackCollection>();
    auto ExtraColl  = std::make_unique<reco::TrackExtraCollection>();

    // Loop over all the tracking collection (in my file I have only one collection, called generalTracks, but in principle there could be more)
    for (const auto& trackToken : trackTokens_) { 

        // Get the tracks associated to this collection:
        edm::Handle<edm::View<reco::Track>> tracksHandle;
        iEvent.getByToken(trackToken, tracksHandle);
        const edm::View<reco::Track>& tracks = *tracksHandle;

        // Create a smart pointer (RefToBase) per track and fill all the pointers into a vector:
        edm::RefToBaseVector<reco::Track> trackRefs;
        for (size_t i = 0; i < tracks.size(); ++i) {
            trackRefs.push_back(tracks.refAt(i));
        }

        // Create the association map using the method associateRecoToSim of the associator
        // I give as input the reco track colelction (trackRefs) and the tracking particle collection (tpCollection)
        auto recoToSim = associatorByHits.associateRecoToSim(trackRefs, tpCollection);

        // Loop over all the track pointer
        for (const auto& trackRef : trackRefs) {
            TotTrueParticles += 1;
            // Find the pointer association into the map:
            auto found = recoToSim.find(trackRef);
            // Look for the association:
            if (found != recoToSim.end() && !found->val.empty()) {
                TotCMSSW_Matched += 1;
                const auto& tpPair = found->val.front();
                const TrackingParticleRef& tpRef = tpPair.first;
                math::XYZTLorentzVectorD p4 = tpRef->p4();
                auto V = tpRef->vertex();
                double quality = tpPair.second;

                if (verbose){
                    std::cout << "===== Reco track associated to a Tracking Particle: =====" << std::endl;
                    std::cout << "px: " << p4.px() << "; py: " << p4.py() << "; pz: " << p4.pz() << "; Energy: " << p4.energy() << std::endl;
                    std::cout << "Vertex position -> x: " << V.x()*10 << "; y: " << V.y()*10 << "; z: " << V.z()*10 << std::endl;
                    std::cout << "Eta: " << tpRef->eta() << "; Phi: " << tpRef->phi() << std::endl;
                    std::cout << "Mass: " << tpRef->mass() << "; Charge: " << tpRef->charge() << "; ParticleID: " << tpRef->pdgId() << "; Quality: " << quality << std::endl;
                }

                // if(tpRef->eta() > 0.8 || tpRef->eta() < -0.8){
                //     std::cout << "[WARNING] Eta out of range. Continuing..." << std::endl;
                //     continue;
                // }
                //std::vector<double> p4_V4{p4.px(), p4.py(), p4.pz(), p4.energy()};
                //P_and_E.push_back(p4);
                //std::vector<double> vtx_V3{V.x(), V.y(), V.z()};
                
                P_and_E = {p4.px(), p4.py(), p4.pz(), p4.energy()};
                vertex = {V.x()*10, V.y()*10, V.z()*10};
                Eta_ = tpRef->eta();
                Phi_ = tpRef->phi();
                Mass_ = tpRef->mass();
                Charge_ = tpRef->charge();
                ParticleID_ = tpRef->pdgId();
                Quality_ = quality;

                const reco::Track& recoTrack = *trackRef;
            
                // Needed prpperties: phi, eta, q/P, particle hypotesis, covariance matrix
                Eigen::Vector4d pos4(recoTrack.vertex().x(),  recoTrack.vertex().y(),  recoTrack.vertex().z(), 0.0);
                auto phi = recoTrack.phi();
                auto eta = recoTrack.eta();
                auto theta =  2 * std::atan(std::exp(-eta));
                Eigen::Vector3d dir(std::cos(phi) * std::sin(theta), std::sin(phi) * std::sin(theta), std::cos(theta));
                auto qoverP = recoTrack.qoverp(); 
                auto covariance = recoTrack.covariance();
                int particleID = 0;
                if(found != recoToSim.end() && !found->val.empty()) {
                    const auto& tpPair = found->val.front();
                    const TrackingParticleRef& tpRef = tpPair.first;
                    particleID = tpRef->pdgId();
                }
                pdgID = particleID;

                if(verbose) {
                    std::cout << "==== Parameters of the RECO Track: ==== " << std::endl;
                    std::cout << "4D Position (x, y, z, t): " << pos4[0]*10 << " " << pos4[1]*10 << " " << pos4[2]*10 << " " << pos4[3] << " " << std::endl; // X, Y, Z, T  (from cm to mm)
                    std::cout << "3D Direction: " << dir.x() << " " << dir.y() << " " << dir.z() << std::endl;
                    std::cout << "qoverP: " << qoverP << std::endl;
                    std::cout << "covariance: " << covariance << std::endl;
                    std::cout << "chi2 / ndof = " << recoTrack.chi2() / recoTrack.ndof()<< std::endl;
                    std::cout << "Particle ID = " << pdgID << std::endl;
                }


                pos4_vec = {pos4[0], pos4[1], pos4[2], pos4[3]};
                dir3_vec = {dir.x(), dir.y(), dir.z()};
                qoverP_vec = qoverP;
                covariance_vec.resize(25);
                // vector index = i * N_columns + j
                for (int i = 0; i < 5; ++i) {
                    for (int j = 0; j < 5; ++j) {
                        covariance_vec[i*5 + j] = covariance(i,j);
                    }
                }

                // LOOP OVER THE HITS:
                int h_type = 0;
                const PixelClusterParameterEstimator* pixelCPE = &iSetup.getData(cpeToken_);
                const StripClusterParameterEstimator* stripCPE = &iSetup.getData(stripCpeToken_);

                int hitCount_cmssw = 0;

                // unsigned firstH = hitsCollection->size();
                std::ofstream outF_cmssw("DetEl_CMSSWInfo.txt", std::ios::app);
                std::cout << "CMSSW Hits: " << std::endl;
                for (const auto& hit : recoTrack.recHits()) {

                    // processHitNew(hit);

                    // save the hit in a collection to be pushed than in the event:
                    // hitsCollection->push_back(hit->clone());
                    hitCount_cmssw += 1;
                    DetId detId = hit->geographicalId();
                    const GeomDet* geomDet = trackerGeometry->idToDet(detId);
                    if (!geomDet) continue;

                    unsigned int IntSubDetID = (detId.subdetId());

                    // PixelBarrel = 1; PixelEndcap = 2; TIB = 3; TID = 4; TOB = 5; TEC = 6;
                    h_type = IntSubDetID;

                    LocalHitPosition pos = processHitDEBUG(hit, pixelCPE, stripCPE, trackerGeometry);
                    LocalPoint localP(pos.x, pos.y);
                    const GeomDetUnit& detUnit = *(hit->detUnit());
                    GlobalPoint pos_global = detUnit.surface().toGlobal(localP);

                    double X_mm;
                    double Xerr_mm;
                    double Y_mm; 
                    double Yerr_mm;

                    // Skip dummy modules (?)
                    // std::string topoName = typeid(detUnit.type().topology()).name();
                    // if(topoName.find("DummyTopology") != std::string::npos) continue;

                    if (IntSubDetID == StripSubdetector::TIB){
                        std::cout << "TIB module along track: layer = " << tTopo.tibLayer(detId) 
                                << "; side = " << tTopo.tibSide(detId) 
                                << "; Order = " << tTopo.tibOrder(detId)
                                << "; IsDoubleSide = " << tTopo.tibIsDoubleSide(detId)
                                << "; IsStereo = " << tTopo.tibIsStereo(detId)
                                << "; IsRPhi = " << tTopo.tibIsRPhi(detId)
                                << "; InternalString = " << tTopo.tibIsInternalString(detId)
                                << "; ExternalString = " << tTopo.tibIsExternalString(detId)
                                << "; Glued = " << tTopo.tibGlued(detId) << std::endl;

                        // const RectangularStripTopology& topol = dynamic_cast<const RectangularStripTopology&>(detUnit.type().topology());
                        // float stripLength = topol.localStripLength(hit->localPosition());


                        // X_mm = pos.x * 10;
                        // Xerr_mm = pos.x_err * 10;
                        // Y_mm = (stripLength / 2) * 10;
                        // Yerr_mm = (stripLength * 10) / std::sqrt(12.0f);
                        X_mm = pos.x * 10;
                        Xerr_mm = pos.x_err * 10;
                        Y_mm = pos.y * 10;
                        Yerr_mm = pos.y_err* 10;
                    }
                    else if (IntSubDetID == StripSubdetector::TOB){
                        std::cout << "TOB module along track: layer = " << tTopo.tobLayer(detId) 
                                << "; side = " << tTopo.tobSide(detId)
                                << "; Rod = " << tTopo.tobRod(detId)
                                << "; IsDoubleSide = " << tTopo.tobIsDoubleSide(detId)
                                << "; IsStereo = " << tTopo.tobIsStereo(detId)
                                << "; IsRPhi = " << tTopo.tobIsRPhi(detId)
                                << "; Glued = " << tTopo.tobGlued(detId) << std::endl;

                        // const RectangularStripTopology& topol = dynamic_cast<const RectangularStripTopology&>(detUnit.type().topology());
                        // float stripLength = topol.localStripLength(hit->localPosition());

                        // X_mm = pos.x * 10;
                        // Xerr_mm = pos.x_err * 10;
                        // Y_mm = (stripLength / 2) * 10;
                        // Yerr_mm = (stripLength * 10) / std::sqrt(12.0f);
                        X_mm = pos.x * 10;
                        Xerr_mm = pos.x_err * 10;
                        Y_mm = pos.y * 10;
                        Yerr_mm = pos.y_err* 10;
                    }
                    else {
                        X_mm = pos.x * 10;
                        Xerr_mm = pos.x_err * 10;
                        Y_mm = pos.y * 10;
                        Yerr_mm = pos.y_err* 10;
                    }

                    // if (IntSubDetID == StripSubdetector::TIB || IntSubDetID == StripSubdetector::TOB) {

                    //     std::cout << "Before the cast. ID " << detId.rawId() << " Topology real type: " << typeid(detUnit.type().topology()).name() << std::endl;

                    //     const RectangularStripTopology& topol = dynamic_cast<const RectangularStripTopology&>(detUnit.type().topology());

                    //     float stripLength = topol.localStripLength(hit->localPosition());
                    //     std::cout << "After cast" << std::endl;

                    //     // std::cout << "Strip length: " << stripLength << std::endl;
                    //     X_mm = pos.x * 10;
                    //     Xerr_mm = pos.x_err * 10;
                    //     Y_mm = (stripLength / 2) * 10;
                    //     Yerr_mm = stripLength / std::sqrt(12.0f);

                    //     if (IntSubDetID == StripSubdetector::TIB) {
                    //         std::cout << "TIB HIT" << std::endl;
                    //         bool isStereo = tTopo.tibIsStereo(detId);
                    //         bool isDoubleSide = tTopo.tibIsDoubleSide(detId);
                    //         bool isRPhi = tTopo.tibIsRPhi(detId);
                    //         std::cout << "================================"<< std::endl;
                    //         std::cout << "is Stereo? " << isStereo << std::endl;
                    //         std::cout << "is isRPhi? " << isRPhi << std::endl;
                    //         std::cout << "is Double Side? " << isDoubleSide << std::endl;
                    //         std::cout << "Local Position: X-> " << pos.x << "; Y-> " << pos.y << std::endl;
                    //         std::cout << "================================"<< std::endl;

                    //         auto stereoSurf = detUnit.surface();
                    //         auto R = stereoSurf.rotation();
                    //         std::cout << "Z ELEMENT : " << R.zz() << std::endl;
                    //         // if(R.zz() < 0) {
                    //         //     X_mm = X_mm;
                    //         //     Y_mm = -1 * Y_mm;
                    //         //     std::cout << "[CORRECTED TIB HITS] X = " << X_mm << "; Y = " << Y_mm << std::endl;
                    //         // }

                    //         auto X_corrected = X_mm;
                    //         auto Y_corrected = Y_mm;
                    //         std::cout << "[TIB HIT RAW] X = " << X_mm << "; Y = " << Y_mm << std::endl;
                    //         // if(isStereo) {
                    //         //     std::cout << "Ruoto" << std::endl; 
                    //         //     X_corrected = R.xx()*X_mm + R.yx()*X_mm;
                    //         //     Y_corrected = R.xy()*X_mm + R.yy()*Y_mm;
                    //         //     X_mm = X_corrected;
                    //         //     Y_mm = Y_corrected;
                    //         // }



                    //         GlobalVector n = detUnit.surface().toGlobal(LocalVector(0,0,1));
                    //         GlobalPoint center = detUnit.surface().position();
                    //         GlobalVector radial = GlobalVector(center.x(), center.y(), center.z());
                    //         GlobalVector rhat = radial.unit(); 
                    //         // COnfronto con il versore radiale uscente:
                    //         bool isFlipped = (n.dot(rhat) < 0);

                    //         //if(isFlipped){
                    //         if(R.zz() < 0){
                    //             X_mm = X_mm;
                    //             Y_mm = -1 * Y_mm;
                    //         }

                    //         std::cout << "[TIB HIT MOD] X = " << X_mm << "; Y = " << Y_mm << std::endl;

                    //         // funziona:
                    //         // auto X_corrected = X_mm;
                    //         // auto Y_corrected = Y_mm;
                    //         // if(R.zz() < 0){
                    //         //     X_mm = X_corrected;
                    //         //     Y_mm = -1 * Y_corrected;
                    //         // } 
                    //         // else {
                    //         //     X_mm = X_corrected;
                    //         //     Y_mm = Y_corrected;
                    //         // }
      
                    //     }


                    //     if (IntSubDetID == StripSubdetector::TOB) {
                    //         std::cout << "TOB HIT" << std::endl;
                    //         bool isStereo = tTopo.tobIsStereo(detId);
                    //         bool isDoubleSide = tTopo.tobIsDoubleSide(detId);
                    //         bool isRPhi = tTopo.tobIsRPhi(detId);
                    //         std::cout << "================================"<< std::endl;
                    //         std::cout << "is Stereo? " << isStereo << std::endl;
                    //         std::cout << "is isRPhi? " << isRPhi << std::endl;
                    //         std::cout << "is Double Side? " << isDoubleSide << std::endl;
                    //         std::cout << "Local Position: X-> " << pos.x << "; Y-> " << pos.y << std::endl;
                    //         std::cout << "================================"<< std::endl;
                    //         // If stereo, apply the correction:
                    //         auto stereoSurf = detUnit.surface();
                    //         auto R = stereoSurf.rotation();

                    //         GlobalVector n = detUnit.surface().toGlobal(LocalVector(0,0,1));
                    //         GlobalPoint center = detUnit.surface().position();
                    //         GlobalVector radial = GlobalVector(center.x(), center.y(), center.z());
                    //         GlobalVector rhat = radial.unit(); 
                    //         // COnfronto con il versore radiale uscente:
                    //         bool isFlipped = (n.dot(rhat) < 0);


                    //         auto X_corrected = X_mm;
                    //         auto Y_corrected = Y_mm;
                            
                    //         //if(isFlipped){
                    //         if(R.zz() < 0){
                    //             X_mm = -1 * X_corrected;
                    //             Y_mm = -1 * Y_corrected;
                    //             if(isStereo) {
                    //                 X_corrected = R.xx()*X_mm + R.yx()*X_mm;
                    //                 Y_corrected = R.xy()*X_mm + R.yy()*Y_mm;
                    //                 X_mm = X_corrected;
                    //                 Y_mm = Y_corrected;
                    //             }
                    //         } 
                    //         else {
                    //             if(isStereo) {
                    //                 X_corrected = R.xx()*X_mm + R.yx()*X_mm;
                    //                 Y_corrected = R.xy()*X_mm + R.yy()*Y_mm;
                    //             }
                    //             X_mm = X_corrected;
                    //             Y_mm = Y_corrected;
                    //         }





                    //         // if(isStereo) {
                    //         //     auto X_corrected = R.xx()*X_mm + R.yx()*X_mm;
                    //         //     auto Y_corrected = R.xy()*X_mm + R.yy()*Y_mm;
                                    
                    //         //     if(R.zz() < 0){
                    //         //         X_mm = -1 * X_corrected;
                    //         //         Y_mm = Y_corrected;
                    //         //     }
                    //         //     else {
                    //         //         X_mm = X_corrected;
                    //         //         Y_mm = Y_corrected;
                    //         //     }
                    //         //     std::cout << "[CORRECTED TOB HITS] X = " << X_mm << "; Y = " << Y_mm << std::endl;
                    //         // }
                    //     }
                        


                    //     // LocalPoint tib_hit_cmssw(X_mm / 10 ,Y_mm / 10);
                    //     // Acts::Vector2 tib_hit_acts = Acts::Vector2{X_mm,Y_mm};
                    //     // auto cmssw_surf = detUnit.surface();
                    //     // SurfaceConverters surfConv(trackerGeometry);
                    //     // auto acts_surf = surfConv.fromCMSSWtoACTS(cmssw_surf);

                    //     // auto gPos_cmssw = cmssw_surf.toGlobal(tib_hit_cmssw);
                    //     // auto gPos_acts  = acts_surf->localToGlobal(Acts::GeometryContext{},tib_hit_acts, Acts::Vector3(0,0,0));

                    //     // std::cout << "================================"<< std::endl;
                    //     // std::cout << "ACTS local point:   X = " << tib_hit_acts[0] << "; Y = " << tib_hit_acts[1] << std::endl;
                    //     // std::cout << "ACTS global point:  X = " << gPos_acts[0] << "; Y = " << gPos_acts[1] << "; Z = " << gPos_acts[2] << std::endl;
                    //     // std::cout << "CMSSW local point:  X = " << tib_hit_cmssw.x() * 10 << "; Y = " << tib_hit_cmssw.y() * 10 << std::endl;
                    //     // std::cout << "CMSSW global point: X = " << gPos_cmssw.x() * 10 << "; Y = " << gPos_cmssw.y() * 10 << "; Z = " << gPos_cmssw.z() * 10 << std::endl;   
                    //     // std::cout << "================================"<< std::endl;




                    // }
                    // else if (IntSubDetID == StripSubdetector::TID) {
                    //     std::cout << "Before the cast. TID ID " << detId.rawId() << " Topology real type: " << typeid(detUnit.type().topology()).name() << std::endl;

                    //     const RadialStripTopology& topol = dynamic_cast<const RadialStripTopology&>(detUnit.type().topology());
                    //     float stripLength = topol.localStripLength(hit->localPosition());
                    //     std::cout << "After cast" << std::endl;

                    //     X_mm = pos.x * 10;
                    //     Xerr_mm = pos.x_err * 10;
                    //     Y_mm = (stripLength / 2) * 10;
                    //     Yerr_mm = stripLength / std::sqrt(12.0f);

                    //     GlobalVector n = detUnit.surface().toGlobal(LocalVector(0,0,1));
                    //     bool isFlipped = (n.dot(GlobalVector(0,0,1)) < 0);

                    //     if(tTopo.tidIsZMinusSide(detId)){
                    //         std::cout << "TID minus " << std::endl;
                    //         // If it's negative Z, module is flipped if n dot (0,0,1) < 0
                    //         if(isFlipped){
                    //             std::cout << "     Flipping" << std::endl;
                    //             X_mm = X_mm;
                    //             Y_mm = -1 * Y_mm;
                    //         }
                    //     }
                    //     else {
                    //         std::cout << "TID Plus " << std::endl;
                    //         if(!isFlipped){
                    //             std::cout << "     Flipping" << std::endl;
                    //             X_mm = X_mm;
                    //             Y_mm = -1 * Y_mm;
                    //         }
                    //     }

                    //     bool isBackRing = tTopo.tidIsBackRing(detId);
                    //     std::cout << "================================"<< std::endl;
                    //     std::cout << "is sBackRing? " << isBackRing << std::endl;
                    //     std::cout << "Local Position: X-> " << pos.x << "; Y-> " << pos.y << std::endl;
                    //     std::cout << "================================"<< std::endl;


                    // }
                    // else if (IntSubDetID == StripSubdetector::TEC) {
                    //     std::cout << "Before the cast. TEC ID " << detId.rawId() << " Topology real type: " << typeid(detUnit.type().topology()).name() << std::endl;

                    //     const RadialStripTopology& topol = dynamic_cast<const RadialStripTopology&>(detUnit.type().topology());
                    //     float stripLength = topol.localStripLength(hit->localPosition());
                    //     std::cout << "After cast" << std::endl;

                    //     bool isStereo = tTopo.tecIsStereo(detId);

                    //     X_mm = pos.x * 10;
                    //     Xerr_mm = pos.x_err * 10;
                    //     Y_mm = (stripLength / 2) * 10;
                    //     Yerr_mm = stripLength / std::sqrt(12.0f);

                    //     GlobalVector n = detUnit.surface().toGlobal(LocalVector(0,0,1));
                    //     bool isFlipped = (n.dot(GlobalVector(0,0,1)) < 0);

                    //     auto stereoSurf = detUnit.surface();
                    //     auto R = stereoSurf.rotation();
                    //     auto X_corrected = X_mm;
                    //     auto Y_corrected = Y_mm;

                    //     if(tTopo.tecIsZMinusSide(detId)){
                    //         std::cout << "TEC minus " << std::endl;
                    //         // If it's negative Z, module is flipped if n dot (0,0,1) < 0
                    //         if(isFlipped){
                    //             std::cout << "     Flipping" << std::endl;
                    //             X_mm = -1 * X_mm;
                    //             Y_mm = -1 * Y_mm;
                    //             if(isStereo) {
                    //                 X_corrected = R.xx()*X_mm + R.yx()*X_mm;
                    //                 Y_corrected = R.xy()*X_mm + R.yy()*Y_mm;
                    //                 X_mm = X_corrected;
                    //                 Y_mm = Y_corrected;
                    //             }
                    //         } 
                    //         else {
                    //             if(isStereo) {
                    //                 X_corrected = R.xx()*X_mm + R.yx()*X_mm;
                    //                 Y_corrected = R.xy()*X_mm + R.yy()*Y_mm;
                    //             }
                    //             X_mm = X_corrected;
                    //             Y_mm = Y_corrected;
                    //         }
                    //     }
                    //     else {
                    //         std::cout << "TEC Plus " << std::endl;
                    //         if(!isFlipped){
                    //             std::cout << "     Flipping" << std::endl;
                    //             X_mm = -1 * X_mm;
                    //             Y_mm = -1 * Y_mm;
                    //             if(isStereo) {
                    //                 X_corrected = R.xx()*X_mm + R.yx()*X_mm;
                    //                 Y_corrected = R.xy()*X_mm + R.yy()*Y_mm;
                    //                 X_mm = X_corrected;
                    //                 Y_mm = Y_corrected;
                    //             }
                    //         }
                    //         else {
                    //             if(isStereo) {
                    //                 X_corrected = R.xx()*X_mm + R.yx()*X_mm;
                    //                 Y_corrected = R.xy()*X_mm + R.yy()*Y_mm;
                    //             }
                    //             X_mm = X_corrected;
                    //             Y_mm = Y_corrected;
                    //         }
                    //     }

                    // }              
                    // else {
                    //     X_mm = pos.x * 10;
                    //     Xerr_mm = pos.x_err * 10;
                    //     Y_mm = pos.y * 10;
                    //     Yerr_mm = pos.y_err* 10;
                    // }

                    std::cout << "[Raw Hits before conversion] type = " << h_type << "; X = " << X_mm << " +- " << Xerr_mm << "; Y = " << Y_mm << " +- " << Yerr_mm << std::endl;

                    // Fitting:
                    if (pos.valid){
                        // if(h_type == 1 ) std::cout << "[DEBUG] PixelBarrel Hit: " << std::endl;
                        // if(h_type == 2 ) std::cout << "[DEBUG] PixelEndcap Hit: " << std::endl;
                        // if(h_type == 3 ) std::cout << "[DEBUG] TIB Hit: " << std::endl;
                        // if(h_type == 4 ) std::cout << "[DEBUG] TID Hit: " << std::endl;
                        // if(h_type == 5 ) std::cout << "[DEBUG] TOB Hit: " << std::endl;
                        // if(h_type == 6 ) std::cout << "[DEBUG] TEC Hit: " << std::endl;

                        // std::cout << "[DEBUG] Hit local pos: (" << pos.x*10 << ", " << pos.y*10 << ") ± (" << pos.x_err*10 << ", " << pos.y_err*10 << ")" << std::endl;
                        // std::cout << "[CMSSW] Hit local pos: (" << pos.x*10 << ", " << pos.y*10 << ")" << std::endl;
                        hit_type_.push_back(h_type);
                        local_x_.push_back(X_mm); 
                        local_y_.push_back(Y_mm);
                        local_x_err_.push_back(Xerr_mm);
                        local_y_err_.push_back(Yerr_mm);
                        mod_id_.push_back(detId.rawId());
                        // std::cout << "[CMSSW] Hit local pos: (" << X_mm << " +- " << Xerr_mm << ") (" << Y_mm << " +- " << Yerr_mm << ")" << std::endl;
                        // std::cout << "DetID from CMSSW: " << detId.rawId() << std::endl;
                        outF_cmssw << detId.rawId() << "; " << h_type << "; " << pos_global << '\n';
                    } else {
                        std::cout << "ERROR: Invalid Hit. Skipping." << std::endl;
                    }
                    
                }
                outF_cmssw << "-----\n";
                outF_cmssw.close();

                // edm::RefProd<TrackingRecHitCollection> recHitsRefProd = recHitsHandle.refProd();
                // unsigned nH = rHits->size();


                // ===== Convert CMSSW hits into ACTS hits =====
                // std::cout << "Converting from CMSSW hits fo ACTS hits..." << std::endl;
                FromCMSSWtoACTS_hits::HitsInfo hitsInfo;
                hitsInfo.hit_type  = hit_type_;
                hitsInfo.loc_x     = local_x_;
                hitsInfo.loc_y     = local_y_;
                hitsInfo.loc_x_err = local_x_err_;
                hitsInfo.loc_y_err = local_y_err_;
                hitsInfo.mod_ids   = mod_id_;
                FromCMSSWtoACTS_hits CMSSWtoACTS_converter(hitsInfo, detEls);
                std::vector<Acts::SourceLink> ACTS_hits = CMSSWtoACTS_converter.convert();


                // DEBUG
                std::ofstream outF("DetEl_ACTSInfo_AfterConversion.txt", std::ios::app);
                // std::cout << "ACTS Hits after conversion: " << std::endl;
                for( auto hit : ACTS_hits){
                    auto surf = CMSSurfaceAccessor(hit);
                    auto detEl = static_cast<const Acts::CMSDetectorElement*>(surf->associatedDetectorElement());
                    if(detEl) {
                        auto this_ID = detEl->detID();
                        auto subDet = detEl->subDetector();
                        auto surf = detEl->surface().getSharedPtr();
                        auto globalPos = surf->localToGlobal(local_gCtx, Acts::Vector2(0, 0), Acts::Vector3(0,0,0));

                        // PRINT HERE THE NRM TO THE ACTS SURFACE AND THE ID:
                        auto nrm_acts = surf->normal(Acts::GeometryContext{}, surf->center(Acts::GeometryContext{}), Acts::Vector3{0,0,0});
                        std::cout << "[ACTS] Normal to module " << this_ID << ": " << nrm_acts.transpose() << std::endl;
        
                        // std::cout << "[DEBUG] ACTS detId before the fit " << this_ID <<"; subDet = " << subDet << std::endl;
                        outF << surf->geometryId() << "; " << this_ID << "; " << subDet << "; " << globalPos.transpose() << '\n';

                        auto& cmsDetSL = hit.get<CMSDetectorSourceLink>();

                        // std::cout << "[ACTS] cmsDetSL.lPos = " << cmsDetSL.lPos.transpose() << std::endl; //"; Error: " << cmsDetSL.lCov.transpose() << std::endl;

                    }
                }
                outF << "-----\n";
                outF.close();
                // DEBUG



                // std::cout << ">>>>> Number of source links: " << ACTS_hits.size() << std::endl;
                // std::cout << ">>>>> Number Rec Hits: " << hitCount_cmssw << std::endl;
                hit_type_.clear();
                local_x_.clear();
                local_y_.clear();
                local_x_err_.clear();
                local_y_err_.clear();
                mod_id_.clear();


                // ===== Prepare to perform the refit on ACTS hits =====
                // Obtain the initial parameters starting from TP params
                Acts::GeometryContext gCtx;
                Acts::MagneticFieldContext mfCtx;
                auto pSurface = Acts::Surface::makeShared<Acts::PerigeeSurface>(Acts::Vector3( V.x()*10,  V.y()*10,  V.z()*10));
                Acts::Vector4 tp_pos4_acts = Acts::Vector4(V.x()*10, V.y()*10, V.z()*10, 0);
                Acts::Vector3 tp_dir_acts = Acts::Vector3(p4.px(), p4.py(), p4.pz());
                double tp_qOverP_acts = tpRef->charge() / std::sqrt(p4.px()*p4.px() + p4.py()*p4.py() + p4.pz()*p4.pz());
                auto tp_pType =  Acts::ParticleHypothesis::muon();
                Acts::BoundMatrix tp_cov;

                Acts::Result<Acts::BoundVector> trans_res = Acts::transformFreeToBoundParameters(Acts::Vector3(V.x()*10, V.y()*10, V.z()*10), 0, tp_dir_acts, tp_qOverP_acts, *pSurface, gCtx);
                if(trans_res.ok()){ // (false)
                    auto bvector = trans_res.value();
                    Acts::EstimateTrackParamCovarianceConfig EstCov_cfg;
                    tp_cov = Acts::estimateTrackParamCovariance(EstCov_cfg, bvector, true);
                }
                else{
                    std::cout << "ERROR: Cannot extimate covariance matrix since transformFreeToBoundParameters failed" << std::endl;
                    std::cout << "       Continue with the diagonal one" << std::endl;
                    tp_cov = Acts::BoundSquareMatrix::Identity()*1e-3;//*1e5;
                }

                //tp_cov = Acts::BoundSquareMatrix::Identity()*1e-3;//*1e5;

                auto start_param_res = Acts::BoundTrackParameters::create(gCtx, pSurface, tp_pos4_acts, tp_dir_acts, tp_qOverP_acts, tp_cov, tp_pType);
                Acts::BoundTrackParameters start_param = *start_param_res;

                // DEBUG: Propagate the true particle 
                Acts::PlyVisualization3D Hits_obj;
                Acts::PlyVisualization3D Traj_obj;
                MyConcretePropagator my_prop(prop);
                PropagationAlgorithm_Config PropAlg_cfg;
                myContext myCtx;
                myCtx.geoContext = Acts::GeometryContext{};
                myCtx.magFieldContext = Acts::MagneticFieldContext{};
                std::shared_ptr<const Acts::Logger> Myprop_logger = Acts::getDefaultLogger("Concrete Propagator", Acts::Logging::Level::INFO);
                auto result = my_prop.execute(myCtx, PropAlg_cfg, *Myprop_logger, start_param);
                PropagationSummary PropSum = result.value().first;
                std::vector<Acts::Vector3> hits_vec;
                for(const auto& step : PropSum.steps){
                    Hits_obj.vertex(Acts::Vector3{step.position[0], step.position[1], step.position[2]}, {255, 165, 0}); 
                    hits_vec.push_back(Acts::Vector3{step.position[0], step.position[1], step.position[2]});
                }
                for(unsigned int i = 0; i < hits_vec.size() - 1; i++) {
                    Traj_obj.line(hits_vec[i], hits_vec[i+1],  {255, 165, 0}); 
                }
                Hits_obj.write("ObjFiles/TP_hits.ply");
                Hits_obj.clear();
                Traj_obj.write("ObjFiles/TP_traj.ply");
                Traj_obj.clear();
                // END of DEBUG


                // ===== Define the KF  class =====
                //  What's needed:
                // - propagator
                // - logger
                using Propagator_t = Acts::Propagator<Acts::EigenStepper<>, Acts::Navigator>;
                using Trajectory_t = Acts::VectorMultiTrajectory;
                using TrackBackend = Acts::VectorTrackContainer;
                using TrajectoryBackend = Acts::VectorMultiTrajectory;
                using TrackContainer_t = Acts::TrackContainer<TrackBackend, TrajectoryBackend>;
                std::unique_ptr<const Acts::Logger> KF_logger = Acts::getDefaultLogger("KalmanFitter", Acts::Logging::INFO);
                Acts::KalmanFitter<Propagator_t, Trajectory_t> Kfitter(prop, std::move(KF_logger));

                // ===== KalmanFitterExtensions ===== (needed by KalmanFilterOptions)
                Acts::KalmanFitterExtensions<Trajectory_t> extensions;
                extensions.surfaceAccessor.connect<&CMSSurfaceAccessor>();               // How to obtain the surface from the SourceLink 
                extensions.calibrator.template connect<&CMSKFCalibrator<Trajectory_t>>(); // Shows how to unpack the raw hit info given a SourceLink

               
               
                Acts::GainMatrixUpdater kfUpdater;
                Acts::GainMatrixSmoother kfSmoother;
                extensions.updater.connect<&Acts::GainMatrixUpdater::operator()<Trajectory_t>>(&kfUpdater);
                extensions.smoother.connect<&Acts::GainMatrixSmoother::operator()<Trajectory_t>>(&kfSmoother);

                // ===== KalmanFilterOptions ===== 
                Acts::CalibrationContext calibContext; 
                Acts::PropagatorPlainOptions pOptions(gCtx, mfCtx);
                pOptions.surfaceTolerance = 1e-4;

                // ==== TrackContainer =====
                // Stores the info of the track and trajectory step by step 
                using TrackBackend = Acts::VectorTrackContainer;
                using TrajectoryBackend = Acts::VectorMultiTrajectory;
                TrackBackend trk_bkn;
                TrajectoryBackend trj_bkn;
                TrackContainer_t trk_container(trk_bkn, trj_bkn);

                // == Perform the real re-fit =====
                Acts::KalmanFitterOptions<Trajectory_t> kfOptions(gCtx, mfCtx, std::cref(calibContext), std::move(extensions), pOptions, pSurface.get());
                auto fit_result = Kfitter.fit(ACTS_hits.begin(), ACTS_hits.end(), start_param, kfOptions, trk_container);

                if (fit_result.ok()) {

                    // HERE
                    // Uncomment only then you analyze ONE track
                    // std::cout << "Saving all the surface into an obj file" << std::endl;
                    SingleTrack3DVisualizer(ACTS_hits, fit_result.value(), trackerGeometry);

                    TotACTS_reFitted += 1;
                    
                    Acts::TrackProxy trackProxy = fit_result.value();

                    Acts::Vector3 fitted_dir = trackProxy.direction();   
                    double loc0 = trackProxy.loc0();
                    double loc1 = trackProxy.loc1();
                    Acts::Vector3 fitted_pos = pSurface->localToGlobal(gCtx, Acts::Vector2(loc0, loc1), Acts::Vector3(0,0,0));  //Acts::Vector3(0,0,0)
                    double theta = trackProxy.theta();
                    double phi = trackProxy.phi();
                    double qOverP = trackProxy.qOverP();
                    auto fitted_cov = trackProxy.covariance();
                    //std::vector<double> actsParams = {loc0, loc1, phi, theta, qOverP}; 
                    // Eigen::Matrix<double,6,6> cov_copy = trackProxy.covariance().eval();
                    // auto fitted_cov_converted = convertCovACTStoCMSSW({loc0, loc1, phi, theta, qOverP}, pSurface, cov_copy);
                    // std::cout << "Fitted cov: " << fitted_cov << std::endl;
                    // std::cout << "Fitted cov (converted): " << fitted_cov_converted << std::endl;

                    if(verbose){
                        std::cerr << "#FIT successfull#" << std::endl;
                        std::cout << ">>>> Fitted parameters:" << std::endl;
                        std::cout << "Fitted global position: x = " << fitted_pos.transpose()[0] << " y = " << fitted_pos.transpose()[1] << " z = " << fitted_pos.transpose()[2] << " t = 0" << std::endl;
                        std::cout << "Fitted direction (normalized): x = " << fitted_dir.transpose()[0] << " y = " << fitted_dir.transpose()[1] << " z = " << fitted_dir.transpose()[2] << std::endl;
                        std::cout << "loc0 = " << loc0 << std::endl;
                        std::cout << "loc1 = " << loc1 << std::endl;
                        std::cout << "phi = " << phi << std::endl;
                        std::cout << "theta = " << theta << std::endl;
                        std::cout << "q Over P = " << qOverP << std::endl;
                        std::cout << "covariance = " << std::endl;
                        std::cout << fitted_cov << std::endl;
                    }

                    // ===== Construct the reco::Track =====
                    auto chi2 = static_cast<double>(trackProxy.chi2());
                    auto ndof = static_cast<double>(trackProxy.nDoF());
                    reco::Track::Point refPoint(fitted_pos.transpose()[0] / 10, fitted_pos.transpose()[1] / 10, fitted_pos.transpose()[2] / 10); // from mm to cm 
                    auto P = abs(1 / qOverP);
                    reco::Track::Vector Momentum(P*fitted_dir.transpose()[0], P*fitted_dir.transpose()[1], P*fitted_dir.transpose()[2]);
                    int charge = (qOverP > 0) ? +1 : -1;
                    reco::Track::CovarianceMatrix cov5;
                    constexpr int dimTrack = reco::Track::dimension;

                    for (int i = 0; i < dimTrack; ++i) {
                        for (int j = 0; j <= i; ++j) { 
                            cov5(i,j) = fitted_cov(i,j);  
                        }
                    }

                    reco::Track recoTrack_acts(chi2, ndof, refPoint, Momentum, charge, cov5);


                    // ===== Define the Track Extras associated to this recoTrack =====
                    // Take seedDir, seedRef, outerId and innerID from the Track State of the CMSSW reco track
                    auto trackExtra_cmssw = recoTrack.extra();
                    unsigned int outerId_cmssw = 0;
                    unsigned int innerId_cmssw = 0;
                    unsigned int firstRecHit = 0, recHitsSize = 0;
                    std::unique_ptr<TrackingRecHitCollection> hitCollection;
                    PropagationDirection seedDir_cmssw{};
                    edm::RefToBase<TrajectorySeed> seedRef_cmssw;
                    if (trackExtra_cmssw.isNonnull()) {
                        const reco::TrackExtra& trackExtra = *trackExtra_cmssw;
                        seedDir_cmssw = trackExtra.seedDirection();
                        seedRef_cmssw = trackExtra.seedRef();
                        innerId_cmssw = trackExtra.innerDetId();
                        outerId_cmssw = trackExtra.outerDetId();
                        // test:
                        firstRecHit   = trackExtra.firstRecHit();
                        recHitsSize   = trackExtra.recHitsSize();

                        hitCollection = std::make_unique<TrackingRecHitCollection>(trackExtra.recHitsProduct());
                    }
                    edm::OrphanHandle<TrackingRecHitCollection> hitHandle = iEvent.put(std::move(hitCollection));
                    TrackingRecHitRefProd hitRefProd(hitHandle);



                    // OUTER 
                    // auto oState = trackProxy.outermostTrackState();
                    // std::shared_ptr<reco::Track::Point> oStatePosition_ptr;
                    // std::shared_ptr<reco::Track::Vector> oStateMomentum_ptr;
                    // std::shared_ptr<reco::Track::CovarianceMatrix> oCovariance_ptr;
                    // if(oState.hasReferenceSurface()) {
                    //     // Get the Momentum
                    //     auto oAbsoluteP = abs(1 / oState.parameters()[4]);
                    //     Acts::Vector3 oDir = makeDirectionFromPhiTheta(oState.parameters()[2], oState.parameters()[3]);
                    //     oStateMomentum_ptr = std::make_shared<reco::Track::Vector>(oAbsoluteP*oDir[0], oAbsoluteP*oDir[1], oAbsoluteP*oDir[2]);
                        
                    //     // Get the Global Position
                    //     auto oSurf = oState.referenceSurface().getSharedPtr();

                    //     auto detEl = static_cast<const Acts::CMSDetectorElement*>(oSurf->associatedDetectorElement());
                    //     if(detEl) {
                    //         auto this_ID = detEl->detID();
                    //         std::cout << "Outer ID: " << this_ID << std::endl;
                    //         outerId_cmssw = this_ID;
                    //     }

                    //     Acts::Vector3 oGlopalPos = oSurf->localToGlobal(gCtx, Acts::Vector2(oState.parameters()[0], oState.parameters()[1]), oDir); //(indeces from Acts/Definitions/TrackParametrization.hpp)
                    //     oStatePosition_ptr = std::make_shared<reco::Track::Point>(oGlopalPos[0], oGlopalPos[1], oGlopalPos[2]); // [mm]
                        
                    //     // Get the covariance
                    //     reco::Track::CovarianceMatrix oCovariance;
                    //     for (int i = 0; i < reco::Track::dimension; ++i) {
                    //         for (int j = 0; j <= i; ++j) { 
                    //             oCovariance(i,j) = oState.covariance()(i,j);  
                    //         }
                    //     }
                    //     oCovariance_ptr = std::make_shared<reco::Track::CovarianceMatrix>(oCovariance);

                    //     // Debug printout:
                    //     std::cout << " ===== OUTER STATE PARAMETERS =====" << std::endl;
                    //     std::cout << "Position: " << *oStatePosition_ptr << std::endl;
                    //     std::cout << "Momentum: " << *oStateMomentum_ptr << std::endl;
                    //     std::cout << "Covariance: " << *oCovariance_ptr << std::endl;
                    // }
                    // else {
                    //     std::cout << "ERROR: Outer state does not have a reference surface!" << std::endl;
                    // }

                    bool outerOK = false, innerOK = false;
                    
                    std::shared_ptr<reco::Track::Point> iStatePosition_ptr;
                    std::shared_ptr<reco::Track::Point> oStatePosition_ptr;
                    std::shared_ptr<reco::Track::Vector> iStateMomentum_ptr;
                    std::shared_ptr<reco::Track::Vector> oStateMomentum_ptr;
                    std::shared_ptr<reco::Track::CovarianceMatrix> iCovariance_ptr;
                    std::shared_ptr<reco::Track::CovarianceMatrix> oCovariance_ptr;
                    // HERE
                    for(auto this_state : trackProxy.trackStatesReversed()){
                        if(this_state.hasReferenceSurface()){
                            const Acts::Surface& this_surf = this_state.referenceSurface();
                            std::vector<double> params;
                            for(int i = 0; i < 5; i++) params[i] = this_state.parameters()[i];
                                    

                            if(this_surf.associatedDetectorElement() != nullptr){

                                auto detEl = dynamic_cast<const Acts::CMSDetectorElement*>(this_surf.associatedDetectorElement());
                                auto this_ID = detEl->detID();
                                
                                // std::cout << "[DEBUG] ThisID: " << this_ID << std::endl;
                                if (this_ID == outerId_cmssw) {
                                    // std::cout << "[DEBUG] Outer detId Identified! With id: " << this_ID << std::endl;
                                    outerOK = true;
                                    // Get the Momentum
                                    auto oAbsoluteP = abs(1 / this_state.parameters()[4]);
                                    Acts::Vector3 oDir = makeDirectionFromPhiTheta(this_state.parameters()[2], this_state.parameters()[3]);
                                    oStateMomentum_ptr = std::make_shared<reco::Track::Vector>(oAbsoluteP*oDir[0], oAbsoluteP*oDir[1], oAbsoluteP*oDir[2]);

                                    // Get the Global Position
                                    auto oSurf = this_state.referenceSurface().getSharedPtr();
                                    Acts::Vector3 oGlopalPos = oSurf->localToGlobal(gCtx, Acts::Vector2(this_state.parameters()[0], this_state.parameters()[1]), oDir); //(indeces from Acts/Definitions/TrackParametrization.hpp)
                                    oStatePosition_ptr = std::make_shared<reco::Track::Point>(oGlopalPos[0], oGlopalPos[1], oGlopalPos[2]); // [mm]

                                    // Get the converted coviarince of this track state:
                                    // auto convertedCov = convertCovACTStoCMSSW(params, oSurf, this_state.covariance());

                                    // Get the covariance
                                    reco::Track::CovarianceMatrix oCovariance;
                                    for (int i = 0; i < reco::Track::dimension; ++i) {
                                        for (int j = 0; j <= i; ++j) { 
                                            oCovariance(i,j) = this_state.covariance()(i,j);  
                                        }
                                    }
                                    oCovariance_ptr = std::make_shared<reco::Track::CovarianceMatrix>(oCovariance);


                                    // Debug printout:
                                    // std::cout << " ===== OUTER STATE PARAMETERS =====" << std::endl;
                                    // std::cout << "Position: " << *oStateMomentum_ptr << std::endl;
                                    // std::cout << "Momentum: " << *oStatePosition_ptr << std::endl;
                                    // std::cout << "Covariance: " << *oCovariance_ptr << std::endl;
                                } 
                                else if (this_ID == innerId_cmssw) {
                                    // std::cout << "[DEBUG] Inner detId Identified! With id: " << this_ID << std::endl;
                                    innerOK = true;
                                    // Get the Momentum
                                    auto iAbsoluteP = abs(1 / this_state.parameters()[4]);
                                    Acts::Vector3 iDir = makeDirectionFromPhiTheta(this_state.parameters()[2], this_state.parameters()[3]);
                                    iStateMomentum_ptr = std::make_shared<reco::Track::Vector>(iAbsoluteP*iDir[0], iAbsoluteP*iDir[1], iAbsoluteP*iDir[2]);

                                    // Get the Global Position
                                    auto iSurf = this_state.referenceSurface().getSharedPtr();
                                    Acts::Vector3 iGlopalPos = iSurf->localToGlobal(gCtx, Acts::Vector2(this_state.parameters()[0], this_state.parameters()[1]), iDir); //(indeces from Acts/Definitions/TrackParametrization.hpp)
                                    iStatePosition_ptr = std::make_shared<reco::Track::Point>(iGlopalPos[0], iGlopalPos[1], iGlopalPos[2]); // [mm]

                                    // Get the converted coviarince of this track state:
                                    // auto convertedCov = convertCovACTStoCMSSW(params, iSurf, this_state.covariance());

                                    // Get the covariance
                                    reco::Track::CovarianceMatrix iCovariance;
                                    for (int i = 0; i < reco::Track::dimension; ++i) {
                                        for (int j = 0; j <= i; ++j) { 
                                            iCovariance(i,j) = this_state.covariance()(i,j);  
                                        }
                                    }
                                    iCovariance_ptr = std::make_shared<reco::Track::CovarianceMatrix>(iCovariance);


                                    // Debug printout:
                                    // std::cout << " ===== INNER STATE PARAMETERS =====" << std::endl;
                                    // std::cout << "Position: " << *iStateMomentum_ptr << std::endl;
                                    // std::cout << "Momentum: " << *iStatePosition_ptr << std::endl;
                                    // std::cout << "Covariance: " << *iCovariance_ptr << std::endl;

                                }
                            }
                        }
                    }
                    
                    if (outerOK && innerOK) {
                        // std::cout << ">>>>> INNER AND OUTER DETID IDENTIFIED" << std::endl;
                        TotACTS_GoodOoutIndetID += 1;
                        reco::TrackExtra trackExtra_acts(*oStatePosition_ptr, *oStateMomentum_ptr, true, *iStatePosition_ptr, *iStateMomentum_ptr, true, *oCovariance_ptr, outerId_cmssw, *iCovariance_ptr, innerId_cmssw, seedDir_cmssw, seedRef_cmssw);
                        trackExtra_acts.setHits(hitRefProd, firstRecHit, recHitsSize);
                        
                        // ===== Assign the reco::Hits to the trackExtra =====
                        // auto hitsCollection = std::make_unique<TrackingRecHitCollection>();
                        // auto const firstHitIndex = hitIndex;
                        // unsigned int nHitsAdded = 0;
                        // for (trackingRecHit_iterator hit = recoTrack.recHitsBegin(); hit != recoTrack.recHitsEnd(); ++hit, ++hitIndex) {
                        //     hitsCollection->push_back((*hit)->clone());
                        //     ++nHitsAdded;
                        // }

                        // trackExtra_acts.setHits(hitsCollection.get(), firstHitIndex, nHitsAdded);
                        // iEvent.put(std::move(hitsCollection));


                        // ===== Assign TrackResiduals to the TrackExtra =====
                        reco::TrackResiduals TrackRes_acts;
                        TrackRes_acts.resize(trackProxy.nMeasurements());
                        int idx = 0;
                        // to do that, a loop on all the track state is needed:
                        for(auto this_state : trackProxy.trackStatesReversed()){
                            // Get the cluster position:
                            if(this_state.hasCalibrated()){
                                auto clusterPos = this_state.calibrated<2>();
                                double x_cluster = clusterPos[0] / 10; // from mm to cm
                                double y_cluster = clusterPos[1] / 10; // from mm to cm
                                
                                auto fitPos = this_state.smoothed();               
                                auto fitCov = this_state.smoothedCovariance();   

                                double x_fit =   fitPos[Acts::eBoundLoc0] / 10; // from mm to cm
                                double y_fit =   fitPos[Acts::eBoundLoc1] / 10; // from mm to cm

                                double x_err = sqrt(fitCov(Acts::eBoundLoc0,Acts::eBoundLoc0)) / 10; // from mm to cm
                                double y_err = sqrt(fitCov(Acts::eBoundLoc1,Acts::eBoundLoc1)) / 10; // from mm to cm

                                auto resX = x_fit - x_cluster;
                                auto pullX = resX / x_err;
                                auto resY = y_fit - y_cluster;
                                auto pullY = resY / y_err;

                                TrackRes_acts.setResidualXY(idx, resX, resY);
                                TrackRes_acts.setPullXY(idx, pullX, pullY);
                                idx += 1;

                                // std::cout << "resX: " << resX << ", pullX: " << pullX  << ", resY: " << resY << ", pullY: " << pullY << std::endl;
                            }
                        }
                        trackExtra_acts.setResiduals(TrackRes_acts);

                        //recoTrack_acts.setExtra(trackExtra_acts);

                        ExtraColl->push_back(trackExtra_acts);
                        TracksColl->push_back(recoTrack_acts);
                        // TEMPORARY (related to the inner and outer detID match)
                        TracksColl_CMSSW->push_back(recoTrack);

                        // Save the hit collection in the event:
                        //iEvent.put(std::move(hitsCollection), "actsRecHits");
                    }



                    // DEBUG: fill a txt file with different informations:
                    // I) Total number of measurements for this track:
                    unsigned int nMeas = trackProxy.nMeasurements();
                    // II) Total number of holes:
                    unsigned int nHoles = trackProxy.nHoles();
                    // III) Does it have a match with outer/inner detId?
                    bool O_match = outerOK;
                    bool I_match = innerOK;
                    // IV) Eta of the particle:
                    double eta = -std::log(std::tan(trackProxy.theta() / 2.0));
                    // V) Number of reco::Hits from CMSSW
                    /// NOTE: in order to have only VALID hits I need to loop on all the hits:
                    unsigned int nHits_cmssw = 0;
                    for(const auto& hit : recoTrack.recHits()){
                        if(hit) {
                            nHits_cmssw += 1;
                        }
                    }

                    // Save everything in a file:
                    outfile << nMeas << "\t" << nHoles << "\t" << O_match << "\t" << I_match << "\t" << eta << "\t" << nHits_cmssw << "\n";

                    



                } else {
                    std::cout << "ERROR: Fit failed, moving to the next track" << fit_result.error() << std::endl;
                }
                
                trk_len_ = std::distance(recoTrack.recHits().begin(), recoTrack.recHits().end());
            } else {
                std::cout << "Not associated Track" << std::endl;
            }
            std::cout << "################################################################################################ " << std::endl;
        }
    }

    edm::OrphanHandle<reco::TrackExtraCollection> ExtraHandle = iEvent.put(std::move(ExtraColl));

    for (long unsigned int k = 0; k < TracksColl->size(); k++) {
        const reco::TrackExtraRef theTrackExtraRef(ExtraHandle, k);
        (*TracksColl)[k].setExtra(theTrackExtraRef);
    }

    iEvent.put(std::move(TracksColl), "recoTracksCollACTS");
    iEvent.put(std::move(TracksColl_CMSSW), "recoTracksCollCMSSW");
}


#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(ACTSRefitTracksProducerDEBUG);