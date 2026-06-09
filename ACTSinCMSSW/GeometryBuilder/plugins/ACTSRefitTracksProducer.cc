

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
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/CommonTopologies/interface/RadialStripTopology.h"
#include "Geometry/CommonTopologies/interface/RectangularStripTopology.h"
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
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"

#include "Geometry/Records/interface/ACTSTrackerGeometryRecord.h"

#include "Acts/Geometry/TrackingGeometry.hpp"
#include "Acts/MagneticField/MagneticFieldContext.hpp"
#include "Acts/EventData/TransformationHelpers.hpp"
#include "Acts/EventData/TrackContainer.hpp"
#include "Acts/EventData/VectorTrackContainer.hpp"
#include "Acts/EventData/TrackStateProxy.hpp"
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

#include "TFile.h"
#include "TTree.h"
#include "TVector3.h"
#include <Eigen/Core>

#include <vector>
#include <iostream>
#include <cmath>
#include <cstring> 
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>

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

LocalHitPosition processHit(const TrackingRecHit* hit,
                            const PixelClusterParameterEstimator* pixelCPE,
                            const StripClusterParameterEstimator* stripCPE,
                            const TrackerGeometry* trackerGeometry) {

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


//class ACTSRefitTracksProducer : public edm::one::EDAnalyzer<> {
class ACTSRefitTracksProducer : public edm::one::EDProducer<> {
public:
  explicit ACTSRefitTracksProducer(const edm::ParameterSet&);
  ~ACTSRefitTracksProducer() override;

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


ACTSRefitTracksProducer::ACTSRefitTracksProducer(const edm::ParameterSet& iConfig)
    : ACTStrkGeomInfoToken_(esConsumes<TrackingGeometryWithDetEls, ACTSTrackerGeometryRecord>()),
      trackerGeomToken_(esConsumes<TrackerGeometry, TrackerDigiGeometryRecord>()),
      trackAssociatorToken_(consumes<reco::TrackToTrackingParticleAssociator>(iConfig.getUntrackedParameter<edm::InputTag>("trackAssociator"))),  
      trackingParticleToken_(consumes<TrackingParticleCollection>(iConfig.getParameter<edm::InputTag>("trackingParticles"))),  
      trackLabels_(iConfig.getParameter<std::vector<edm::InputTag>>("trackLabels")),
      pixelClusterToken_(consumes<edmNew::DetSetVector<SiPixelCluster>>(edm::InputTag("displacedMuonReducedTrackExtras"))),
      stripClusterToken_(consumes<edmNew::DetSetVector<SiStripCluster>>(edm::InputTag("displacedMuonReducedTrackExtras"))),
      cpeToken_(esConsumes<PixelClusterParameterEstimator, TkPixelCPERecord>(edm::ESInputTag("", "PixelCPEGeneric"))),
      stripCpeToken_(esConsumes<StripClusterParameterEstimator, TkStripCPERecord>(edm::ESInputTag("", "StripCPEfromTrackAngle"))),
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


ACTSRefitTracksProducer::~ACTSRefitTracksProducer() {
    std::cout << "SUMMARY: " << std::endl;
    std::cout << "Total Number of true tracks: " << TotTrueParticles << std::endl;
    std::cout << "Total Number of matched tracks from CMSSW: " << TotCMSSW_Matched << " (efficiency = " << TotCMSSW_Matched / TotTrueParticles * 100 << "%)" << std::endl;
    std::cout << "Total Number of reFit from ACTS: " << TotACTS_reFitted << std::endl;
    std::cout << "Total Number of tracks with matching Outer and Inner detID: " << TotACTS_GoodOoutIndetID << " (" << TotACTS_GoodOoutIndetID / TotCMSSW_Matched * 100 << "%)" <<std::endl;
}

//void ACTSRefitTracksProducer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
void ACTSRefitTracksProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
    bool verbose = false;

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
        edm::LogWarning("ACTSRefitTracksProducer") << "Pixel clusters collection not found!";
    }

    edm::Handle<edmNew::DetSetVector<SiStripCluster>> stripClusters;
    iEvent.getByToken(stripClusterToken_, stripClusters);
    if (!stripClusters.isValid()) {
        edm::LogWarning("ACTSRefitTracksProducer") << "Strip clusters collection not found!";
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
        edm::LogError("ACTSRefitTracksProducer") << "CMSSW TrackerGeometry is nullptr!";
        return;  
    }
    // ACTS
    const auto& trkGeo_and_DetEls = iSetup.getData(ACTStrkGeomInfoToken_);
    DetElVect detEls = trkGeo_and_DetEls.detElements;
    std::shared_ptr<Acts::TrackingGeometry> ACTStrackingGeom = trkGeo_and_DetEls.trackingGeometry;
    if (!ACTStrackingGeom) {
        edm::LogError("ACTSRefitTracksProducer") << "ACTS TrackerGeometry is nullptr!";
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
                for (const auto& hit : recoTrack.recHits()) {
                    // save the hit in a collection to be pushed than in the event:
                    // hitsCollection->push_back(hit->clone());
                    hitCount_cmssw += 1;
                    DetId detId = hit->geographicalId();
                    const GeomDet* geomDet = trackerGeometry->idToDet(detId);
                    if (!geomDet) continue;

                    unsigned int IntSubDetID = (detId.subdetId());

                    // PixelBarrel = 1; PixelEndcap = 2; TIB = 3; TID = 4; TOB = 5; TEC = 6;
                    h_type = IntSubDetID;
                    LocalHitPosition pos = processHit(hit, pixelCPE, stripCPE, trackerGeometry);
                    LocalPoint localP(pos.x, pos.y);
                    const GeomDetUnit& detUnit = *(hit->detUnit());
                    GlobalPoint pos_global = detUnit.surface().toGlobal(localP);
                    

                    // Fitting:
                    if (pos.valid){
                        // if(h_type == 1 ) std::cout << "[DEBUG] PixelBarrel Hit: " << std::endl;
                        // if(h_type == 2 ) std::cout << "[DEBUG] PixelEndcap Hit: " << std::endl;
                        // if(h_type == 3 ) std::cout << "[DEBUG] TIB Hit: " << std::endl;
                        // if(h_type == 4 ) std::cout << "[DEBUG] TID Hit: " << std::endl;
                        // if(h_type == 5 ) std::cout << "[DEBUG] TOB Hit: " << std::endl;
                        // if(h_type == 6 ) std::cout << "[DEBUG] TEC Hit: " << std::endl;

                        // std::cout << "[DEBUG] Hit local pos: (" << pos.x*10 << ", " << pos.y*10 << ") ± (" << pos.x_err*10 << ", " << pos.y_err*10 << ")" << std::endl;
                        hit_type_.push_back(h_type);
                        local_x_.push_back(pos.x*10); // from cm to mm
                        local_y_.push_back(pos.y*10);
                        local_x_err_.push_back(pos.x_err*10);
                        local_y_err_.push_back(pos.y_err*10);
                        mod_id_.push_back(detId.rawId());
                        //std::cout << "DetID from CMSSW: " << detId.rawId() << std::endl;
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

                for( auto hit : ACTS_hits){
                    auto surf = CMSSurfaceAccessor(hit);
                    auto detEl = static_cast<const Acts::CMSDetectorElement*>(surf->associatedDetectorElement());
                    if(detEl) {
                        auto this_ID = detEl->detID();
                        auto subDet = detEl->subDetector();
                        auto surf = detEl->surface().getSharedPtr();
                        auto globalPos = surf->localToGlobal(local_gCtx, Acts::Vector2(0, 0), Acts::Vector3(0,0,0));
        
                        // std::cout << "[DEBUG] ACTS detId before the fit " << this_ID <<"; subDet = " << subDet << std::endl;
                        outF << surf->geometryId() << "; " << this_ID << "; " << subDet << "; " << globalPos.transpose() << '\n';
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
                pOptions.surfaceTolerance = 1e-6;//1e-4;

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
                    for(auto this_state : trackProxy.trackStatesReversed()){
                        if(this_state.hasReferenceSurface()){
                            const Acts::Surface& this_surf = this_state.referenceSurface();

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
                        std::cout << ">>>>> INNER AND OUTER DETID IDENTIFIED" << std::endl;
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
DEFINE_FWK_MODULE(ACTSRefitTracksProducer);