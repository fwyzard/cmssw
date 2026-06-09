#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterialACTSLike.h"

using namespace std;

Acts::BoundMatrix convertCovCMSSWtoACTS(AlgebraicSymMatrix55 cov_cmssw){

    // Local copy of cmssw matrix
    Eigen::Matrix<double,5,5> local_CMSSW;
    local_CMSSW.setZero();
    for (int i = 0; i < 5; ++i){
        for (int j = 0; j < 5; ++j){
            local_CMSSW(i,j) = cov_cmssw(i,j);
        }
    }

    // Create the Jacobian and fill it
    Eigen::Matrix<double,5,5> J = Eigen::Matrix<double,5,5>::Zero();
    J(0, 3) = 10.0;
    J(1, 4) = 10.0;
    J(2, 2) = 1.0;
    J(3, 1) = -1.0;
    J(4, 0) = 1.0;

    // Apply the Jacobian to the local CMSSW matrix:
    Eigen::Matrix<double,5,5> local_ACTS = J * local_CMSSW * J.transpose();

    // Define the final ACTS matrix (we need to add the time):
    Acts::BoundMatrix cov_acts = Acts::BoundMatrix::Zero();
    cov_acts.block<5,5>(0,0) = local_ACTS;
    cov_acts(5,5) = 1.0;

    return cov_acts;
}

using Covariance = const Eigen::Ref<const Eigen::Matrix<double,6,6>>; // Eigen::Ref to take both const and non-const matrices
CurvilinearTrajectoryError convertCovACTStoCMSSW(const Covariance cov_acts){
  Eigen::Matrix<double,5,5> Cacts;
  Cacts.setZero();
  for (int i = 0; i < 5; ++i){
      for (int j = 0; j < 5; ++j){
          Cacts(i,j) = cov_acts(i,j);
      }
  }

  Eigen::Matrix<double,5,5> J = Eigen::Matrix<double,5,5>::Zero();

  J(0, Acts::eBoundQOverP) = 1.0;
  J(1, Acts::eBoundTheta) = -1.0; // lambda = pi/2 - theta
  J(2, Acts::eBoundPhi) = 1.0;
  J(3, Acts::eBoundLoc0) = 0.1; // se loc0 ~ x
  J(4, Acts::eBoundLoc1) = 0.1; 

  Eigen::Matrix<double,5,5> cmsCov = J * Cacts * J.transpose();

  AlgebraicSymMatrix55 temp_m;
  for (int i = 0; i < 5; ++i){
      for (int j = 0; j <= i; ++j){
          temp_m(i,j) = cmsCov(i,j);
      }
  }
  CurvilinearTrajectoryError cov_cmssw(temp_m);

  return cov_cmssw;
}

PropagatorWithMaterialACTSLike::~PropagatorWithMaterialACTSLike() {}

PropagatorWithMaterialACTSLike::PropagatorWithMaterialACTSLike(PropagationDirection dir,
                                               const float mass,
                                               const MagneticField* mf,
                                               const float maxDPhi,
                                               bool useRungeKutta,
                                               float ptMin,
                                               bool useOldAnalPropLogic,
                                               //
                                               std::shared_ptr<TrackingGeometryWithDetEls> TrkandDetEls,
                                               const TrackerGeometry* trkGeo_cmssw)
    : Propagator(dir),
      rkProduct(mf, dir),
      theGeometricalPropagator(useRungeKutta ? rkProduct.propagator.clone()
                                             : new AnalyticalPropagator(mf, dir, maxDPhi, useOldAnalPropLogic)),
      theMEUpdator(new CombinedMaterialEffectsUpdator(mass, ptMin)),
      theMaterialLocation(atDestination),
      field(mf),
      useRungeKutta_(useRungeKutta),
      trkGeo_and_DetEls_(TrkandDetEls),
      trkGeo_cmssw_(trkGeo_cmssw){

  // ===== Define the ACTS propagator to be used in this class =====
  // I) EigenStepper
  std::shared_ptr<const CMSMagneticFieldProvider>  magFieldPtr;
  if(field != nullptr){
    magFieldPtr = std::make_shared<const CMSMagneticFieldProvider>(*field);
  } else {
    std::cout << "[ERROR] B field ptr is null" << std::endl;
  }
  Acts::EigenStepper<> es(magFieldPtr);
  // II) Navigator
  Acts::Navigator::Config navi_cfg;
  navi_cfg.trackingGeometry = trkGeo_and_DetEls_->trackingGeometry;
  navi_cfg.resolveSensitive = true;
  navi_cfg.resolveMaterial = true;
  navi_cfg.resolvePassive = true;
  std::shared_ptr<const Acts::Logger> navi_logger = Acts::getDefaultLogger("Navigator", Acts::Logging::Level::INFO);
  Acts::Navigator navi(navi_cfg, std::move(navi_logger));
  std::shared_ptr<const Acts::Logger> prop_logger = Acts::getDefaultLogger("Propagator", Acts::Logging::Level::INFO);
  // III) Propagator
  Acts::Propagator prop(es, navi, std::move(prop_logger));
  // IV) Define the concrete propagator
  ConcProp = DeepCopyPointerByClone<MyConcretePropagator>(new MyConcretePropagator(prop));
}

TsosWP PropagatorWithMaterialACTSLike::getTsosWP_fromACTS_params(Acts::Vector4 pos4, Acts::Vector3 dir, double qOverP,  Acts::BoundMatrix covMat_acts, const Surface& surfTarget) const {
  // ===== From CMSSW parameters we create ACTS start parameters =====
  /// NOTE: Is the particle hypothesis needed?
  auto start_param = Acts::BoundTrackParameters::createCurvilinear(pos4, dir, qOverP, covMat_acts, Acts::ParticleHypothesis::muon());

  // ===== Convert the target surface from CMSSW into an ACTS's surface =====
  std::shared_ptr<Acts::TrackingGeometry> trkGeo = trkGeo_and_DetEls_->trackingGeometry;
  SurfaceConverters surfConv(trkGeo_cmssw_);
  std::shared_ptr<Acts::Surface> surf_ACTS_target = surfConv.fromCMSSWtoACTS(surfTarget);

  // ===== Propagate to the target surface =====
  PropagationAlgorithm_Config cfg;
  std::shared_ptr<const Acts::Logger> prop_logger = Acts::getDefaultLogger("Concrete Propagator", Acts::Logging::Level::INFO);
  auto propRes = ConcProp->execute(cfg, *prop_logger, start_param, *surf_ACTS_target); 

  // ===== Get bound parameters from ACTS and use them to build the tsos for CMSSW =====
  BoundParameters bParam = propRes.first;
  double length = propRes.second;

  auto prop_pos = bParam.position(Acts::GeometryContext{});
  auto prop_dir = bParam.direction();
  auto prop_cov = bParam.covariance();
  GlobalPoint gp(prop_pos[0], prop_pos[1], prop_pos[2]);
  GlobalVector gv(prop_dir[0], prop_dir[1], prop_dir[2]);
  TrackCharge prop_charge = static_cast<int>(bParam.charge());

  FreeTrajectoryState fts_fromACTS(gp, gv, prop_charge, field);
  fts_fromACTS.setCurvilinearError(convertCovACTStoCMSSW(prop_cov.value()));
  TrajectoryStateOnSurface tsos_fromACTS(fts_fromACTS, surfTarget);
  TsosWP newTsosWP_FromACTS(tsos_fromACTS, length);

  return newTsosWP_FromACTS;
}

pair<TrajectoryStateOnSurface, double> PropagatorWithMaterialACTSLike::propagateWithPath(const FreeTrajectoryState& fts, const Plane& plane) const {

  // ===== Get parameters from FreeTrajectoryState =====
  GlobalPoint gPoint = fts.position();
  GlobalVector gMomentum = fts.momentum();
  double qOverP = fts.signedInverseMomentum();
  const auto& curvErr = fts.curvilinearError();
  AlgebraicSymMatrix55 covMat_cmssw_init = curvErr.matrix();  

  Acts::Vector4 pos4 = {gPoint.x(), gPoint.y(), gPoint.z(), 0};
  Acts::Vector3 dir = {gMomentum.x(), gMomentum.y(), gMomentum.z()};
  Acts::BoundMatrix covMat_acts_init = convertCovCMSSWtoACTS(covMat_cmssw_init);

  return getTsosWP_fromACTS_params(pos4, dir, qOverP, covMat_acts_init, plane);
}

pair<TrajectoryStateOnSurface, double> PropagatorWithMaterialACTSLike::propagateWithPath(const FreeTrajectoryState& fts, const Cylinder& cylinder) const {


  // ===== Get parameters from FreeTrajectoryState =====
  GlobalPoint gPoint = fts.position();
  GlobalVector gMomentum = fts.momentum();
  double qOverP = fts.signedInverseMomentum();
  const auto& curvErr = fts.curvilinearError();
  AlgebraicSymMatrix55 covMat_cmssw_init = curvErr.matrix();  

  Acts::Vector4 pos4 = {gPoint.x(), gPoint.y(), gPoint.z(), 0};
  Acts::Vector3 dir = {gMomentum.x(), gMomentum.y(), gMomentum.z()};
  Acts::BoundMatrix covMat_acts_init = convertCovCMSSWtoACTS(covMat_cmssw_init);

  return getTsosWP_fromACTS_params(pos4, dir, qOverP, covMat_acts_init, cylinder);
}

pair<TrajectoryStateOnSurface, double> PropagatorWithMaterialACTSLike::propagateWithPath(const TrajectoryStateOnSurface& tsos, const Plane& plane) const {

  // ===== Get parameters from FreeTrajectoryState =====
  GlobalPoint gPoint = tsos.globalPosition();
  GlobalVector gDir = tsos.globalDirection();
  double qOverP = tsos.signedInverseMomentum();
  const auto& curvErr = tsos.curvilinearError();
  AlgebraicSymMatrix55 covMat_cmssw_init = curvErr.matrix();  

  Acts::Vector4 pos4 = {gPoint.x(), gPoint.y(), gPoint.z(), 0};
  Acts::Vector3 dir = {gDir.x(), gDir.y(), gDir.z()};
  Acts::BoundMatrix covMat_acts_init = convertCovCMSSWtoACTS(covMat_cmssw_init);

  return getTsosWP_fromACTS_params(pos4, dir, qOverP, covMat_acts_init, plane);;
}

pair<TrajectoryStateOnSurface, double> PropagatorWithMaterialACTSLike::propagateWithPath(const TrajectoryStateOnSurface& tsos, const Cylinder& cylinder) const {

  // ===== Get parameters from FreeTrajectoryState =====
  GlobalPoint gPoint = tsos.globalPosition();
  GlobalVector gDir = tsos.globalDirection();
  double qOverP = tsos.signedInverseMomentum();
  const auto& curvErr = tsos.curvilinearError();
  AlgebraicSymMatrix55 covMat_cmssw_init = curvErr.matrix();  

  Acts::Vector4 pos4 = {gPoint.x(), gPoint.y(), gPoint.z(), 0};
  Acts::Vector3 dir = {gDir.x(), gDir.y(), gDir.z()};
  Acts::BoundMatrix covMat_acts_init = convertCovCMSSWtoACTS(covMat_cmssw_init);

  return getTsosWP_fromACTS_params(pos4, dir, qOverP, covMat_acts_init, cylinder);;
}





