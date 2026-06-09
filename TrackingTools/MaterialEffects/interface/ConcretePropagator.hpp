#ifndef _CONCRETEPROPAGATOR_H_
#define _CONCRETEPROPAGATOR_H_

#include "Acts/Propagator/EigenStepper.hpp"
#include "Acts/Propagator/Navigator.hpp"
#include "Acts/Propagator/Propagator.hpp"
#include "Acts/Propagator/MaterialInteractor.hpp"
#include "Acts/Geometry/TrackingGeometry.hpp"
#include "Acts/Propagator/detail/SteppingLogger.hpp"

using BoundParameters = Acts::GenericBoundTrackParameters<Acts::ParticleHypothesis>;

// using PropagationOutput = std::pair<PropagationSummary, Acts::RecordedMaterial>;
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
  double ptLoopers = 700 * Acts::UnitConstants::MeV;
  /// Max step size steering
  double maxStepSize = 5 * Acts::UnitConstants::m;
};

template <typename propagator_t>
class ConcretePropagator {
 public:
  explicit ConcretePropagator(propagator_t propagator)
      : m_propagator{std::move(propagator)} {}

    ConcretePropagator* clone() const {
            return new ConcretePropagator(*this);
        }

  template <typename StartParameters>
  std::pair<BoundParameters, double> execute(
      const PropagationAlgorithm_Config& cfg,
      const Acts::Logger& logger,
      const StartParameters& startParameters,
      const Acts::Surface& targetSurf) const {
    ACTS_DEBUG("Test propagation/extrapolation starts");

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

    PropagatorOptions options(Acts::GeometryContext{}, Acts::MagneticFieldContext{});
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

    auto propRes = m_propagator.propagate(startParameters, targetSurf, options);

    if(!propRes.ok()){
      std::cout << "ERROR: Propagation failed!" << std::endl;
    }
    BoundParameters finalBound = *propRes->endParameters;
    double length = propRes->pathLength;

    return std::make_pair(finalBound, length);

  }

 private:
  propagator_t m_propagator;
};

#endif
