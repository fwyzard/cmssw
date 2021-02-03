import FWCore.ParameterSet.Config as cms

ticlSeedingGlobal = cms.EDProducer("TICLSeedingRegionProducer",
    algoId = cms.int32(2),
    algo_verbosity = cms.int32(0),
    cutTk = cms.string('1.48 < abs(eta) < 3.0 && pt > 1. && quality("highPurity") && hitPattern().numberOfLostHits("MISSING_OUTER_HITS") < 5'),
    mightGet = cms.optional.untracked.vstring,
    propagator = cms.string('PropagatorWithMaterial'),
    tracks = cms.InputTag("generalTracks")
)
