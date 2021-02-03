import FWCore.ParameterSet.Config as cms

BeamSpotAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_offlineBeamSpot_*_*')
)