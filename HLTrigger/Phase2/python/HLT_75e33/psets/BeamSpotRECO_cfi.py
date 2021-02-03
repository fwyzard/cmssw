import FWCore.ParameterSet.Config as cms

BeamSpotRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_offlineBeamSpot_*_*')
)