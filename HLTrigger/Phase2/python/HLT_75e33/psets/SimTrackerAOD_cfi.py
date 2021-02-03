import FWCore.ParameterSet.Config as cms

SimTrackerAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_allTrackMCMatch_*_*')
)