import FWCore.ParameterSet.Config as cms

SimTrackerRAW = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_allTrackMCMatch_*_*')
)