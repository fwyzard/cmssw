import FWCore.ParameterSet.Config as cms

SimTrackerPREMIX = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep Phase2TrackerDigiedmDetSetVector_mix_*_*',
        'keep *_*_Phase2OTDigiSimLink_*',
        'keep *_simSiPixelDigis_*_*'
    )
)