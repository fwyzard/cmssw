import FWCore.ParameterSet.Config as cms

SimTrackerDEBUG = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep PixelDigiSimLinkedmDetSetVector_simSiPixelDigis_*_*',
        'keep StripDigiSimLinkedmDetSetVector_simSiStripDigis_*_*',
        'drop *_mix_simSiPixelDigis*_*',
        'drop *_mix_simSiStripDigis*_*',
        'keep *_allTrackMCMatch_*_*'
    )
)