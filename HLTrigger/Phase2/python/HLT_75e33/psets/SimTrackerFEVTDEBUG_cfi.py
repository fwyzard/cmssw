import FWCore.ParameterSet.Config as cms

SimTrackerFEVTDEBUG = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_simSiPixelDigis_*_*',
        'keep *_simSiStripDigis_*_*',
        'drop *_mix_simSiPixelDigis*_*',
        'drop *_mix_simSiStripDigis*_*',
        'keep *_allTrackMCMatch_*_*',
        'keep *_trackingParticleRecoTrackAsssociation_*_*',
        'keep *_assoc2secStepTk_*_*',
        'keep *_assoc2thStepTk_*_*',
        'keep *_assoc2GsfTracks_*_*',
        'keep *_assocOutInConversionTracks_*_*',
        'keep *_assocInOutConversionTracks_*_*',
        'keep *_TTClusterAssociatorFromPixelDigis_*_*',
        'keep *_TTStubAssociatorFromPixelDigis_*_*'
    )
)