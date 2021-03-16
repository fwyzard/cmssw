import FWCore.ParameterSet.Config as cms

SimGeneralFEVTDEBUG = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'drop *_trackingtruthprod_*_*',
        'drop *_electrontruth_*_*',
        'keep *_mix_MergedTrackTruth_*',
        'keep CrossingFramePlaybackInfoNew_*_*_*'
    )
)