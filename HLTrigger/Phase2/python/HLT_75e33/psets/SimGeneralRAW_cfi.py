import FWCore.ParameterSet.Config as cms

SimGeneralRAW = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep CrossingFramePlaybackInfoNew_*_*_*', 
        'keep PileupSummaryInfos_*_*_*', 
        'keep int6stdbitsetstdpairs_*_AffectedAPVList_*', 
        'keep int_*_bunchSpacing_*', 
        'keep *_genPUProtons_*_*', 
        'keep *_mix_MergedTrackTruth_*'
    )
)