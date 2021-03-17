import FWCore.ParameterSet.Config as cms

TICL_FEVT = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_ticlTrackstersTrkEM_*_*', 
        'keep *_ticlTrackstersEM_*_*', 
        'keep *_ticlTrackstersHAD_*_*', 
        'keep *_ticlTrackstersTrk_*_*', 
        'keep *_ticlTrackstersMIP_*_*', 
        'keep *_ticlTrackstersMerge_*_*', 
        'keep *_ticlTrackstersHFNoseEM_*_*', 
        'keep *_ticlTrackstersHFNoseMIP_*_*', 
        'keep *_ticlTrackstersHFNoseMerge_*_*', 
        'keep *_pfTICL_*_*', 
        'keep *_ticlMultiClustersFromTrackstersEM_*_*', 
        'keep *_ticlMultiClustersFromTrackstersHAD_*_*', 
        'keep *_ticlMultiClustersFromTrackstersTrk_*_*', 
        'keep *_ticlMultiClustersFromTrackstersTrkEM_*_*', 
        'keep *_ticlMultiClustersFromTrackstersMIP_*_*', 
        'keep *_ticlMultiClustersFromTrackstersMerge_*_*'
    )
)