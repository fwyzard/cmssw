import FWCore.ParameterSet.Config as cms

TICL_AOD = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_ticlMultiClustersFromTrackstersEM_*_*',
        'keep *_ticlMultiClustersFromTrackstersHAD_*_*',
        'keep *_ticlMultiClustersFromTrackstersTrk_*_*',
        'keep *_ticlMultiClustersFromTrackstersTrkEM_*_*',
        'keep *_ticlMultiClustersFromTrackstersMIP_*_*',
        'keep *_ticlMultiClustersFromTrackstersMerge_*_*'
    )
)