import FWCore.ParameterSet.Config as cms

RecoLocalFastTimeRECO = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_mtdRecHits_*_*',
        'keep *_mtdClusters_*_*'
    )
)