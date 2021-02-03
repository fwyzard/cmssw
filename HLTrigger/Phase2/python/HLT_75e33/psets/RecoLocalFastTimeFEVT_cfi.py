import FWCore.ParameterSet.Config as cms

RecoLocalFastTimeFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_mtdUncalibratedRecHits_*_*', 
        'keep *_mtdRecHits_*_*', 
        'keep *_mtdClusters_*_*', 
        'keep *_mtdTrackingRecHits_*_*'
    )
)