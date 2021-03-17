import FWCore.ParameterSet.Config as cms

RecoLocalTrackerRECO = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep DetIdedmEDCollection_siStripDigis_*_*', 
        'keep DetIdedmEDCollection_siPixelDigis_*_*', 
        'keep PixelFEDChanneledmNewDetSetVector_siPixelDigis_*_*', 
        'keep *_siPixelClusters_*_*', 
        'keep *_siStripClusters_*_*', 
        'keep ClusterSummary_clusterSummaryProducer_*_*', 
        'keep *_siPhase2Clusters_*_*'
    )
)