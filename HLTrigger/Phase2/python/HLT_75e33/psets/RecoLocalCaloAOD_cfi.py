import FWCore.ParameterSet.Config as cms

RecoLocalCaloAOD = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_castorreco_*_*', 
        'keep *_reducedHcalRecHits_*_*', 
        'keep HcalUnpackerReport_castorDigis_*_*', 
        'keep HcalUnpackerReport_hcalDigiAlCaMB_*_*', 
        'keep HcalUnpackerReport_hcalDigis_*_*', 
        'keep *_HGCalRecHit_*_*', 
        'keep recoCaloClusters_hgcalLayerClusters_*_*', 
        'keep *_hgcalLayerClusters_timeLayerCluster_*', 
        'keep *_hgcalLayerClusters_InitialLayerClustersMask_*'
    )
)