import FWCore.ParameterSet.Config as cms

RecoEcalAOD = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_selectDigi_*_*',
        'keep EcalRecHitsSorted_reducedEcalRecHitsEB_*_*',
        'keep EcalRecHitsSorted_reducedEcalRecHitsEE_*_*',
        'keep EcalRecHitsSorted_reducedEcalRecHitsES_*_*',
        'keep recoSuperClusters_correctedHybridSuperClusters_*_*',
        'keep recoCaloClusters_hybridSuperClusters_*_*',
        'keep recoSuperClusters_hybridSuperClusters_uncleanOnlyHybridSuperClusters_*',
        'keep recoCaloClusters_multi5x5SuperClusters_multi5x5EndcapBasicClusters_*',
        'keep recoSuperClusters_correctedMulti5x5SuperClustersWithPreshower_*_*',
        'keep recoPreshowerClusters_multi5x5SuperClustersWithPreshower_*_*',
        'keep recoPreshowerClusterShapes_multi5x5PreshowerClusterShape_*_*',
        'keep recoSuperClusters_particleFlowSuperClusterECAL_*_*',
        'keep recoCaloClusters_particleFlowSuperClusterECAL_*_*',
        'keep recoSuperClusters_particleFlowSuperClusterOOTECAL_*_*',
        'keep recoCaloClusters_particleFlowSuperClusterOOTECAL_*_*',
        'keep recoSuperClusters_particleFlowSuperClusterHGCal__*',
        'keep recoCaloClusters_particleFlowSuperClusterHGCal__*',
        'keep recoSuperClusters_particleFlowSuperClusterHGCalFromMultiCl__*',
        'keep recoCaloClusters_particleFlowSuperClusterHGCalFromMultiCl__*'
    )
)