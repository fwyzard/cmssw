import FWCore.ParameterSet.Config as cms

RecoEcalRECO = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_hybridSuperClusters_*_*',
        'keep recoSuperClusters_correctedHybridSuperClusters_*_*',
        'keep *_multi5x5SuperClusters_*_*',
        'keep recoSuperClusters_multi5x5SuperClustersWithPreshower_*_*',
        'keep *_particleFlowSuperClusterECAL_*_*',
        'keep *_particleFlowSuperClusterOOTECAL_*_*',
        'drop recoClusterShapes_*_*_*',
        'drop recoBasicClustersToOnerecoClusterShapesAssociation_*_*_*',
        'drop recoBasicClusters_multi5x5BasicClusters_multi5x5BarrelBasicClusters_*',
        'drop recoSuperClusters_multi5x5SuperClusters_multi5x5BarrelSuperClusters_*',
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
        'keep recoCaloClusters_particleFlowSuperClusterHGCalFromMultiCl__*',
        'keep *_particleFlowSuperClusterHGCal_*_*',
        'keep *_particleFlowSuperClusterHGCalFromMultiCl_*_*'
    )
)