import FWCore.ParameterSet.Config as cms

RecoLocalCaloRECO = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_hbhereco_*_*', 
        'keep *_hbheprereco_*_*', 
        'keep *_hfprereco_*_*', 
        'keep *_hfreco_*_*', 
        'keep *_horeco_*_*', 
        'keep HBHERecHitsSorted_hbherecoMB_*_*', 
        'keep HORecHitsSorted_horecoMB_*_*', 
        'keep HFRecHitsSorted_hfrecoMB_*_*', 
        'keep ZDCDataFramesSorted_hcalDigis_*_*', 
        'keep ZDCDataFramesSorted_castorDigis_*_*', 
        'keep QIE10DataFrameHcalDataFrameContainer_hcalDigis_ZDC_*', 
        'keep ZDCRecHitsSorted_zdcreco_*_*', 
        'keep *_castorreco_*_*', 
        'keep *_reducedHcalRecHits_*_*', 
        'keep HcalUnpackerReport_castorDigis_*_*', 
        'keep HcalUnpackerReport_hcalDigiAlCaMB_*_*', 
        'keep HcalUnpackerReport_hcalDigis_*_*', 
        'keep *_HGCalRecHit_*_*', 
        'keep recoCaloClusters_hgcalLayerClusters_*_*', 
        'keep *_hgcalLayerClusters_timeLayerCluster_*', 
        'keep *_hgcalLayerClusters_InitialLayerClustersMask_*', 
        'keep *_ecalPreshowerRecHit_*_*', 
        'keep *_ecalRecHit_*_*', 
        'keep *_ecalCompactTrigPrim_*_*', 
        'keep *_ecalTPSkim_*_*', 
        'keep EBSrFlagsSorted_ecalDigis__*', 
        'keep EESrFlagsSorted_ecalDigis__*', 
        'keep *_mix_EBTimeDigi_*', 
        'keep *_mix_EETimeDigi_*', 
        'keep *_ecalDetailedTimeRecHit_*_*', 
        'keep *_hgcalMultiClusters_*_*', 
        'keep *_iterHGCalMultiClusters_*_*'
    )
)