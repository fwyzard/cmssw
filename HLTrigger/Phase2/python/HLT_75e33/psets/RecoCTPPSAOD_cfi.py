import FWCore.ParameterSet.Config as cms

RecoCTPPSAOD = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep TotemTriggerCounters_totemTriggerRawToDigi_*_*', 
        'keep TotemFEDInfos_totemRPRawToDigi_*_*', 
        'keep TotemRPDigiedmDetSetVector_totemRPRawToDigi_*_*', 
        'keep TotemVFATStatusedmDetSetVector_totemRPRawToDigi_*_*', 
        'keep TotemRPClusteredmDetSetVector_totemRPClusterProducer_*_*', 
        'keep TotemRPRecHitedmDetSetVector_totemRPRecHitProducer_*_*', 
        'keep TotemRPUVPatternedmDetSetVector_totemRPUVPatternFinder_*_*', 
        'keep TotemRPLocalTrackedmDetSetVector_totemRPLocalTrackFitter_*_*', 
        'keep TotemFEDInfos_ctppsDiamondRawToDigi_*_*', 
        'keep CTPPSDiamondDigiedmDetSetVector_ctppsDiamondRawToDigi_*_*', 
        'keep TotemVFATStatusedmDetSetVector_ctppsDiamondRawToDigi_*_*', 
        'keep CTPPSDiamondRecHitedmDetSetVector_ctppsDiamondRecHits_*_*', 
        'keep CTPPSDiamondLocalTrackedmDetSetVector_ctppsDiamondLocalTracks_*_*', 
        'keep TotemTimingDigiedmDetSetVector_totemTimingRawToDigi_*_*', 
        'keep TotemTimingRecHitedmDetSetVector_totemTimingRecHits_*_*', 
        'keep TotemTimingLocalTrackedmDetSetVector_totemTimingLocalTracks_*_*', 
        'keep CTPPSPixelDigiedmDetSetVector_ctppsPixelDigis_*_*', 
        'keep CTPPSPixelDataErroredmDetSetVector_ctppsPixelDigis_*_*', 
        'keep CTPPSPixelClusteredmDetSetVector_ctppsPixelClusters_*_*', 
        'keep CTPPSPixelRecHitedmDetSetVector_ctppsPixelRecHits_*_*', 
        'keep CTPPSPixelLocalTrackedmDetSetVector_ctppsPixelLocalTracks_*_*', 
        'keep CTPPSLocalTrackLites_ctppsLocalTrackLiteProducer_*_*', 
        'keep recoForwardProtons_ctppsProtons_*_*'
    )
)