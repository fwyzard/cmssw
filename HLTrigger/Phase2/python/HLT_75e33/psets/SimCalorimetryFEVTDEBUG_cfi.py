import FWCore.ParameterSet.Config as cms

SimCalorimetryFEVTDEBUG = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_simEcalDigis_*_*', 
        'keep *_simEcalPreshowerDigis_*_*', 
        'keep *_simEcalTriggerPrimitiveDigis_*_*', 
        'keep *_simEcalEBTriggerPrimitiveDigis_*_*', 
        'keep *_simHcalDigis_*_*', 
        'keep ZDCDataFramesSorted_simHcalUnsuppressedDigis_*_*', 
        'drop ZDCDataFramesSorted_mix_simHcalUnsuppressedDigis*_*', 
        'keep *_simHcalTriggerPrimitiveDigis_*_*', 
        'keep *_mix_HcalSamples_*', 
        'keep *_mixData_HcalSamples_*', 
        'keep *_mix_HcalHits_*', 
        'keep *_mixData_HcalHits_*', 
        'keep *_DMHcalTriggerPrimitiveDigis_*_*', 
        'keep *_simHcalUnsuppressedDigis_*_*', 
        'keep *_DMHcalDigis_*_*', 
        'keep *_mix_EETimeDigi_*', 
        'keep *_mix_EBTimeDigi_*', 
        'keep *_simEcalUnsuppressedDigis_*_*', 
        'keep *_simHGCalUnsuppressedDigis_EE_*', 
        'keep *_simHGCalUnsuppressedDigis_HEfront_*', 
        'keep *_simHGCalUnsuppressedDigis_HEback_*', 
        'keep *_mix_MergedCaloTruth_*'
    )
)