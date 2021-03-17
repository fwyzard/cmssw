import FWCore.ParameterSet.Config as cms

SimCalorimetryRAW = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep EBSrFlagsSorted_simEcalDigis_*_*', 
        'keep EESrFlagsSorted_simEcalDigis_*_*', 
        'keep *_simHcalUnsuppressedDigis_*_*', 
        'keep *_mix_EETimeDigi_*', 
        'keep *_mix_EBTimeDigi_*', 
        'keep *_simEcalUnsuppressedDigis_*_*', 
        'keep *_simHGCalUnsuppressedDigis_EE_*', 
        'keep *_simHGCalUnsuppressedDigis_HEfront_*', 
        'keep *_simHGCalUnsuppressedDigis_HEback_*', 
        'keep *_mix_MergedCaloTruth_*'
    )
)