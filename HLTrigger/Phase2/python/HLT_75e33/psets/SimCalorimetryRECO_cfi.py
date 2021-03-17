import FWCore.ParameterSet.Config as cms

SimCalorimetryRECO = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_simHGCalUnsuppressedDigis_EE_*', 
        'keep *_simHGCalUnsuppressedDigis_HEfront_*', 
        'keep *_simHGCalUnsuppressedDigis_HEback_*', 
        'keep *_mix_MergedCaloTruth_*'
    )
)