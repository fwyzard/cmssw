import FWCore.ParameterSet.Config as cms

SimCalorimetryPREMIX = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep EBDigiCollection_simEcalDigis_*_*',
        'keep EEDigiCollection_simEcalDigis_*_*',
        'keep ESDigiCollection_simEcalUnsuppressedDigis_*_*',
        'keep *_simHcalDigis_*_*',
        'drop ESDigiCollection_simEcalUnsuppressedDigis_*_*',
        'drop EEDigiCollection_simEcalDigis_*_*',
        'keep *_simHGCalUnsuppressedDigis_EE_*',
        'keep *_simHGCalUnsuppressedDigis_HEfront_*',
        'keep *_simHGCalUnsuppressedDigis_HEback_*',
        'keep *_mix_MergedCaloTruth_*'
    )
)