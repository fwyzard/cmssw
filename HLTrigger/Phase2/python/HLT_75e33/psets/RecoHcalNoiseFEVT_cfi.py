import FWCore.ParameterSet.Config as cms

RecoHcalNoiseFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep recoHcalNoiseRBXs_hcalnoise_*_*',
        'keep HcalNoiseSummary_hcalnoise_*_*'
    )
)