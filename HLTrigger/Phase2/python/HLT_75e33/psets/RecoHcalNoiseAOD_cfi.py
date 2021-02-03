import FWCore.ParameterSet.Config as cms

RecoHcalNoiseAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('keep HcalNoiseSummary_hcalnoise_*_*')
)