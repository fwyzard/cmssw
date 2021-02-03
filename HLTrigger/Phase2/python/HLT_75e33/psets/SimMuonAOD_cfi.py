import FWCore.ParameterSet.Config as cms

SimMuonAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_muonSimClassifier_*_*')
)