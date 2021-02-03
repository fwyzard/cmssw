import FWCore.ParameterSet.Config as cms

SimMuonFEVTDEBUG = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_simMuonCSCDigis_*_*', 
        'keep *_simMuonDTDigis_*_*', 
        'keep *_simMuonRPCDigis_*_*', 
        'keep *_simMuonGEMDigis_*_*', 
        'keep *_simMuonME0PseudoDigis_*_*', 
        'keep *_simMuonME0PseudoReDigis_*_*', 
        'keep *_simMuonME0Digis_*_*'
    )
)