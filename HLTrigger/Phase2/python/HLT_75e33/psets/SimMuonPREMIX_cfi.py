import FWCore.ParameterSet.Config as cms

SimMuonPREMIX = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_mix_g4SimHitsMuonDTHits_*', 
        'keep *_mix_g4SimHitsMuonCSCHits_*', 
        'keep *_mix_g4SimHitsMuonRPCHits_*', 
        'keep *_mix_g4SimHitsMuonGEMHits_*', 
        'keep *_mix_g4SimHitsMuonME0Hits_*'
    )
)