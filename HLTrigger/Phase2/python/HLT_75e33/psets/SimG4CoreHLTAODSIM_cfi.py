import FWCore.ParameterSet.Config as cms

SimG4CoreHLTAODSIM = cms.PSet(
    outputCommands = cms.untracked.vstring('keep SimVertexs_g4SimHits_*_*')
)