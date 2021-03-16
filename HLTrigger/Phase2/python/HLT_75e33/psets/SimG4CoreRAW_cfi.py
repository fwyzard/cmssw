import FWCore.ParameterSet.Config as cms

SimG4CoreRAW = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_g4SimHits_*_*',
        'keep edmHepMCProduct_source_*_*'
    )
)