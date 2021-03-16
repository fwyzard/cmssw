import FWCore.ParameterSet.Config as cms

RecoGenJetsRECO = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_ak4GenJets_*_*',
        'keep *_ak8GenJets_*_*',
        'keep *_ak4GenJetsNoNu_*_*',
        'keep *_ak8GenJetsNoNu_*_*',
        'keep *_genParticle_*_*'
    )
)