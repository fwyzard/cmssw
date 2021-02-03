import FWCore.ParameterSet.Config as cms

LHEEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'drop *', 
        'keep LHERunInfoProduct_*_*_*', 
        'keep LHEEventProduct_*_*_*', 
        'keep *_externalLHEProducer_LHEScriptOutput_*'
    ),
    splitLevel = cms.untracked.int32(0)
)