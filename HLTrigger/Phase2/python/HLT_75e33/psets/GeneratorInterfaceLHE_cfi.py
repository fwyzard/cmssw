import FWCore.ParameterSet.Config as cms

GeneratorInterfaceLHE = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep LHERunInfoProduct_*_*_*', 
        'keep LHEEventProduct_*_*_*', 
        'keep *_externalLHEProducer_LHEScriptOutput_*'
    )
)