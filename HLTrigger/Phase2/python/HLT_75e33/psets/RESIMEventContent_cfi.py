import FWCore.ParameterSet.Config as cms

RESIMEventContent = cms.PSet(
    inputCommands = cms.untracked.vstring(
        'drop *', 
        'keep *_randomEngineStateProducer_*_*', 
        'keep LHERunInfoProduct_*_*_*', 
        'keep LHEEventProduct_*_*_*', 
        'keep GenRunInfoProduct_generator_*_*', 
        'keep GenLumiInfoHeader_generator_*_*', 
        'keep GenLumiInfoProduct_generator_*_*', 
        'keep GenEventInfoProduct_generator_*_*', 
        'keep edmHepMCProduct_generatorSmeared_*_*', 
        'keep GenFilterInfo_*_*_*', 
        'keep *_genParticles_*_*'
    )
)