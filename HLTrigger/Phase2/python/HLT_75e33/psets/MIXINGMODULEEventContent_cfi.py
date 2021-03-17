import FWCore.ParameterSet.Config as cms

MIXINGMODULEEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'drop *', 
        'keep *_cfWriter_*_*'
    ),
    splitLevel = cms.untracked.int32(0)
)