import FWCore.ParameterSet.Config as cms

DQMEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'drop *', 
        'keep *_MEtoEDMConverter_*_*'
    ),
    splitLevel = cms.untracked.int32(0)
)