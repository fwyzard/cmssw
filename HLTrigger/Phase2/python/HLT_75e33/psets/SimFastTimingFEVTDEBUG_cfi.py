import FWCore.ParameterSet.Config as cms

SimFastTimingFEVTDEBUG = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_mix_FTLBarrel_*', 
        'keep *_mix_FTLEndcap_*', 
        'keep *_mix_InitialVertices_*'
    )
)