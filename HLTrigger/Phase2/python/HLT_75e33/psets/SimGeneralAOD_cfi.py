import FWCore.ParameterSet.Config as cms

SimGeneralAOD = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep PileupSummaryInfos_*_*_*', 
        'keep int_*_bunchSpacing_*', 
        'keep *_genPUProtons_*_*'
    )
)