import FWCore.ParameterSet.Config as cms

RecoLocalFastTimeAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_mtdClusters_*_*')
)