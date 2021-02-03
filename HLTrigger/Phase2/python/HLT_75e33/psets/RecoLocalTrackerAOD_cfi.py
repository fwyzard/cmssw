import FWCore.ParameterSet.Config as cms

RecoLocalTrackerAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('keep ClusterSummary_clusterSummaryProducer_*_*')
)