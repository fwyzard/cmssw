import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.scrambler = cms.EDAnalyzer("Scrambler",
    message = cms.untracked.string("Hello world!")
)

process.path = cms.Path( process.scrambler )

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32( 10 )
)

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool( False )
)
