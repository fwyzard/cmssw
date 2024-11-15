import FWCore.ParameterSet.Config as cms

process = cms.Process('TEST')
process.options.numberOfThreads = 4

process.source = cms.Source("EmptySource")
process.maxEvents.input = 10;

process.things = cms.EDProducer("ThingProducer")

process.p = cms.Path(process.things)

process.getter = cms.EDAnalyzer("ThingEventAnalyzer",
    input = cms.untracked.InputTag('things')
)

process.o = cms.EndPath(process.getter)
