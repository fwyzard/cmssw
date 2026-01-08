import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.source = cms.Source("EmptySource")
process.maxEvents.input = 10

process.producer = cms.EDProducer("Producer",
    value = cms.int32(10)
)

process.path = cms.Path(
    process.producer
)

process.looper = cms.Looper("Looper",
    repeat = cms.uint32(3),
    module = cms.string("producer"),
    parameter = cms.string("value")
)
