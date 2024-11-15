import FWCore.ParameterSet.Config as cms

process = cms.Process("MPIServer")

process.options.numberOfStreams = 1
process.options.numberOfThreads = 1

process.maxEvents.input = -1

process.load("HeterogeneousCore.MPIServices.MPIService_cfi")
process.MPIService.pmix_server_uri = 'file:server.uri'

from eventlist_cff import eventlist
process.source = cms.Source("ConfigurableSource",
    events = cms.untracked(eventlist)
)

from HeterogeneousCore.MPICore.mpiDriver_cfi import mpiDriver as mpiDriver_
process.mpiDriver = mpiDriver_.clone()

process.things = cms.EDProducer("ThingProducer",
    offsetDelta = cms.int32(3),
    nThings = cms.int32(10)
)

process.analyzer = cms.EDAnalyzer("ThingEventAnalyzer",
    input = cms.untracked.InputTag('things')
)

process.sender = cms.EDProducer("MPISender",
    channel = cms.InputTag("mpiDriver"),
    tag = cms.int32(42),
    data =  cms.InputTag("things")
)

process.path = cms.Path(process.mpiDriver + process.things + process.analyzer + process.sender)
