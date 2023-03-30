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
    offsetDelta = cms.int32(100),
    nThings = cms.int32(50)
)

process.path = cms.Path(process.mpiDriver + process.things)
