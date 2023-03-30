import FWCore.ParameterSet.Config as cms

process = cms.Process("MPIClient")

process.load("HeterogeneousCore.MPIServices.MPIService_cfi")
process.MPIService.pmix_server_uri = 'file:server.uri'

process.source = cms.Source("MPISource")

process.maxEvents.input = -1

process.options.numberOfThreads = 1
process.options.numberOfStreams = 1
# EventIDChecker requires synchronizing on LuminosityBlock boundaries
process.options.numberOfConcurrentLuminosityBlocks = 1

# FIXME this module needs to be updated to a different data product
#from HeterogeneousCore.MPICore.mpiReporter_cfi import mpiReporter as mpiReporter_
#process.mpiReporter = mpiReporter_.clone()
#process.path = cms.Path(process.mpiReporter)

from eventlist_cff import eventlist
process.check = cms.EDAnalyzer("EventIDChecker",
    eventSequence = cms.untracked(eventlist)
)

process.endp = cms.EndPath(process.check)
