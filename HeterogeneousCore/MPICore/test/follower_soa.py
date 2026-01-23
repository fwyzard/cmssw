import FWCore.ParameterSet.Config as cms

process = cms.Process("MPIClient")

process.options.numberOfThreads = 1
process.options.numberOfStreams = 1
process.options.numberOfConcurrentLuminosityBlocks = 1
process.options.numberOfConcurrentRuns = 1
process.options.wantSummary = False

process.load("HeterogeneousCore.MPIServices.MPIService_cfi")

from HeterogeneousCore.MPICore.modules import *

process.source = MPISource()

process.maxEvents.input = -1

# receive and validate a portable object, a portable collection, and some portable multicollections
process.load("Configuration.StandardSequences.Accelerators_cff")
process.load("HeterogeneousCore.AlpakaCore.ProcessAcceleratorAlpaka_cfi")


process.receiver = MPIReceiver(
    upstream = "source",
    instance = 42,
    products = cms.VPSet(cms.PSet(
        type = cms.string("PortableHostCollection<portabletest::TestSoALayout<128,false> >"),
        label = cms.string("")
     ),
     cms.PSet(
        type = cms.string("PortableHostObject<portabletest::TestStruct>"),
        label = cms.string("")
     ),
     cms.PSet(
        type = cms.string("PortableHostCollection<portabletest::SoABlocks2<128,false> >"),
        label = cms.string("")
     ),
     cms.PSet(
        type = cms.string("PortableHostCollection<portabletest::SoABlocks3<128,false> >"),
        label = cms.string("")
     ),
     cms.PSet(
       type = cms.string("ushort"),
       label = cms.string("backend")
     )
    )
)

process.validatePortableCollections = cms.EDAnalyzer("TestAlpakaAnalyzer",
    source = cms.InputTag("receiver")
)

process.validatePortableObject = cms.EDAnalyzer("TestAlpakaObjectAnalyzer",
    source = cms.InputTag("receiver")
)

process.pathSoA = cms.Path(
    process.receiver +
    process.validatePortableCollections +
    process.validatePortableObject
)
