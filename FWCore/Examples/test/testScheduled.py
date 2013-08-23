import FWCore.ParameterSet.Config as cms

process = cms.Process( "TEST" )

process.options = cms.untracked.PSet(
    numberOfStreams = cms.untracked.uint32(  4 ),
    numberOfThreads = cms.untracked.uint32( 16),
    allowUnscheduled = cms.untracked.bool( False ),
    wantSummary = cms.untracked.bool( True )
)

process.Tracer = cms.Service( "Tracer" )

process.source = cms.Source( "EmptySource" )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32( 100 )
)

from FWCore.Examples.sampleProducer_cfi   import sampleProducer
from FWCore.Examples.sampleMerger_cfi     import sampleMerger
from FWCore.Examples.sampleFilter_cfi     import sampleFilter
from FWCore.Examples.sampleFilterMany_cfi import sampleFilterMany
from FWCore.Examples.sampleAnalyzer_cfi   import sampleAnalyzer

process.sampleProducerHelloWorld = sampleProducer.clone()
process.sampleProducerHelloWorld.data = [ "Hello world" ]

process.sampleProducerAnswer = sampleProducer.clone()
process.sampleProducerAnswer.data = [ "42" ]

process.sampleProducerOther = sampleProducer.clone()
process.sampleProducerOther.data = [ "That's all folks" ]

process.sampleMerger = sampleMerger.clone()
process.sampleMerger.source = [
    cms.InputTag('sampleProducerHelloWorld'),
    cms.InputTag('sampleProducerOther'),
]

process.sampleFilter = sampleFilterMany.clone()
process.sampleFilter.source = cms.VInputTag( cms.InputTag('sampleMerger'), cms.InputTag('sampleProducerAnswer') ) 
process.sampleFilter.pattern = '42'

process.path = cms.Path(
    process.sampleProducerHelloWorld +
    process.sampleProducerOther +
    process.sampleProducerAnswer +
    process.sampleMerger +
    process.sampleFilter )


subprocess = cms.Process("SUB")
process.addSubProcess(cms.SubProcess(process = subprocess, SelectEvents = cms.untracked.PSet(), outputCommands = cms.untracked.vstring()))

# not needed - the main process Tracer service sees also the signals from the subprocess
#subprocess.Tracer = cms.Service( "Tracer" )

subprocess.sampleAnalyzer = sampleAnalyzer.clone()
subprocess.sampleAnalyzer.source = cms.InputTag('sampleMerger', '', 'TEST')

subprocess.subpath = cms.Path( subprocess.sampleAnalyzer )
