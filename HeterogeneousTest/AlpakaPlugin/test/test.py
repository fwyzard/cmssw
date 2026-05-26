import FWCore.ParameterSet.Config as cms

process = cms.Process('TEST')
process.load('Configuration.StandardSequences.Accelerators_cff')
process.load('HeterogeneousCore.AlpakaCore.ProcessAcceleratorAlpaka_cfi')

process.source = cms.Source('EmptySource')

process.analyzer = cms.EDAnalyzer('Analyzer@alpaka',
    size = cms.uint32( 1024*1024 )
)

process.path = cms.Path(process.analyzer)

process.maxEvents.input = 1
