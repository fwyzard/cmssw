import FWCore.ParameterSet.Config as cms
from HeterogeneousCore.CUDACore.SwitchProducerCUDA import SwitchProducerCUDA

process = cms.Process('Writer')

process.source = cms.Source('EmptySource')

process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.Accelerators_cff')

process.AlpakaServiceCudaAsync = cms.Service('AlpakaServiceCudaAsync')
process.AlpakaServiceSerialSync = cms.Service('AlpakaServiceSerialSync')
process.MessageLogger.AlpakaService = cms.untracked.PSet()

process.testProducerCuda = cms.EDProducer('alpaka_cuda_async::XyzIdAlpakaProducer',
    size = cms.int32(42)
)

process.testTranscriberFromCuda = cms.EDProducer('alpaka_cuda_async::XyzIdAlpakaTranscriber',
    source = cms.InputTag('testProducerCuda')
)

process.testProducerCpu = cms.EDProducer('alpaka_serial_sync::XyzIdAlpakaProducer',
    size = cms.int32(42)
)

process.testProducer = SwitchProducerCUDA(
    cpu = cms.EDAlias(
        testProducerCpu = cms.VPSet(cms.PSet(type = cms.string('*')))
    ),
    cuda = cms.EDAlias(
        testTranscriberFromCuda = cms.VPSet(cms.PSet(type = cms.string('*')))
    )
)

process.testAnalyzer = cms.EDAnalyzer('XyzIdAlpakaAnalyzer',
    source = cms.InputTag('testProducer')
)

process.testProducerSerial = cms.EDProducer('alpaka_serial_sync::XyzIdAlpakaProducer',
    size = cms.int32(99)
)

process.testAnalyzerSerial = cms.EDAnalyzer('XyzIdAlpakaAnalyzer',
    source = cms.InputTag('testProducerSerial')
)

process.output = cms.OutputModule('PoolOutputModule',
    fileName = cms.untracked.string('test.root'),
    outputCommands = cms.untracked.vstring(
        'drop *',
        'keep *_testProducer_*_*',
        'keep *_testProducerSerial_*_*',
  )
)

process.producer_task = cms.Task(process.testProducerCuda, process.testTranscriberFromCuda, process.testProducerCpu)

process.process_path = cms.Path(
    process.testProducer +
    process.testAnalyzer,
    process.producer_task)

process.serial_path = cms.Path(
    process.testProducerSerial +
    process.testAnalyzerSerial)

process.output_path = cms.EndPath(process.output)

process.maxEvents.input = 10
