import FWCore.ParameterSet.Config as cms

process = cms.Process("Writer")

process.source = cms.Source("EmptySource")

#process.load("FWCore.MessageService.MessageLogger_cfi")
#process.load("FWCore.Services.Tracer_cfi")

process.xyzIdAlpakaProducerCuda = cms.EDProducer('alpaka_cuda_async::XyzIdAlpakaProducer',
    size = cms.int32(42)
)

process.xyzIdAlpakaTranscriberFromCuda = cms.EDProducer('alpaka_cuda_async::XyzIdAlpakaTranscriber',
    source = cms.InputTag('xyzIdAlpakaProducerCuda')
)

process.xyzIdAlpakaAnalyzerFromCuda = cms.EDAnalyzer('alpaka_serial_sync::XyzIdAlpakaAnalyzer',
    source = cms.InputTag('xyzIdAlpakaTranscriberFromCuda')
)

process.xyzIdAlpakaProducerSerial = cms.EDProducer('alpaka_serial_sync::XyzIdAlpakaProducer',
    size = cms.int32(42)
)

process.xyzIdAlpakaAnalyzerSerial = cms.EDAnalyzer('alpaka_serial_sync::XyzIdAlpakaAnalyzer',
    source = cms.InputTag('xyzIdAlpakaProducerSerial')
)

process.output = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string("xyzid.root"),
    outputCommands = cms.untracked.vstring(
        'drop *',
        'keep *_xyzIdAlpakaTranscriberFromCuda_*_*',
        'keep *_xyzIdAlpakaProducerSerial_*_*',
  )
)

process.cuda_path = cms.Path(
    process.xyzIdAlpakaProducerCuda +
    process.xyzIdAlpakaTranscriberFromCuda +
    process.xyzIdAlpakaAnalyzerFromCuda)

process.serial_path = cms.Path(
    process.xyzIdAlpakaProducerSerial +
    process.xyzIdAlpakaAnalyzerSerial)

process.output_path = cms.EndPath(process.output)

process.maxEvents.input = 10
