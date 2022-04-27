import FWCore.ParameterSet.Config as cms

process = cms.Process("Reader")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:xyzid.root')
)

#process.load("FWCore.MessageService.MessageLogger_cfi")
#process.load("FWCore.Services.Tracer_cfi")

process.xyzIdAlpakaAnalyzerFromCuda = cms.EDAnalyzer('alpaka_serial_sync::XyzIdAlpakaAnalyzer',
    source = cms.InputTag('xyzIdAlpakaTranscriberFromCuda')
)

process.xyzIdAlpakaAnalyzerSerial = cms.EDAnalyzer('alpaka_serial_sync::XyzIdAlpakaAnalyzer',
    source = cms.InputTag('xyzIdAlpakaProducerSerial')
)

process.cuda_path = cms.Path(process.xyzIdAlpakaAnalyzerFromCuda)

process.serial_path = cms.Path(process.xyzIdAlpakaAnalyzerSerial)

process.maxEvents.input = 10
