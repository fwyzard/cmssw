import FWCore.ParameterSet.Config as cms

process = cms.Process('Reader')

process.source = cms.Source('PoolSource',
    fileNames = cms.untracked.vstring('file:test.root')
)

process.load('FWCore.MessageService.MessageLogger_cfi')

process.AlpakaServiceSerialSync = cms.Service('AlpakaServiceSerialSync')
process.MessageLogger.AlpakaService = cms.untracked.PSet()

process.testAnalyzer = cms.EDAnalyzer('XyzIdAlpakaAnalyzer',
    source = cms.InputTag('testProducer')
)

process.testAnalyzerSerial = cms.EDAnalyzer('XyzIdAlpakaAnalyzer',
    source = cms.InputTag('testProducerSerial')
)

process.cuda_path = cms.Path(process.testAnalyzer)

process.serial_path = cms.Path(process.testAnalyzerSerial)

process.maxEvents.input = 10
