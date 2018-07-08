import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool( False ),
    numberOfThreads = cms.untracked.uint32( 1 ),
    numberOfStreams = cms.untracked.uint32( 1 )
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:/data/store/data/Run2017F/SingleMuon/RAW/v1/000/306/121/00000/444F6D81-9CC0-E711-8D26-02163E01A1EA.root'
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32( -1 )
)

process.load("HLTrigger.Timer.eventTimestamp_cfi");

process.path = cms.Path(process.eventTimestamp)
