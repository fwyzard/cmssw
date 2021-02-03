import FWCore.ParameterSet.Config as cms

TcdsEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_tcdsDigis_*_*')
)