import FWCore.ParameterSet.Config as cms

hltTripleMuon3DR0 = cms.EDFilter("HLT2L1TkMuonL1TkMuonMuRefDR",
    MinDR = cms.double(0),
    MinN = cms.int32(3),
    inputTag1 = cms.InputTag("hltL1TripleMuFiltered3"),
    inputTag2 = cms.InputTag("hltL1TripleMuFiltered3"),
    originTag1 = cms.VInputTag("L1TkMuons::HLTX"),
    originTag2 = cms.VInputTag("L1TkMuons::HLTX"),
    saveTags = cms.bool(True)
)
