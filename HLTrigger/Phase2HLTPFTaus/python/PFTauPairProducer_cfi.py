import FWCore.ParameterSet.Config as cms

PFTauPairs = cms.EDProducer("PFTauPairProducer",
    srcPFTaus = cms.InputTag(''),
    srcPFTauSumChargedIso = cms.InputTag(''),
    min_pfTau_pt = cms.double(20.0), 
    max_pfTau_pt = cms.double(-1.),
    min_pfTau_absEta = cms.double(-1.),
    max_pfTau_absEta = cms.double(3.0),
    min_pfTau_sumChargedIso = cms.double(-1.),
    max_pfTau_sumChargedIso = cms.double(-1.),
    min_dR = cms.double(-1.),
    max_dR = cms.double(-1.),
    min_dz = cms.double(-1.),
    max_dz = cms.double(-1.),
)
