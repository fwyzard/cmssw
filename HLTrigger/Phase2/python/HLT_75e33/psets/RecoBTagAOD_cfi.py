import FWCore.ParameterSet.Config as cms

RecoBTagAOD = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_softPFElectronBJetTags_*_*',
        'keep *_softPFMuonBJetTags_*_*',
        'keep *_pfTrackCountingHighEffBJetTags_*_*',
        'keep *_pfJetProbabilityBJetTags_*_*',
        'keep *_pfJetBProbabilityBJetTags_*_*',
        'keep *_pfSimpleSecondaryVertexHighEffBJetTags_*_*',
        'keep *_pfSimpleInclusiveSecondaryVertexHighEffBJetTags_*_*',
        'keep *_pfCombinedSecondaryVertexV2BJetTags_*_*',
        'keep *_pfCombinedInclusiveSecondaryVertexV2BJetTags_*_*',
        'keep *_pfGhostTrackBJetTags_*_*',
        'keep *_pfCombinedMVAV2BJetTags_*_*',
        'keep *_inclusiveCandidateSecondaryVertices_*_*',
        'keep *_inclusiveCandidateSecondaryVerticesCvsL_*_*',
        'keep *_pfCombinedCvsLJetTags_*_*',
        'keep *_pfCombinedCvsBJetTags_*_*',
        'keep *_pfChargeBJetTags_*_*',
        'keep *_pfDeepCSVJetTags_*_*',
        'keep *_pfDeepCMVAJetTags_*_*',
        'keep *_pixelClusterTagInfos_*_*'
    )
)