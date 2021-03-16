import FWCore.ParameterSet.Config as cms

RecoMuonAOD = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep recoMuons_muons_*_*',
        'keep booledmValueMap_muons_*_*',
        'keep doubleedmValueMap_muons_muPFMean*_*',
        'keep doubleedmValueMap_muons_muPFSum*_*',
        'keep *_muons_muonShowerInformation_*',
        'keep recoMuonTimeExtraedmValueMap_muons_*_*',
        'keep recoMuonCosmicCompatibilityedmValueMap_muons_*_*',
        'keep uintedmValueMap_muons_*_*',
        'keep *_particleFlow_muons_*',
        'keep recoTracks_standAloneMuons_*_*',
        'keep recoTrackExtras_standAloneMuons_*_*',
        'keep TrackingRecHitsOwned_standAloneMuons_*_*',
        'keep recoTracks_globalMuons_*_*',
        'keep recoTrackExtras_globalMuons_*_*',
        'keep recoTracks_tevMuons_*_*',
        'keep recoTrackExtras_tevMuons_*_*',
        'keep recoTracks_generalTracks_*_*',
        'keep recoTracks_displacedTracks_*_*',
        'keep recoTracksToOnerecoTracksAssociation_tevMuons_*_*',
        'keep recoTracks_displacedGlobalMuons_*_*',
        'keep recoTrackExtras_displacedGlobalMuons_*_*',
        'keep TrackingRecHitsOwned_displacedGlobalMuons_*_*',
        'keep recoTracks_cosmicMuons_*_*',
        'keep recoMuons_muonsFromCosmics_*_*',
        'keep recoTracks_cosmicMuons1Leg_*_*',
        'keep recoMuons_muonsFromCosmics1Leg_*_*',
        'keep recoTracks_refittedStandAloneMuons_*_*',
        'keep recoTrackExtras_refittedStandAloneMuons_*_*',
        'keep TrackingRecHitsOwned_refittedStandAloneMuons_*_*',
        'keep recoTracks_displacedStandAloneMuons__*',
        'keep recoTrackExtras_displacedStandAloneMuons_*_*',
        'keep TrackingRecHitsOwned_displacedStandAloneMuons_*_*'
    )
)