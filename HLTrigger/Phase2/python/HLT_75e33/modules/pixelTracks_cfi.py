import FWCore.ParameterSet.Config as cms

pixelTracks = cms.EDProducer("PixelTrackProducer",
    Cleaner = cms.string('pixelTrackCleanerBySharedHits'),
    Filter = cms.InputTag("pixelTrackFilterByKinematics"),
    Fitter = cms.InputTag("pixelFitterByHelixProjections"),
    SeedingHitSets = cms.InputTag("pixelTracksHitSeeds"),
    mightGet = cms.untracked.vstring(
        '', 
        'RegionsSeedingHitSets_pixelTracksHitSeeds__RECO2', 
        'RegionsSeedingHitSets_pixelTracksHitSeeds__RECO2'
    ),
    passLabel = cms.string('pixelTracks')
)
