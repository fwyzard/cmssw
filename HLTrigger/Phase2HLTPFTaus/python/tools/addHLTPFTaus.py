import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.ak4PFJets_cfi import ak4PFJets 

from RecoTauTag.RecoTau.PFRecoTauQualityCuts_cfi import PFTauQualityCuts
from RecoTauTag.RecoTau.RecoTauJetRegionProducer_cfi import RecoTauJetRegionProducer
from RecoTauTag.RecoTau.PFRecoTauChargedHadronProducer_cff import ak4PFJetsRecoTauChargedHadrons
import RecoTauTag.RecoTau.PFRecoTauChargedHadronBuilderPlugins_cfi as builders
import RecoTauTag.RecoTau.PFRecoTauChargedHadronQualityPlugins_cfi as ranking
from RecoTauTag.RecoTau.RecoTauPiZeroProducer_cff import ak4PFJetsLegacyHPSPiZeros
from RecoTauTag.RecoTau.RecoTauCombinatoricProducer_cfi import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByHPSSelection_cfi import *
from RecoTauTag.RecoTau.RecoTauCleaner_cfi import RecoTauCleaner
import RecoTauTag.RecoTau.RecoTauCleanerPlugins as cleaners
from RecoTauTag.RecoTau.RecoTauPiZeroUnembedder_cfi import RecoTauPiZeroUnembedder
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByLeadingObjectPtCut_cfi import pfRecoTauDiscriminationByLeadingObjectPtCut
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByIsolation_cfi import pfRecoTauDiscriminationByIsolation
from RecoTauTag.RecoTau.TauDiscriminatorTools import noPrediscriminants
from RecoTauTag.RecoTau.PFTauPrimaryVertexProducer_cfi import PFTauPrimaryVertexProducer
from RecoTauTag.RecoTau.PFTauSecondaryVertexProducer_cfi import PFTauSecondaryVertexProducer
from RecoTauTag.RecoTau.PFTauTransverseImpactParameters_cfi import PFTauTransverseImpactParameters

#----------------------------------------------------------------------------------------------------
# Define parameters of "hadron-plus-strips" (HPS) and shrinking-cone (SC) tau reconstruction algorithms

# CV: pT thresholds for charged PFCandidates to be considered when building tau candidate (minSignalTrackPt) 
#     and when computing the isolation of the tau candidate (minIsolationTrackPt)
#     Both thresholds are set to pT > 0.9 by default, motivated by this presentation
#       https://indico.cern.ch/event/921383/contributions/3879407/attachments/2044996/3426284/200526_hltupg_jme.pdf
#    (cf. "TRK v6" and "TRK v6 + skimTrk" on slides 3 and 5)
minSignalTrackPt      =  0.9
##minLeadTrackPt        =  5.0
minLeadTrackPt        =  1.0
minIsolationTrackPt   =  0.9

signalConeSize_hps    = "min(max(3.6/pt(), 0.08), 0.12)"
signalConeSize_sc     = "min(max(3.6/pt(), 0.08), 0.12)"

isolationConeSize_hps =  0.5
isolationConeSize_sc  =  0.5

minSeedJetPt          = 14.0
maxSeedJetAbsEta      =  4.0

minTauPt              = 15.0
maxTauAbsEta          =  3.0
#----------------------------------------------------------------------------------------------------

def addPFTauDiscriminator(process, discriminatorName, discriminator, pftauSequence):
    setattr(process, discriminatorName, discriminator)
    pftauSequence += discriminator
    return discriminator

def addPFTauSelector(process, selectorName, srcPFTaus, pftauDiscriminators, pftauSequence):
    selector = cms.EDFilter("PFTauSelector",
        src = cms.InputTag(srcPFTaus),
        cut = cms.string("pt > %1.2f & abs(eta) < %1.2f" % (minTauPt, maxTauAbsEta)),  
        discriminators = cms.VPSet([ cms.PSet( 
            discriminator = cms.InputTag(pftauDiscriminator), 
            selectionCut = cms.double(0.5) 
        ) for pftauDiscriminator in pftauDiscriminators ])
    )
    setattr(process, selectorName, selector)
    pftauSequence += selector
    return selector

#----------------------------------------------------------------------------------------------------
# CV: add additional tauID discriminators and tau lifetime information for DeepTau training

def addPFTauDiscriminatorsForDeepTau(process, 
                                     srcPFTaus, srcPFCandidates, srcVertices, 
                                     pftauQualityCuts,
                                     pftauSequence, 
                                     pfTauLabel, suffix):

    hltPFTauDiscriminationByDecayModeFindingNewDMs = addPFTauDiscriminator(process, "hlt%sDiscriminationByDecayModeFindingNewDMs%s" % (pfTauLabel, suffix),
        hpsSelectionDiscriminator.clone(
            PFTauProducer = cms.InputTag(srcPFTaus),
            decayModes = cms.VPSet(
                decayMode_1Prong0Pi0,
                decayMode_1Prong1Pi0,
                decayMode_1Prong2Pi0,
                decayMode_2Prong0Pi0,
                decayMode_2Prong1Pi0,
                decayMode_3Prong0Pi0,
                decayMode_3Prong1Pi0
            ),
            requireTauChargedHadronsToBeChargedPFCands = cms.bool(True),
            minPixelHits = cms.int32(1)
        ),
        pftauSequence)
    srcPFTauDiscriminationByDecayModeFindingNewDMs = hltPFTauDiscriminationByDecayModeFindingNewDMs.label()

    requireDecayMode = cms.PSet(
        BooleanOperator = cms.string("and"),
        decayMode = cms.PSet(
            Producer = cms.InputTag(srcPFTauDiscriminationByDecayModeFindingNewDMs),
            cut = cms.double(0.5)
        )
    )

    hpsPFTauBasicDiscriminators = addPFTauDiscriminator(process, "hlt%sBasicDiscriminators%s" % (pfTauLabel, suffix),
        pfRecoTauDiscriminationByIsolation.clone(
            PFTauProducer = cms.InputTag(srcPFTaus),
            particleFlowSrc = cms.InputTag(srcPFCandidates),
            vertexSrc = cms.InputTag(srcVertices),
            Prediscriminants = requireDecayMode,
            deltaBetaPUTrackPtCutOverride     = True, # Set the boolean = True to override.
            deltaBetaPUTrackPtCutOverride_val = 0.5,  # Set the value for new value.
            customOuterCone = 0.5,
            isoConeSizeForDeltaBeta = 0.8,
            deltaBetaFactor = "0.20",
            qualityCuts = pftauQualityCuts,
            IDdefinitions = cms.VPSet(
                cms.PSet(
                    IDname = cms.string("ChargedIsoPtSum"),
                    ApplyDiscriminationByTrackerIsolation = cms.bool(True),
                    storeRawSumPt = cms.bool(True)
                ),
                cms.PSet(
                    IDname = cms.string("NeutralIsoPtSum"),
                    ApplyDiscriminationByECALIsolation = cms.bool(True),
                    storeRawSumPt = cms.bool(True)
                ),
                cms.PSet(
                    IDname = cms.string("NeutralIsoPtSumWeight"),
                    ApplyDiscriminationByWeightedECALIsolation = cms.bool(True),
                    storeRawSumPt = cms.bool(True),
                    UseAllPFCandsForWeights = cms.bool(True)
                ),
                cms.PSet(
                    IDname = cms.string("TauFootprintCorrection"),
                    storeRawFootprintCorrection = cms.bool(True)
                ),
                cms.PSet(
                    IDname = cms.string("PhotonPtSumOutsideSignalCone"),
                    storeRawPhotonSumPt_outsideSignalCone = cms.bool(True)
                ),
                cms.PSet(
                    IDname = cms.string("PUcorrPtSum"),
                    applyDeltaBetaCorrection = cms.bool(True),
                    storeRawPUsumPt = cms.bool(True)
                ),
                cms.PSet(
                    IDname = cms.string("ByRawCombinedIsolationDBSumPtCorr3Hits"),
                    ApplyDiscriminationByTrackerIsolation = cms.bool(True),
                    ApplyDiscriminationByECALIsolation = cms.bool(True),
                    applyDeltaBetaCorrection = cms.bool(True),
                    storeRawSumPt = cms.bool(True)
                )
            )
        ),
        pftauSequence)

    hpsPFTauBasicDiscriminatorsdR03 = addPFTauDiscriminator(process, "hlt%sBasicDiscriminatorsdR03%s" % (pfTauLabel, suffix),
        hpsPFTauBasicDiscriminators.clone(
            customOuterCone = 0.3,
            deltaBetaFactor = '0.0720' # 0.2*(0.3/0.5)^2
        ),
        pftauSequence)

    return srcPFTauDiscriminationByDecayModeFindingNewDMs

def addPFTauTransverseImpactParametersForDeepTau(process, 
                                                 srcPFTaus, srcPFTauDiscriminationByDecayModeFinding, srcVertices, srcBeamSpot, 
                                                 pftauSequence, 
                                                 pfTauLabel, suffix):

    hltPFTauPrimaryVertexProducer = PFTauPrimaryVertexProducer.clone(
        PFTauTag = cms.InputTag(srcPFTaus),
        ElectronTag = cms.InputTag(""),
        MuonTag = cms.InputTag(""),
        PVTag = cms.InputTag(srcVertices),
        beamSpot = cms.InputTag(srcBeamSpot),
        Algorithm = cms.int32(0),
        useBeamSpot = cms.bool(True),
        RemoveMuonTracks = cms.bool(False),
        RemoveElectronTracks = cms.bool(False),
        useSelectedTaus = cms.bool(False),
        discriminators = cms.VPSet(
            cms.PSet(
                discriminator = cms.InputTag(srcPFTauDiscriminationByDecayModeFinding),
                selectionCut = cms.double(0.5)
            )
        ),
        cut = cms.string("pt > 20.0 & abs(eta) < 2.4") # CV: run vertex reconstruction only for taus passing pT and eta cuts in order not to waste computing time !!
    )
    setattr(process, "hlt%sPrimaryVertexProducer%s" % (pfTauLabel, suffix), hltPFTauPrimaryVertexProducer)
    pftauSequence += hltPFTauPrimaryVertexProducer
    srcPFTauPrimaryVertexProducer = hltPFTauPrimaryVertexProducer.label()

    hltPFTauSecondaryVertexProducer = PFTauSecondaryVertexProducer.clone(
        PFTauTag = cms.InputTag(srcPFTaus)
    )
    setattr(process, "hlt%sSecondaryVertexProducer%s" % (pfTauLabel, suffix), hltPFTauSecondaryVertexProducer)
    pftauSequence += hltPFTauSecondaryVertexProducer
    srcPFTauSecondaryVertexProducer = hltPFTauSecondaryVertexProducer.label()

    hltPFTauTransverseImpactParameters = PFTauTransverseImpactParameters.clone(
        PFTauTag = cms.InputTag(srcPFTaus),
        PFTauPVATag = cms.InputTag(srcPFTauPrimaryVertexProducer),
        PFTauSVATag = cms.InputTag(srcPFTauSecondaryVertexProducer),
        useFullCalculation = cms.bool(True)
    )
    setattr(process, "hlt%sTransverseImpactParameters%s" % (pfTauLabel, suffix), hltPFTauTransverseImpactParameters)
    pftauSequence += hltPFTauTransverseImpactParameters
#----------------------------------------------------------------------------------------------------

def addHLTPFTaus(process, algorithm, 
                 srcPFCandidates, srcVertices, srcBeamSpot, 
                 isolation_maxDeltaZ, isolation_maxDeltaZToLeadTrack, isolation_minTrackHits, 
                 suffix = ""):
    pfTauLabel = None
    signalConeSize = None
    isolationConeSize = None
    if algorithm == "shrinking-cone":
        pfTauLabel = "PFTau"
        signalConeSize = signalConeSize_sc
        isolationConeSize = isolationConeSize_sc
    elif algorithm == "hps":
        pfTauLabel = "HpsPFTau"
        signalConeSize = signalConeSize_hps
        isolationConeSize = isolationConeSize_hps
    else:
        raise ValueError("Invalid parameter algorithm = '%s' !!" % algorithm)

    pftauSequence = cms.Sequence()

    hltQualityCuts = PFTauQualityCuts.clone()
    hltQualityCuts.signalQualityCuts.minTrackPt = cms.double(minSignalTrackPt)
    hltQualityCuts.isolationQualityCuts.minTrackPt = cms.double(minIsolationTrackPt)
    hltQualityCuts.isolationQualityCuts.maxDeltaZ = cms.double(isolation_maxDeltaZ)
    hltQualityCuts.isolationQualityCuts.maxDeltaZToLeadTrack = cms.double(isolation_maxDeltaZToLeadTrack)
    hltQualityCuts.isolationQualityCuts.minTrackHits = cms.uint32(isolation_minTrackHits)
    hltQualityCuts.primaryVertexSrc = cms.InputTag(srcVertices)
    #------------------------------------------------------------------------------------------------
    # CV: fix for Phase-2 HLT tau trigger studies
    #    (pT of PFCandidates within HGCal acceptance is significantly higher than track pT !!)
    hltQualityCuts.leadingTrkOrPFCandOption = cms.string('minLeadTrackOrPFCand')
    #------------------------------------------------------------------------------------------------

    hltPFTauAK4PFJets = ak4PFJets.clone(
        src = cms.InputTag(srcPFCandidates),
        srcPVs = cms.InputTag(srcVertices)
    )
    setattr(process, "hlt%sAK4PFJets%s" % (pfTauLabel, suffix), hltPFTauAK4PFJets)
    pftauSequence += hltPFTauAK4PFJets
    srcPFTauAK4PFJets = hltPFTauAK4PFJets.label()

    #------------------------------------------------------------------------------------------------
    ## CV: restrict PFTau reconstruction to genuine taus when debbugging,
    ##     in order to avoid getting swamped by debug output !!
    ##genTaus = cms.EDFilter("GenParticleSelector",
    ##    src = cms.InputTag('genParticles'),
    ##    cut = cms.string('abs(pdgId) = 15 & status = 2 & pt > 20. & abs(eta) > 2.0 & abs(eta) < 2.4'),
    ##    stableOnly = cms.bool(True),
    ##    filter = cms.bool(False)
    ##) 
    ##setattr(process, "genTausFor%s%s" % (pfTauLabel, suffix), genTaus)
    ##pftauSequence += genTaus
    ##srcGenTaus = genTaus.label()
    ##
    ##genMatchedAK4PFJets = cms.EDFilter("PFJetAntiOverlapSelector",
    ##  src = cms.InputTag(srcPFTauAK4PFJets),
    ##  srcNotToBeFiltered = cms.VInputTag(srcGenTaus),
    ##  dRmin = cms.double(0.3),
    ##  invert = cms.bool(True),
    ##  filter = cms.bool(False)                                                          
    ##)
    ##setattr(process, "genMatchedAK4PFJetsFor%s%s" % (pfTauLabel, suffix), genMatchedAK4PFJets)
    ##pftauSequence += genMatchedAK4PFJets
    ##srcPFTauAK4PFJets = genMatchedAK4PFJets.label()
    #------------------------------------------------------------------------------------------------

    hltPFTau08Region = RecoTauJetRegionProducer.clone(
        src = cms.InputTag(srcPFTauAK4PFJets),
        pfCandSrc = cms.InputTag(srcPFCandidates),
        minJetPt = cms.double(minSeedJetPt),
        maxJetAbsEta = cms.double(maxSeedJetAbsEta)
    )
    setattr(process, "hlt%sPFJets08Region%s" % (pfTauLabel, suffix), hltPFTau08Region)
    pftauSequence += hltPFTau08Region
    srcPFTau08Region = hltPFTau08Region.label()

    builders_chargedPFCandidates = builders.chargedPFCandidates.clone(
        qualityCuts = hltQualityCuts
    )

    ranking_isChargedPFCandidate_withHGCalFix = cms.PSet(
        name = cms.string("ChargedPFCandidate"),
        plugin = cms.string("PFRecoTauChargedHadronQualityPluginHGCalWorkaround"),
    )

    hltPFTauPFJetsRecoTauChargedHadrons = ak4PFJetsRecoTauChargedHadrons.clone(
        jetSrc = cms.InputTag(srcPFTauAK4PFJets),
        minJetPt = cms.double(minSeedJetPt),
        maxJetAbsEta = cms.double(maxSeedJetAbsEta),
        outputSelection = cms.string('pt > %1.2f' % minSignalTrackPt),
        builders = cms.VPSet(
            builders_chargedPFCandidates
        ),
        ranking = cms.VPSet(
            ranking_isChargedPFCandidate_withHGCalFix
        )
    )
    setattr(process, "hlt%sPFJetsRecoTauChargedHadrons%s" % (pfTauLabel, suffix), hltPFTauPFJetsRecoTauChargedHadrons)
    pftauSequence += hltPFTauPFJetsRecoTauChargedHadrons
    srcPFTauPFJetsRecoTauChargedHadrons = hltPFTauPFJetsRecoTauChargedHadrons.label()

    hltPFTauPiZeros = ak4PFJetsLegacyHPSPiZeros.clone(
        jetSrc = cms.InputTag(srcPFTauAK4PFJets),
        minJetPt = cms.double(minSeedJetPt),
        maxJetAbsEta = cms.double(maxSeedJetAbsEta)
    )
    hltPFTauPiZeros.builders[0].qualityCuts = hltQualityCuts
    setattr(process, "hlt%sPiZeros%s" % (pfTauLabel, suffix), hltPFTauPiZeros)
    pftauSequence += hltPFTauPiZeros
    srcPFTauPiZeros = hltPFTauPiZeros.label()

    srcPFTausTmp = None
    if algorithm == "shrinking-cone":
        hltPFTausSansRef = combinatoricRecoTaus.clone(      
            jetSrc = cms.InputTag(srcPFTauAK4PFJets),
            minJetPt = cms.double(minSeedJetPt),
            maxJetAbsEta = cms.double(maxSeedJetAbsEta),
            jetRegionSrc = cms.InputTag(srcPFTau08Region),
            chargedHadronSrc = cms.InputTag(srcPFTauPFJetsRecoTauChargedHadrons),
            piZeroSrc = cms.InputTag(srcPFTauPiZeros),
            builders = cms.VPSet(cms.PSet( 
                plugin = cms.string("RecoTauBuilderConePlugin"),
                name = cms.string("shrinkingConeTau"),
                matchingCone = cms.string("0.3"),
                pfCandSrc = cms.InputTag(srcPFCandidates),
                usePFLeptons = cms.bool(True),
                leadObjectPt = cms.double(minSignalTrackPt),
                signalConeChargedHadrons = cms.string(signalConeSize),
                maxSignalConeChargedHadrons = cms.int32(3),
                signalConeNeutralHadrons = cms.string("0.1"),
                signalConePiZeros = cms.string(signalConeSize_sc),                
                minAbsPhotonSumPt_insideSignalCone = cms.double(2.5),
                minRelPhotonSumPt_insideSignalCone = cms.double(0.1),
                isoConeChargedHadrons = cms.string("%1.2f" % isolationConeSize),
                isoConeNeutralHadrons = cms.string("%1.2f" % isolationConeSize),
                isoConePiZeros = cms.string("%1.2f" % isolationConeSize),
                qualityCuts = hltQualityCuts,
            )),
            modifiers = cms.VPSet()
        )
        setattr(process, "hlt%ssSansRef%s" % (pfTauLabel, suffix), hltPFTausSansRef)
        pftauSequence += hltPFTausSansRef
        srcPFTausTmp = hltPFTausSansRef.label()
    elif algorithm == "hps":
        hltCombinatoricRecoTaus = combinatoricRecoTaus.clone(      
            jetSrc = cms.InputTag(srcPFTauAK4PFJets),
            minJetPt = cms.double(minSeedJetPt),
            maxJetAbsEta = cms.double(maxSeedJetAbsEta),
            jetRegionSrc = cms.InputTag(srcPFTau08Region),
            chargedHadronSrc = cms.InputTag(srcPFTauPFJetsRecoTauChargedHadrons),
            piZeroSrc = cms.InputTag(srcPFTauPiZeros),
            builders = cms.VPSet(cms.PSet(
                name = cms.string("combinatoric"),
                plugin = cms.string("RecoTauBuilderCombinatoricPlugin"),
                pfCandSrc = cms.InputTag(srcPFCandidates),
                isolationConeSize = cms.double(isolationConeSize_hps),
                qualityCuts = hltQualityCuts,
                decayModes = cms.VPSet(
                    combinatoricDecayModeConfigs.config1prong0pi0,
                    combinatoricDecayModeConfigs.config1prong1pi0,
                    combinatoricDecayModeConfigs.config1prong2pi0,
                    ##combinatoricDecayModeConfigs.config2prong0pi0,
                    ##combinatoricDecayModeConfigs.config2prong1pi0,
                    combinatoricDecayModeConfigs.config3prong0pi0,
       	            combinatoricDecayModeConfigs.config3prong1pi0
                ),
                signalConeSize = cms.string(signalConeSize_hps),
                minAbsPhotonSumPt_insideSignalCone = cms.double(2.5),
                minRelPhotonSumPt_insideSignalCone = cms.double(0.10),
                minAbsPhotonSumPt_outsideSignalCone = cms.double(1.e+9),
                minRelPhotonSumPt_outsideSignalCone = cms.double(1.e+9),
                verbosity = cms.int32(0)
            )),
            modifiers = cms.VPSet(cms.PSet(  
                name = cms.string("tau_mass"),
                plugin = cms.string("PFRecoTauMassPlugin"),
                verbosity = cms.int32(0)
            ))
        )
        setattr(process, "hlt%sCombinatoricProducer%s" % (pfTauLabel, suffix), hltCombinatoricRecoTaus)
        pftauSequence += hltCombinatoricRecoTaus
        srcCombinatoricRecoTaus = hltCombinatoricRecoTaus.label()

        hltPFTauSelectionDiscriminator = hpsSelectionDiscriminator.clone(
            PFTauProducer = cms.InputTag(srcCombinatoricRecoTaus),
            # CV: mass-window cuts taken from configuation used for HLT tau trigger during LHC Run 2
            decayModes = cms.VPSet(
                decayMode_1Prong0Pi0.clone(
                    maxMass = cms.string("1.")
                ),
                decayMode_1Prong1Pi0.clone(
                    maxMass = cms.string("max(1.72, min(1.72*sqrt(pt/100.), 4.2))")
                ),
                decayMode_1Prong2Pi0.clone(
                    maxMass = cms.string("max(1.72, min(1.72*sqrt(pt/100.), 4.0))")
                ),
                ##decayMode_2Prong0Pi0.clone(
                ##    maxMass = cms.string("1.2")
                ##),
                ##decayMode_2Prong1Pi0.clone(
                ##    maxMass = cms.string("max(1.6, min(1.6*sqrt(pt/100.), 4.0))")
                ##),
                decayMode_3Prong0Pi0.clone(
                    maxMass = cms.string("1.6")
                ),
                decayMode_3Prong1Pi0.clone(
                    maxMass = cms.string("1.6")
                )
            ),
            requireTauChargedHadronsToBeChargedPFCands = cms.bool(True)
        )
        setattr(process, "hlt%sSelectionDiscriminationByHPS%s" % (pfTauLabel, suffix), hltPFTauSelectionDiscriminator)
        pftauSequence += hltPFTauSelectionDiscriminator
        srcPFTauSelectionDiscriminationByHPS = hltPFTauSelectionDiscriminator.label()

        cleaner_leadTrackPt = cms.PSet(
            name = cms.string("leadTrackPt"),
            plugin = cms.string("RecoTauCleanerPluginHGCalWorkaround"),
            tolerance = cleaners.tolerance_default
        )

        hltPFTauCleaner = RecoTauCleaner.clone(
            src = cms.InputTag(srcCombinatoricRecoTaus),
            cleaners = cms.VPSet(
                cleaners.charge, # CV: to be disabled when using 2-prongs !!
                cms.PSet(  
                    name = cms.string("HPS_Select"),
                    plugin = cms.string("RecoTauDiscriminantCleanerPlugin"),
                    src = cms.InputTag(srcPFTauSelectionDiscriminationByHPS)
                ),
                cleaners.killSoftTwoProngTaus,
                cleaner_leadTrackPt,
                cleaners.chargedHadronMultiplicity,
                cleaners.pt,
                cleaners.stripMultiplicity,
                cleaners.chargeIsolation
            )
        )
        setattr(process, "hlt%sCleaner%s" % (pfTauLabel, suffix), hltPFTauCleaner)
        pftauSequence += hltPFTauCleaner
        srcPFTausTmp = hltPFTauCleaner.label()
    else:
        raise ValueError("Invalid parameter algorithm = '%s' !!" % algorithm)
    
    hltPFTaus = RecoTauPiZeroUnembedder.clone(
        src = cms.InputTag(srcPFTausTmp)
    )
    setattr(process, "hlt%ss%s" % (pfTauLabel, suffix), hltPFTaus)
    pftauSequence += hltPFTaus
    srcPFTaus = hltPFTaus.label()

    pftauDiscriminators = []

    hltPFTauDiscriminationByTrackFinding = addPFTauDiscriminator(process, "hlt%sDiscriminationByTrackFinding%s" % (pfTauLabel, suffix),
        pfRecoTauDiscriminationByLeadingObjectPtCut.clone(
            PFTauProducer = cms.InputTag(srcPFTaus),
            UseOnlyChargedHadrons = cms.bool(True),
            MinPtLeadingObject = cms.double(0.0)
        ),
        pftauSequence)
    pftauDiscriminators.append(hltPFTauDiscriminationByTrackFinding.label())

    hltPFTauDiscriminationByTrackPt = addPFTauDiscriminator(process, "hlt%sDiscriminationByTrackPt%s" % (pfTauLabel, suffix),
        pfRecoTauDiscriminationByLeadingObjectPtCut.clone(
            PFTauProducer = cms.InputTag(srcPFTaus),
            UseOnlyChargedHadrons = cms.bool(True),
            MinPtLeadingObject = cms.double(minLeadTrackPt)
        ),
        pftauSequence)
    pftauDiscriminators.append(hltPFTauDiscriminationByTrackPt.label())
       
    hltSelectedPFTaus = addPFTauSelector(process, "hltSelected%ss%s" % (pfTauLabel, suffix),
        srcPFTaus,
        pftauDiscriminators, 
        pftauSequence)
    srcSelectedPFTaus = hltSelectedPFTaus.label()

    addPFTauDiscriminator(process, "hlt%sDiscriminationByDecayModeFinding%s" % (pfTauLabel, suffix),
        hpsSelectionDiscriminator.clone(
            PFTauProducer = cms.InputTag(srcSelectedPFTaus),
            decayModes = cms.VPSet(
                decayMode_1Prong0Pi0,
                decayMode_1Prong1Pi0,
                decayMode_1Prong2Pi0,
                decayMode_3Prong0Pi0
            ),
            requireTauChargedHadronsToBeChargedPFCands = cms.bool(True),
            minPixelHits = cms.int32(1)
        ),
        pftauSequence)

    # CV: do not cut on charged isolation, but store charged isolation pT-sum in output file instead
    hltPFTauChargedIsoPtSumHGCalFix = addPFTauDiscriminator(process, "hltSelected%sChargedIsoPtSumHGCalFix%s" % (pfTauLabel, suffix),
        cms.EDProducer("PFRecoTauDiscriminationByIsolation",
            PFTauProducer = cms.InputTag(srcSelectedPFTaus),
            particleFlowSrc = cms.InputTag(srcPFCandidates),
            vertexSrc = cms.InputTag(srcVertices),
            qualityCuts = hltQualityCuts,
            Prediscriminants = noPrediscriminants,
            ApplyDiscriminationByTrackerIsolation = cms.bool(True),
            ApplyDiscriminationByECALIsolation = cms.bool(False),
            ApplyDiscriminationByWeightedECALIsolation = cms.bool(False),
            enableHGCalWorkaround = cms.bool(True),
            WeightECALIsolation = cms.double(0.),
            minTauPtForNoIso = cms.double(-99.),
            applyOccupancyCut = cms.bool(False),
            maximumOccupancy = cms.uint32(0),
            applySumPtCut = cms.bool(False),
            maximumSumPtCut = cms.double(-1.),
            applyRelativeSumPtCut = cms.bool(False),
            relativeSumPtCut = cms.double(-1.),
            relativeSumPtOffset = cms.double(0.),
            storeRawOccupancy = cms.bool(False),
            storeRawSumPt = cms.bool(True),
            storeRawPUsumPt = cms.bool(False),
            storeRawFootprintCorrection = cms.bool(False),
            storeRawPhotonSumPt_outsideSignalCone = cms.bool(False),
            customOuterCone = cms.double(-1.),
            applyPhotonPtSumOutsideSignalConeCut = cms.bool(False),
            maxAbsPhotonSumPt_outsideSignalCone = cms.double(1.e+9),
            maxRelPhotonSumPt_outsideSignalCone = cms.double(0.1),
            applyFootprintCorrection = cms.bool(False),
            footprintCorrections = cms.VPSet(),
            applyDeltaBetaCorrection = cms.bool(False),
            deltaBetaPUTrackPtCutOverride = cms.bool(False),
            deltaBetaPUTrackPtCutOverride_val = cms.double(0.5),
            isoConeSizeForDeltaBeta = cms.double(isolationConeSize),
            deltaBetaFactor = cms.string("0.20"),
            applyRhoCorrection = cms.bool(False),
            rhoProducer = cms.InputTag("NotUsed"),
            rhoConeSize = cms.double(0.357),
            rhoUEOffsetCorrection = cms.double(0.),
            UseAllPFCandsForWeights = cms.bool(False),
            verbosity = cms.int32(0)
        ),
        pftauSequence)

    hltPFTauChargedIsoPtSumdR03HGCalFix = addPFTauDiscriminator(process, "hltSelected%sChargedIsoPtSumdR03HGCalFix%s" % (pfTauLabel, suffix),
        hltPFTauChargedIsoPtSumHGCalFix.clone(
            customOuterCone = 0.3,
            deltaBetaFactor = '0.0720' # 0.2*(0.3/0.5)^2
        ),
        pftauSequence)

    hltPFTauNeutralIsoPtSumHGCalFix = addPFTauDiscriminator(process, "hltSelected%sNeutralIsoPtSumHGCalFix%s" % (pfTauLabel, suffix),
        hltPFTauChargedIsoPtSumHGCalFix.clone(
            ApplyDiscriminationByTrackerIsolation = cms.bool(False),
            ApplyDiscriminationByECALIsolation = cms.bool(True),
            WeightECALIsolation = cms.double(1.)
        ),
        pftauSequence)

    hltPFTauNeutralIsoPtSumdR03HGCalFix = addPFTauDiscriminator(process, "hltSelected%sNeutralIsoPtSumdR03HGCalFix%s" % (pfTauLabel, suffix),
        hltPFTauNeutralIsoPtSumHGCalFix.clone(
            customOuterCone = 0.3,
            deltaBetaFactor = '0.0720' # 0.2*(0.3/0.5)^2
        ),
        pftauSequence)

    # CV: add additional tauID discriminators and tau lifetime information for DeepTau training
    srcPFTauDiscriminationByDecayModeFindingNewDMs = addPFTauDiscriminatorsForDeepTau(process, 
        srcSelectedPFTaus, srcPFCandidates, srcVertices, 
        hltQualityCuts, 
        pftauSequence,
        pfTauLabel, suffix)
    addPFTauTransverseImpactParametersForDeepTau(process, 
        srcSelectedPFTaus, srcPFTauDiscriminationByDecayModeFindingNewDMs, srcVertices, srcBeamSpot, 
        pftauSequence,
        pfTauLabel, suffix)

    pftauSequenceName = "HLT%sSequence%s" % (pfTauLabel, suffix)
    setattr(process, pftauSequenceName, pftauSequence)
    return pftauSequence
