import FWCore.ParameterSet.Config as cms

import re

#----------------------------------------------------------------------------------------------------
# CV: Some part of the code in this file has been copied from 
#    1) tthAnalysis/NanoAOD/python/taus_updatedMVAIds_cff.py
#      ( https://github.com/HEP-KBFI/tth-nanoAOD/blob/master/python/taus_updatedMVAIds_cff.py )
#    2) HLTrigger/DeepTauTraining/test/produceDeepTau_rawNtuple_cfg.py
#      ( https://github.com/veelken/hlttrigger-deeptautraining/blob/master/test/produceDeepTau_rawNtuple_cfg.py )
#     with minor modifications
#----------------------------------------------------------------------------------------------------

#DeepTau_version = "2020Sep01wHGCalFix_training_v1"
DeepTau_version = "2021Feb19_training_v4"

def addDeepTauDiscriminator(process, hlt_srcPFTaus, hlt_srcPFJets, hlt_srcVertices, hlt_pfTauLabel, hlt_pfTauSuffix, deepTauSequenceName = "deepTauSequence"):

    deepTauSequence = cms.Sequence()
    setattr(process, deepTauSequenceName, deepTauSequence)

    if not hasattr(process, "hltFixedGridRhoAll"):
        from RecoJets.JetProducers.fixedGridRhoProducer_cfi import fixedGridRhoAll
        process.hltFixedGridRhoAll = fixedGridRhoAll.clone(
          pfCandidatesTag = cms.InputTag('particleFlowTmp')
        )
    deepTauSequence += process.hltFixedGridRhoAll

    if not hasattr(process, "hltPrimaryVertexAssociation"):
        from PhysicsTools.PatAlgos.slimming.primaryVertexAssociation_cfi import primaryVertexAssociation
        process.hltPrimaryVertexAssociation = primaryVertexAssociation.clone(
          particles = cms.InputTag('particleFlowTmp'),
          vertices = cms.InputTag(hlt_srcVertices),
          jets = cms.InputTag(hlt_srcPFJets)
        )
    deepTauSequence += process.hltPrimaryVertexAssociation

    if not hasattr(process, "hltPackedPFCandidates"):
        from PhysicsTools.PatAlgos.slimming.packedPFCandidates_cfi import packedPFCandidates
        process.hltPackedPFCandidates = packedPFCandidates.clone(
          inputCollection = cms.InputTag('particleFlowTmp'),
          inputVertices = cms.InputTag(hlt_srcVertices),
          originalVertices = cms.InputTag(hlt_srcVertices),
          originalTracks = cms.InputTag('generalTracks'),
          vertexAssociator = cms.InputTag('hltPrimaryVertexAssociation:original'),
          PuppiSrc = cms.InputTag(''),
          PuppiNoLepSrc = cms.InputTag(''),    
          chargedHadronIsolation = cms.InputTag(''),
          minPtForChargedHadronProperties = cms.double(0.9),
          secondaryVerticesForWhiteList = cms.VInputTag(),
          minPtForTrackProperties = cms.double(0.9)
        )
    deepTauSequence += process.hltPackedPFCandidates

    if not hasattr(process, "dummyIsolatedTracks"):
        process.dummyIsolatedTracks = cms.EDProducer("EmptyPATIsolatedTrackCollectionProducer")
    deepTauSequence += process.dummyIsolatedTracks

    ##if not hasattr(process, "patJetsHLTAK4PF"):
    ##    from PhysicsTools.PatAlgos.tools.jetTools import addJetCollection
    ##    addJetCollection(
    ##      process,
    ##      labelName = 'HLTAK4PF',
    ##      jetSource = cms.InputTag(hlt_srcPFJets),
    ##      btagDiscriminators = [ 'None' ],
    ##      genJetCollection = cms.InputTag('ak4GenJets'), 
    ##      jetCorrections = ( 'AK4PF', cms.vstring([ 'L2Relative', 'L3Absolute' ]), 'None' ))
    ##    process.makePatJets = cms.Sequence(process.patAlgosToolsTask)
    ##deepTauSequence += process.makePatJets
    ##
    ##if not hasattr(process, "hltSlimmedJets"):
    ##    from PhysicsTools.PatAlgos.slimming.slimmedJets_cfi import slimmedJets
    ##    process.hltSlimmedJets = slimmedJets.clone(
    ##      src = cms.InputTag('patJetsHLTAK4PF'),
    ##      packedPFCandidates = cms.InputTag("hltPackedPFCandidates")
    ##    )
    ##deepTauSequence += process.hltSlimmedJets

    # CV: produce pat::Tau collection
    #if not hasattr(process, "genParticles"):
    #    process.load("PhysicsTools.HepMCCandAlgos.genParticles_cfi")
    #deepTauSequence += process.genParticles

    from PhysicsTools.PatAlgos.mcMatchLayer0.tauMatch_cfi import tauMatch
    moduleName_tauMatch = "hlt%sMatch%s" % (hlt_pfTauLabel, hlt_pfTauSuffix)
    module_tauMatch = tauMatch.clone(
        src = cms.InputTag(hlt_srcPFTaus)
    )
    setattr(process, moduleName_tauMatch, module_tauMatch)
    deepTauSequence += module_tauMatch

    if not hasattr(process, "tauGenJets"):
        process.load("PhysicsTools.JetMCAlgos.TauGenJets_cfi")
    deepTauSequence += process.tauGenJets

    if not hasattr(process, "tauGenJetsSelectorAllHadrons"):
        process.load("PhysicsTools.JetMCAlgos.TauGenJetsDecayModeSelectorAllHadrons_cfi")
    deepTauSequence += process.tauGenJetsSelectorAllHadrons

    from PhysicsTools.PatAlgos.mcMatchLayer0.tauMatch_cfi import tauGenJetMatch
    moduleName_tauGenJetMatch = "hlt%sGenJetMatch%s" % (hlt_pfTauLabel, hlt_pfTauSuffix)
    module_tauGenJetMatch = tauGenJetMatch.clone(
        src = cms.InputTag(hlt_srcPFTaus)
    )
    setattr(process, moduleName_tauGenJetMatch, module_tauGenJetMatch)
    deepTauSequence += module_tauGenJetMatch

    from PhysicsTools.PatAlgos.producersLayer1.tauProducer_cfi import patTaus
    moduleName_patTaus = "hltPat%ss%s" % (hlt_pfTauLabel, hlt_pfTauSuffix)
    module_patTaus = patTaus.clone(
        tauSource = cms.InputTag(hlt_srcPFTaus),
        tauTransverseImpactParameterSource = cms.InputTag('hlt%sTransverseImpactParameters%s' % (hlt_pfTauLabel, hlt_pfTauSuffix)),
        genParticleMatch = cms.InputTag(moduleName_tauMatch),
        genJetMatch = cms.InputTag(moduleName_tauGenJetMatch),
        tauIDSources = cms.PSet()
    )
    from PhysicsTools.PatAlgos.producersLayer1.tauProducer_cfi import singleID, containerID
    singleID(module_patTaus.tauIDSources, 'hlt%sDiscriminationByDecayModeFinding%s' % (hlt_pfTauLabel, hlt_pfTauSuffix), "decayModeFinding")
    singleID(module_patTaus.tauIDSources, 'hlt%sDiscriminationByDecayModeFindingNewDMs%s' % (hlt_pfTauLabel, hlt_pfTauSuffix), "decayModeFindingNewDMs")
    singleID(module_patTaus.tauIDSources, 'hltSelected%sChargedIsoPtSumHGCalFix%s' % (hlt_pfTauLabel, hlt_pfTauSuffix), "chargedIsoPtSumHGCalFix")
    singleID(module_patTaus.tauIDSources, 'hltSelected%sNeutralIsoPtSumHGCalFix%s' % (hlt_pfTauLabel, hlt_pfTauSuffix), "neutralIsoPtSumHGCalFix")
    singleID(module_patTaus.tauIDSources, 'hltSelected%sChargedIsoPtSumdR03HGCalFix%s' % (hlt_pfTauLabel, hlt_pfTauSuffix), "chargedIsoPtSumdR03HGCalFix")
    singleID(module_patTaus.tauIDSources, 'hltSelected%sNeutralIsoPtSumdR03HGCalFix%s' % (hlt_pfTauLabel, hlt_pfTauSuffix), "neutralIsoPtSumdR03HGCalFix")
    containerID(module_patTaus.tauIDSources, 'hlt%sBasicDiscriminators%s' % (hlt_pfTauLabel, hlt_pfTauSuffix), "IDdefinitions", [
      [ "chargedIsoPtSum", "ChargedIsoPtSum" ],
      [ "neutralIsoPtSum", "NeutralIsoPtSum" ],
      [ "puCorrPtSum", "PUcorrPtSum" ],
      [ "neutralIsoPtSumWeight", "NeutralIsoPtSumWeight" ],
      [ "footprintCorrection", "TauFootprintCorrection" ],
      [ "photonPtSumOutsideSignalCone", "PhotonPtSumOutsideSignalCone" ],
      [ "byCombinedIsolationDeltaBetaCorrRaw3Hits", "ByRawCombinedIsolationDBSumPtCorr3Hits" ]
    ])
    containerID(module_patTaus.tauIDSources, 'hlt%sBasicDiscriminatorsdR03%s' % (hlt_pfTauLabel, hlt_pfTauSuffix), "IDdefinitions", [
      [ "chargedIsoPtSumdR03", "ChargedIsoPtSum" ],
      [ "neutralIsoPtSumdR03", "NeutralIsoPtSum" ],
      [ "puCorrPtSumdR03", "PUcorrPtSum" ],
      [ "neutralIsoPtSumWeightdR03", "NeutralIsoPtSumWeight" ],
      [ "footprintCorrectiondR03", "TauFootprintCorrection" ],
      [ "photonPtSumOutsideSignalConedR03", "PhotonPtSumOutsideSignalCone" ],
      [ "byCombinedIsolationDeltaBetaCorrRaw3HitsdR03", "ByRawCombinedIsolationDBSumPtCorr3Hits" ]
    ])
    setattr(process, moduleName_patTaus, module_patTaus)
    deepTauSequence += module_patTaus

    moduleName_selectedPatTaus = "hltSelectedPat%ss%s" % (hlt_pfTauLabel, hlt_pfTauSuffix)
    module_selectedPatTaus = cms.EDProducer("MyPATTauSelector",
      src = cms.InputTag(moduleName_patTaus),
      min_pt = cms.double(20.0),
      max_pt = cms.double(-1.),
      min_absEta = cms.double(-1.),
      max_absEta = cms.double(2.4),
      decayModes = cms.vint32(0, 1, 2, 10, 11),
      min_leadTrackPt = cms.double(1.0),
      max_leadTrackPt = cms.double(-1.),
      tauID_relChargedIso = cms.string("chargedIsoPtSumHGCalFix"),
      min_relChargedIso = cms.double(-1.),
      max_relChargedIso = cms.double(-1.),
      min_absChargedIso = cms.double(-1.),
      max_absChargedIso = cms.double(-1.),
      invert = cms.bool(False)
    )
    setattr(process, moduleName_selectedPatTaus, module_selectedPatTaus)
    deepTauSequence += module_selectedPatTaus

    from PhysicsTools.PatAlgos.slimming.slimmedTaus_cfi import slimmedTaus
    moduleName_slimmedTaus = "hltSlimmed%ss%s" % (hlt_pfTauLabel, hlt_pfTauSuffix)
    module_slimmedTaus = slimmedTaus.clone(
       src = cms.InputTag(moduleName_selectedPatTaus),
       packedPFCandidates = cms.InputTag('hltPackedPFCandidates'),
    )
    setattr(process, moduleName_slimmedTaus, module_slimmedTaus)
    deepTauSequence += module_slimmedTaus

    if not hasattr(process, "dummyElectrons"):
        process.dummyElectrons = cms.EDProducer("EmptyPATElectronCollectionProducer")
    deepTauSequence += process.dummyElectrons 

    if not hasattr(process, "dummyMuons"):
        process.dummyMuons = cms.EDProducer("EmptyPATMuonCollectionProducer")
    deepTauSequence += process.dummyMuons

    moduleName_deepTau_even = "hltDeep%sEven%s" % (hlt_pfTauLabel, hlt_pfTauSuffix)
    deepTau_inputFiles_even = [
      'core:HLTrigger/Phase2HLTPFTaus/data/%s/DeepTauPhase2HLTv2even_step1_final_core.pb' % DeepTau_version,
      'inner:HLTrigger/Phase2HLTPFTaus/data/%s/DeepTauPhase2HLTv2even_step1_final_inner.pb' % DeepTau_version,
      'outer:HLTrigger/Phase2HLTPFTaus/data/%s/DeepTauPhase2HLTv2even_step1_final_outer.pb' % DeepTau_version
    ]
    module_deepTau_even = cms.EDProducer("DeepTauId",
      electrons = cms.InputTag('dummyElectrons'),
      muons = cms.InputTag('dummyMuons'),
      taus = cms.InputTag(moduleName_slimmedTaus),
      pfcands = cms.InputTag('hltPackedPFCandidates'),
      vertices = cms.InputTag(hlt_srcVertices),
      rho = cms.InputTag('hltFixedGridRhoAll'),
      disable_hcalFraction_workaround = cms.bool(True),
      graph_file = cms.vstring(deepTau_inputFiles_even),
      mem_mapped = cms.bool(False),
      version = cms.uint32(2),
      VSeWP = cms.vstring(),
      VSmuWP = cms.vstring(),
      VSjetWP = cms.vstring(),
      debug_level = cms.int32(0),
      disable_dxy_pca = cms.bool(True), 
      is_phase2HLT = cms.bool(True),
      disable_CellIndex_workaround = cms.bool(True),
      disable_FlightLengthSig_workaround = cms.bool(True),
    )
    setattr(process, moduleName_deepTau_even, module_deepTau_even)
    deepTauSequence += module_deepTau_even

    moduleName_deepTau_odd = "hltDeep%sOdd%s" % (hlt_pfTauLabel, hlt_pfTauSuffix)
    deepTau_inputFiles_odd = [
      'core:HLTrigger/Phase2HLTPFTaus/data/%s/DeepTauPhase2HLTv2odd_step1_final_core.pb' % DeepTau_version,
      'inner:HLTrigger/Phase2HLTPFTaus/data/%s/DeepTauPhase2HLTv2odd_step1_final_inner.pb' % DeepTau_version,
      'outer:HLTrigger/Phase2HLTPFTaus/data/%s/DeepTauPhase2HLTv2odd_step1_final_outer.pb' % DeepTau_version
    ]
    module_deepTau_odd = module_deepTau_even.clone(
      graph_file  = cms.vstring(deepTau_inputFiles_odd)
    )
    setattr(process, moduleName_deepTau_odd, module_deepTau_odd)
    deepTauSequence += module_deepTau_odd

    ##if not hasattr(process, "printEventContent"):
    ##    process.printEventContent = cms.EDAnalyzer("EventContentAnalyzer")
    ##    deepTauSequence += process.printEventContent

    # CV: Embed DeepTau tau ID discriminators into pat::Tau object
    moduleName_updatedPatTaus_even = "hltUpdatedPat%ssEven%s" % (hlt_pfTauLabel, hlt_pfTauSuffix)
    module_updatedPatTaus_even = cms.EDProducer("PATTauIDEmbedder",
      src = cms.InputTag(moduleName_slimmedTaus),
      tauIDSources = cms.PSet(
        byDeepTau2017v2VSjetraw = cms.PSet(
          inputTag = cms.InputTag(moduleName_deepTau_even, 'VSjet'),
          workingPointIndex = cms.int32(-1)
        )
      )
    )
    setattr(process, moduleName_updatedPatTaus_even, module_updatedPatTaus_even)
    deepTauSequence += module_updatedPatTaus_even

    moduleName_updatedPatTaus_odd = "hltUpdatedPat%ssOdd%s" % (hlt_pfTauLabel, hlt_pfTauSuffix)
    module_updatedPatTaus_odd = cms.EDProducer("PATTauIDEmbedder",
      src = cms.InputTag(moduleName_slimmedTaus),
      tauIDSources = cms.PSet(
        byDeepTau2017v2VSjetraw = cms.PSet(
          inputTag = cms.InputTag(moduleName_deepTau_odd, 'VSjet'),
          workingPointIndex = cms.int32(-1)
        )
      )
    )
    setattr(process, moduleName_updatedPatTaus_odd, module_updatedPatTaus_odd)
    deepTauSequence += module_updatedPatTaus_odd

    # CV: Merge pat::Tau collections containing tau ID discriminator for DeepTau trained on even and odd events.
    #     Note that tau candidates in events with EVEN event numbers need to be classified
    #     using the DeepTau model trained on events with ODD event numbers and vice versa
    moduleName_updatedPatTaus = "hltUpdatedPat%ss%s" % (hlt_pfTauLabel, hlt_pfTauSuffix)
    module_updatedPatTaus = cms.EDProducer("PATTauEvenOddEventMixer",
      srcEvenEvents = cms.InputTag(moduleName_updatedPatTaus_odd),
      srcOddEvents = cms.InputTag(moduleName_updatedPatTaus_even)
    )
    setattr(process, moduleName_updatedPatTaus, module_updatedPatTaus)
    deepTauSequence += module_updatedPatTaus

    ##moduleName_dumpPatTaus = "dumpHLTPat%ss%s" % (hlt_pfTauLabel, hlt_pfTauSuffix)
    ##module_dumpPatTaus = cms.EDAnalyzer("DumpPATTaus",
    ##  src = cms.InputTag(moduleName_updatedPatTaus)
    ##)
    ##setattr(process, moduleName_dumpPatTaus, module_dumpPatTaus)
    ##deepTauSequence += module_dumpPatTaus

    return deepTauSequence
