import FWCore.ParameterSet.Config as cms

from ..modules.hltPFPuppiMETTypeOne_cfi import *
from ..modules.hltPFPuppiMETTypeOne120_cfi import *
from ..modules.hltPFPuppiMETTypeOneCorrector_cfi import *
from ..modules.hltPFPuppiMHT_cfi import *
from ..modules.hltPFPuppiMHT120_cfi import *
from ..modules.l1tPFPuppiMET220off_cfi import *
from ..sequences.HLTAK4PFPuppiJetsReconstruction_cfi import *
from ..sequences.HLTParticleFlowSequence_cfi import *
from ..sequences.HLTPFPuppiMETReconstruction_cfi import *

HLT_PFPuppiMETTypeOne120_PFPuppiMHT120 = cms.Path(l1tPFPuppiMET220off+HLTParticleFlowSequence+HLTPFPuppiMETReconstruction+HLTAK4PFPuppiJetsReconstruction+hltPFPuppiMETTypeOneCorrector+hltPFPuppiMETTypeOne+hltPFPuppiMETTypeOne120+hltPFPuppiMHT+hltPFPuppiMHT120)
