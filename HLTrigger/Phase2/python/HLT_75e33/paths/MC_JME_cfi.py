import FWCore.ParameterSet.Config as cms

from ..sequences.HLTJMESequence_cfi import *
from ..sequences.HLTParticleFlowSequence_cfi import *

MC_JME = cms.Path(HLTParticleFlowSequence+HLTJMESequence)
