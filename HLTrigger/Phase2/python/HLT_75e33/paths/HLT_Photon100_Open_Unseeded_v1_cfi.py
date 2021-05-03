import FWCore.ParameterSet.Config as cms

from ..modules.hltPrePhoton100OpenUnseeded_cfi import *
from ..sequences.HLTBeginSequence_cfi import *
from ..sequences.HLTEndSequence_cfi import *
from ..sequences.HLTPhoton100OpenUnseededSequence_cfi import *

HLT_Photon100_Open_Unseeded_v1 = cms.Path(HLTBeginSequence+hltPrePhoton100OpenUnseeded+HLTPhoton100OpenUnseededSequence+HLTEndSequence)
