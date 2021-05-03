import FWCore.ParameterSet.Config as cms

from ..modules.hltPreEle5WP70OpenUnseeded_cfi import *
from ..sequences.HLTBeginSequence_cfi import *
from ..sequences.HLTEle5WP70OpenUnseededSequence_cfi import *
from ..sequences.HLTEndSequence_cfi import *

HLT_Ele5_WP70_Open_Unseeded_v1 = cms.Path(HLTBeginSequence+hltPreEle5WP70OpenUnseeded+HLTEle5WP70OpenUnseededSequence+HLTEndSequence)
