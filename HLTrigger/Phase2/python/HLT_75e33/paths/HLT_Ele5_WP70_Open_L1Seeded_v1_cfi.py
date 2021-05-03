import FWCore.ParameterSet.Config as cms

from ..modules.hltPreEle5WP70OpenL1Seeded_cfi import *
from ..sequences.HLTBeginSequence_cfi import *
from ..sequences.HLTEle5WP70OpenL1SeededSequence_cfi import *
from ..sequences.HLTEndSequence_cfi import *

HLT_Ele5_WP70_Open_L1Seeded_v1 = cms.Path(HLTBeginSequence+hltPreEle5WP70OpenL1Seeded+HLTEle5WP70OpenL1SeededSequence+HLTEndSequence)
