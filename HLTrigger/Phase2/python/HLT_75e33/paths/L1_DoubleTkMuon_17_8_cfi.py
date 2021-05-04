import FWCore.ParameterSet.Config as cms

from ..modules.hltDoubleMuon7DZ1p0_cfi import *
from ..modules.hltL1TkDoubleMuFiltered7_cfi import *
from ..modules.hltL1TkSingleMuFiltered15_cfi import *
from ..sequences.HLTBeginSequence_cfi import *
from ..sequences.HLTEndSequence_cfi import *

L1_DoubleTkMuon_17_8 = cms.Path(HLTBeginSequence+hltL1TkDoubleMuFiltered7+hltL1TkSingleMuFiltered15+hltDoubleMuon7DZ1p0+HLTEndSequence)
