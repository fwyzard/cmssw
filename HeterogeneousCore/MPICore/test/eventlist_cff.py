import FWCore.ParameterSet.Config as cms

eventlist = cms.VEventID(
    # Run 100
    cms.EventID(100, 1, 1),
    cms.EventID(100, 1, 2),
    cms.EventID(100, 2, 3),
    cms.EventID(100, 2, 4),
    cms.EventID(100, 3, 5),
    # Run 101
    cms.EventID(101, 1, 1),
    cms.EventID(101, 1, 2),
    cms.EventID(101, 2, 3),
    cms.EventID(101, 2, 4),
    cms.EventID(101, 3, 5),
    # Run 102
    cms.EventID(102, 1, 1),
    cms.EventID(102, 1, 2),
    cms.EventID(102, 2, 3),
    cms.EventID(102, 2, 4),
    cms.EventID(102, 3, 5),
    # Run 103
    cms.EventID(103, 1, 1),
    cms.EventID(103, 1, 2),
    cms.EventID(103, 2, 3),
    cms.EventID(103, 2, 4),
    cms.EventID(103, 3, 5),
)
