## MPIDriver class

This module runs inside a CMSSW job (the "driver") and connects to an "MPISource" in a separate CMSSW job (the "follower").
The follower is informed of all stream transitions seen by the driver, and can replicate them in its job.

Currently all communication is blocking, and there is no acknowledgment or feedback from the "follower" to the "driver".


## MPISource class

This input source runs inside a CMSSW job (the "follower") and opens a port to accept connections.
A separate CMSSW job (the "driver") can connect to it and send messages about the EDM transitions.


## Notes

  - let a "follower" job accept connections and events from multiple "drivers" ?
  - check that all drivers are processing the same Run; if a driver is processing a different Run, all the LuminosityBlocks,
    events (and the corresponding products) coming from it should be built and buffered and processed only when the ongoing
    Run is complete
  - check that all drivers are processing the same LuminosityBlock; if a driver is processing a different LuminosityBlock,
    all the events (and the corresponding products) coming from it should be built and buffered and processed only when the
    ongoing LuminosityBlock is complete
  - when a Run, LuminosityBlock or Event is received, check that they belong to the same ProcessingHistory as the ongoing Run,
    etc. ?
