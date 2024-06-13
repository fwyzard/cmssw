#ifndef HeterogeousCore_MPICore_interface_MPICommunicator_h
#define HeterogeousCore_MPICore_interface_MPICommunicator_h

#include <mpi.h>
#include <string>
#include <map>
#include <cassert>
#include <iostream>
#include <memory>
#include <atomic>
class MPICommunicator {
  //May be we can move tags generation here in the future.
public:
  MPICommunicator(std::string);

  MPICommunicator();
  ~MPICommunicator();

  void publish_and_listen();

  void connect();
  void disconnect();

  MPI_Comm Communicator() const { return mainComm_; };
  std::tuple<int, int> rankAndSize(MPI_Comm comm) const;

private:
  MPI_Comm mainComm_;
  char port_[MPI_MAX_PORT_NAME];
  const std::string serviceName_ = "";
};

class MPIToken {
public:
  MPIToken() = default;
  MPIToken(MPICommunicator const* t) : token_(t) {};
  MPICommunicator const* token_;
};

#endif
