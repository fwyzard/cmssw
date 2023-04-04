#ifndef HeterogeousCore_MPICore_interface_MPICommunicator_h
#define HeterogeousCore_MPICore_interface_MPICommunicator_h

#include <mpi.h>
#include <string>
#include <map>
#include <cassert>



class MPICommunicator {
public:
  MPICommunicator(std::string serviceName);

  MPICommunicator();
  ~MPICommunicator();

  void publish_and_listen();
  void connect();
  void splitCommunicator();
  void disconnect(); 

  MPI_Comm mainCommunicator() const { return mainComm_; } ;
  MPI_Comm controlCommunicator() const { return controlComm_;} ;
  MPI_Comm dataCommunicator() const { return dataComm_; }; 
  std::tuple<int, int> rankAndSize(MPI_Comm) const;

private:
  MPI_Comm mainComm_;
  MPI_Comm controlComm_;
  MPI_Comm dataComm_;
  char port_[MPI_MAX_PORT_NAME];
  const std::string serviceName_ = "";
};

class MPIToken {
public:
  MPIToken() = default;
  MPIToken(MPICommunicator const* t) : token_(t){};
  MPIToken(MPICommunicator const* t, int tagID, int source) : token_(t), tagID_(tagID), source_(source){};
  MPICommunicator const* token_;
  int tagID_;
  int source_; 
};

#endif
