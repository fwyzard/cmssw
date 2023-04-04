#include <iostream>
#include <tuple>
#include <mpi.h>
#include <cassert>

#include "HeterogeneousCore/MPICore/interface/MPICommunicator.h"
#include "FWCore/Framework/interface/MakerMacros.h"

MPICommunicator::MPICommunicator(std::string serviceName) : serviceName_{std::move(serviceName)} {
  edm::LogAbsolute("MPI") << "MPICommunicator::MPICommunicator() is UP.";
}

MPICommunicator::~MPICommunicator(){}; 



void MPICommunicator::disconnect() {
  MPI_Comm_disconnect(&controlComm_); 
  MPI_Comm_disconnect(&dataComm_); 
  MPI_Comm_disconnect(&mainComm_);
  MPI_Info port_info;
  MPI_Info_create(&port_info);
  MPI_Unpublish_name(serviceName_.c_str(), port_info, port_);
  MPI_Close_port(port_);

  edm::LogAbsolute("MPI") << "MPICommunicator::disconnect()";
}

void MPICommunicator::connect() {
  edm::LogAbsolute log("MPI");
  log << "MPICommunicator::connect(). Lookup name " << serviceName_ << "\n";
  MPI_Lookup_name(serviceName_.c_str(), MPI_INFO_NULL, port_);
  MPI_Comm_connect(port_, MPI_INFO_NULL, 0, MPI_COMM_WORLD, &mainComm_);
  auto [rank, size] = rankAndSize(mainComm_);
  log << "MPICommunicator::connect(). MPISource Successfully Connected. MPIController rank = " << rank
      << " and size = " << size << ".";
}

void MPICommunicator::publish_and_listen() {
  edm::LogInfo log("MPI");

  //publish
  MPI_Open_port(MPI_INFO_NULL, port_);
  MPI_Info port_info;
  MPI_Info_create(&port_info);
  MPI_Info_set(port_info, "ompi_global_scope", "true");
  MPI_Info_set(port_info, "ompi_unique", "true");
  MPI_Publish_name(serviceName_.c_str(), port_info, port_);

  log << "MPICommunicator::publish_and_listen. Serivce successfully published its name " << serviceName_
      << " and waiting for MPIController to connect.";
  //listen
  MPI_Comm_accept(port_, MPI_INFO_NULL, 0, MPI_COMM_WORLD, &mainComm_);
  auto [rank, size] = rankAndSize(mainComm_);
  log << "MPICommunicator::publish_and_listen. MPIController is connected. MPISource rank = " << rank
      << " and size = " << size << ".";
}

void MPICommunicator::splitCommunicator() {
  //Key ties are broken according to original rank
  //color 0 for control Communicator
  MPI_Comm_split(mainComm_, 0, 0, &controlComm_);
  //color 1 for data communicator
  MPI_Comm_split(mainComm_, 1, 0, &dataComm_);

  auto [cRank, cSize] = rankAndSize(controlComm_);
  auto [dRank, dSize] = rankAndSize(dataComm_);
  edm::LogInfo log("MPI");
  log << "MPICommunicator::splitCommunicator(). Control Rank= " << cRank << " and size= " << cSize
      << ", Data Rank= " << dRank << ", size= " << dSize;
}

std::tuple<int, int> MPICommunicator::rankAndSize(MPI_Comm comm) const {
  int size;
  int rank;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
  return std::make_tuple(rank, size);
}
