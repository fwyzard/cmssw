#include <iostream>
#include <mpi.h>

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "WrapperHandle.h"
#include "constants.h"
#include "serialization.h"

class SendAnalyzer : public edm::stream::EDAnalyzer<> {
public:
    explicit SendAnalyzer(const edm::ParameterSet &config);
    ~SendAnalyzer() override = default;
    static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
    void beginStream(edm::StreamID sid);
    void analyze(edm::Event const &event,
                 edm::EventSetup const &setup) override;

    unsigned int sid_;
    edm::EDGetTokenT<std::vector<double>> token_;
    edm::EDGetTokenT<int> cpuIDToken_;
    edm::EDGetTokenT<int> mpiTagToken_;
    edm::EDGetTokenT<std::map<std::string, double>> timesToken_;
};

SendAnalyzer::SendAnalyzer(const edm::ParameterSet &config)
    : token_(consumes<std::vector<double>>(
              config.getParameter<edm::InputTag>("result"))),
      cpuIDToken_(consumes<int>(config.getParameter<edm::InputTag>("cpuID"))),
      mpiTagToken_(consumes<int>(config.getParameter<edm::InputTag>("mpiTag"))),
      timesToken_(consumes<std::map<std::string, double>>(
              config.getParameter<edm::InputTag>("times"))) {}

void SendAnalyzer::beginStream(edm::StreamID sid) { sid_ = sid; }

void SendAnalyzer::analyze(edm::Event const &event,
                           edm::EventSetup const &setup) {
    edm::Handle<edm::WrapperBase> handle("std::vector<double>");
    edm::Handle<int> cpuIDHandle, mpiTagHandle;
    edm::Handle<std::map<std::string, double>> timesHandle;
    event.getByToken(token_, handle);
    auto buffer = serialize(*handle);
    event.getByToken(cpuIDToken_, cpuIDHandle);
    event.getByToken(mpiTagToken_, mpiTagHandle);
    event.getByToken(timesToken_, timesHandle);
    auto times = *timesHandle;

    times["jobEnd"] = MPI_Wtime();
#if DEBUG
    edm::LogPrint("SendAnalyzer")
            << "send sends tag " << *mpiTagHandle << ", stream " << sid_
            << " and cpuid " << *cpuIDHandle;
#endif
    MPI_Ssend(buffer.first.get(), buffer.second,
              MPI_CHAR, *cpuIDHandle, *mpiTagHandle,
              MPI_COMM_WORLD);
#if DEBUG
    edm::LogPrint("SendAnalyzer")
            << "stream " << sid_ << " finished sending data";
#endif
    times["sendResEnd"] = MPI_Wtime();

    std::vector<std::string> labels;
    std::vector<double> values;
    for (auto time : times) {
        labels.push_back(time.first);
        values.push_back(time.second);
    }
#if DEBUG
    for (unsigned int i = 0; i < labels.size(); i++) {
        edm::LogPrint("SendAnalyzer") << i << ": " << labels[i];
    }
#endif
    // MPI_Ssend(static_cast<void *>(labels.data()), sizeof(labels.data()),
    // MPI_CHAR,
    //           *cpuIDHandle, 0, MPI_COMM_WORLD);
#if DEBUG
    edm::LogPrint("SendAnalyzer")
            << "send sends times " << *mpiTagHandle << ", stream " << sid_
            << " and cpuid " << *cpuIDHandle;
#endif
    MPI_Ssend(static_cast<void const *>(values.data()), values.size(),
              MPI_DOUBLE, *cpuIDHandle, *cpuIDHandle, MPI_COMM_WORLD);
#if DEBUG
    edm::LogPrint("SendAnalyzer")
            << "stream " << sid_ << " finished sending times";
#endif
}

void SendAnalyzer::fillDescriptions(
        edm::ConfigurationDescriptions &descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("result", edm::InputTag());
    desc.add<edm::InputTag>("cpuID", edm::InputTag());
    desc.add<edm::InputTag>("mpiTag", edm::InputTag());
    desc.add<edm::InputTag>("times", edm::InputTag());
    descriptions.add("sendAnalyzer", desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SendAnalyzer);
