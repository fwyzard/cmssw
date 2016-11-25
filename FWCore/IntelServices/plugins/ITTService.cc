#include <iostream>
#include <set>
#include <string>
#include <vector>
using namespace std::string_literals;

#include <boost/format.hpp>
#include <ittnotify.h>

#include "DataFormats/Common/interface/HLTPathStatus.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "DataFormats/Provenance/interface/RunID.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/ConsumesInfo.h"
#include "FWCore/ServiceRegistry/interface/GlobalContext.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "FWCore/ServiceRegistry/interface/PathContext.h"
#include "FWCore/ServiceRegistry/interface/PathsAndConsumesOfModulesBase.h"
#include "FWCore/ServiceRegistry/interface/ProcessContext.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "FWCore/ServiceRegistry/interface/StreamContext.h"
#include "FWCore/ServiceRegistry/interface/SystemBounds.h"
#include "FWCore/Utilities/interface/BranchType.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/ProductKindOfType.h"
#include "FWCore/Utilities/interface/TimeOfDay.h"
#include "HLTrigger/Timer/interface/ProcessCallGraph.h"

namespace edm {
  class ConfigurationDescriptions;
  class GlobalContext;
  class HLTPathStatus;
  class LuminosityBlock;
  class ModuleCallingContext;
  class ModuleDescription;
  class PathContext;
  class PathsAndConsumesOfModulesBase;
  class ProcessContext;
  class Run;
  class StreamContext;

  namespace service {
    class ITTService {
    public:
      ITTService(const ParameterSet&,ActivityRegistry&);
      ~ITTService() = default;

      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

      void preallocate(service::SystemBounds const&);

      // these signal pairs are not guaranteed to happen in the same thread

      void preBeginJob(PathsAndConsumesOfModulesBase const&, ProcessContext const&);
      void postBeginJob();
      // there is no preEndJob signal
      void postEndJob();

      void preGlobalBeginRun(edm::GlobalContext const&);
      void postGlobalBeginRun(edm::GlobalContext const&);
      void preGlobalEndRun(edm::GlobalContext const&);
      void postGlobalEndRun(edm::GlobalContext const&);

      void preGlobalBeginLumi(edm::GlobalContext const&);
      void postGlobalBeginLumi(edm::GlobalContext const&);
      void preGlobalEndLumi(edm::GlobalContext const&);
      void postGlobalEndLumi(edm::GlobalContext const&);

      void preStreamBeginRun(edm::StreamContext const&);
      void postStreamBeginRun(edm::StreamContext const&);
      void preStreamEndRun(edm::StreamContext const&);
      void postStreamEndRun(edm::StreamContext const&);

      void preStreamBeginLumi(edm::StreamContext const&);
      void postStreamBeginLumi(edm::StreamContext const&);
      void preStreamEndLumi(edm::StreamContext const&);
      void postStreamEndLumi(edm::StreamContext const&);

      void preEvent(edm::StreamContext const&);
      void postEvent(edm::StreamContext const&);

      void prePathEvent(edm::StreamContext const&, edm::PathContext const&);
      void postPathEvent(edm::StreamContext const&, edm::PathContext const&, edm::HLTPathStatus const&);

      void preModuleEventPrefetching(edm::StreamContext const&, edm::ModuleCallingContext const&);
      void postModuleEventPrefetching(edm::StreamContext const&, edm::ModuleCallingContext const&);

      // these signal pairs are guaranteed to be called within the same thread

      void preOpenFile(std::string const&, bool);
      void postOpenFile(std::string const&, bool);

      void preCloseFile(std::string const&, bool);
      void postCloseFile(std::string const&, bool);

      void preSourceConstruction(ModuleDescription const&);
      void postSourceConstruction(ModuleDescription const&);

      void preSourceRun();
      void postSourceRun();

      void preSourceLumi();
      void postSourceLumi();

      void preSourceEvent(StreamID);
      void postSourceEvent(StreamID);

      void preModuleConstruction(ModuleDescription const& md);
      void postModuleConstruction(ModuleDescription const& md);

      void preModuleBeginJob(ModuleDescription const& md);
      void postModuleBeginJob(ModuleDescription const& md);

      void preModuleEndJob(ModuleDescription const& md);
      void postModuleEndJob(ModuleDescription const& md);

      void preModuleGlobalBeginRun(GlobalContext const&, ModuleCallingContext const&);
      void postModuleGlobalBeginRun(GlobalContext const&, ModuleCallingContext const&);
      void preModuleGlobalEndRun(GlobalContext const&, ModuleCallingContext const&);
      void postModuleGlobalEndRun(GlobalContext const&, ModuleCallingContext const&);

      void preModuleGlobalBeginLumi(GlobalContext const&, ModuleCallingContext const&);
      void postModuleGlobalBeginLumi(GlobalContext const&, ModuleCallingContext const&);
      void preModuleGlobalEndLumi(GlobalContext const&, ModuleCallingContext const&);
      void postModuleGlobalEndLumi(GlobalContext const&, ModuleCallingContext const&);

      void preModuleBeginStream(StreamContext const&, ModuleCallingContext const&);
      void postModuleBeginStream(StreamContext const&, ModuleCallingContext const&);
      void preModuleEndStream(StreamContext const&, ModuleCallingContext const&);
      void postModuleEndStream(StreamContext const&, ModuleCallingContext const&);

      void preModuleStreamBeginRun(StreamContext const&, ModuleCallingContext const&);
      void postModuleStreamBeginRun(StreamContext const&, ModuleCallingContext const&);
      void preModuleStreamEndRun(StreamContext const&, ModuleCallingContext const&);
      void postModuleStreamEndRun(StreamContext const&, ModuleCallingContext const&);

      void preModuleStreamBeginLumi(StreamContext const&, ModuleCallingContext const&);
      void postModuleStreamBeginLumi(StreamContext const&, ModuleCallingContext const&);
      void preModuleStreamEndLumi(StreamContext const&, ModuleCallingContext const&);
      void postModuleStreamEndLumi(StreamContext const&, ModuleCallingContext const&);

      void preEventReadFromSource(StreamContext const&, ModuleCallingContext const&);
      void postEventReadFromSource(StreamContext const&, ModuleCallingContext const&);

      void preModuleEvent(StreamContext const&, ModuleCallingContext const&);
      void postModuleEvent(StreamContext const&, ModuleCallingContext const&);

      void preModuleEventDelayedGet(StreamContext const&, ModuleCallingContext const&);
      void postModuleEventDelayedGet(StreamContext const&, ModuleCallingContext const&);

    private:
      
      struct itt_identifier {
        __itt_string_handle * label;
        __itt_id              id;

        itt_identifier() :
          label(nullptr),
          id(__itt_null)
        { }

        itt_identifier(__itt_string_handle * l, __itt_id i) :
          label(l),
          id(i)
        { }

        itt_identifier(__itt_domain * domain, const char * name) :
          label(__itt_string_handle_create(name)),
          id(__itt_id_make(domain, reinterpret_cast<unsigned long long>(label)))
        {
          __itt_id_create(domain, id);
        }

        itt_identifier(__itt_domain * domain, std::string const& name) :
          label(__itt_string_handle_create(name.c_str())),
          id(__itt_id_make(domain, reinterpret_cast<unsigned long long>(label)))
        {
          __itt_id_create(domain, id);
        }

        void destroy() {
          // (ab)use the high QWORD of the ID value to store a pointer to its creator __itt_domain
          __itt_domain * domain = reinterpret_cast<__itt_domain *>(id.d1);
          __itt_id_destroy(domain, id);
        }

      };


      // modules and paths
      ProcessCallGraph callgraph_;

      // event map, used at construction time
      // (assume that the framework coould construct different modules in parallel)
      tbb::concurrent_unordered_map<unsigned int, __itt_event> ctor_events_;

      // domains
      __itt_domain *              globalDomain_;
      std::vector<__itt_domain *> streamDomains_;

      // module tasks
      std::vector<itt_identifier> tasks_;

      // regions for concurrent runs, lumisections, events
      std::vector<itt_identifier> runs_;
      std::vector<itt_identifier> lumis_;
      std::vector<itt_identifier> events_;
    };
  }
}

using namespace edm::service;

ITTService::ITTService(ParameterSet const& config, ActivityRegistry& registry) :
  globalDomain_(__itt_domain_create("cern.cms.global")),
  streamDomains_()
{
  // make sure ITT collection is enabled
  __itt_resume();

  registry.watchPreallocate(this, &ITTService::preallocate);

  // these signal pairs are not guaranteed to happen in the same thread

  registry.watchPreBeginJob(this, &ITTService::preBeginJob);
  registry.watchPostBeginJob(this, &ITTService::postBeginJob);
  // there is no preEndJob signal
  registry.watchPostEndJob(this, &ITTService::postEndJob);

  registry.watchPreGlobalBeginRun(this, &ITTService::preGlobalBeginRun);
  registry.watchPostGlobalBeginRun(this, &ITTService::postGlobalBeginRun);
  registry.watchPreGlobalEndRun(this, &ITTService::preGlobalEndRun);
  registry.watchPostGlobalEndRun(this, &ITTService::postGlobalEndRun);

  registry.watchPreGlobalBeginLumi(this, &ITTService::preGlobalBeginLumi);
  registry.watchPostGlobalBeginLumi(this, &ITTService::postGlobalBeginLumi);
  registry.watchPreGlobalEndLumi(this, &ITTService::preGlobalEndLumi);
  registry.watchPostGlobalEndLumi(this, &ITTService::postGlobalEndLumi);

  registry.watchPreStreamBeginRun(this, &ITTService::preStreamBeginRun);
  registry.watchPostStreamBeginRun(this, &ITTService::postStreamBeginRun);
  registry.watchPreStreamEndRun(this, &ITTService::preStreamEndRun);
  registry.watchPostStreamEndRun(this, &ITTService::postStreamEndRun);

  registry.watchPreStreamBeginLumi(this, &ITTService::preStreamBeginLumi);
  registry.watchPostStreamBeginLumi(this, &ITTService::postStreamBeginLumi);
  registry.watchPreStreamEndLumi(this, &ITTService::preStreamEndLumi);
  registry.watchPostStreamEndLumi(this, &ITTService::postStreamEndLumi);

  registry.watchPreEvent(this, &ITTService::preEvent);
  registry.watchPostEvent(this, &ITTService::postEvent);

  registry.watchPrePathEvent(this, &ITTService::prePathEvent);
  registry.watchPostPathEvent(this, &ITTService::postPathEvent);

  registry.watchPreModuleEventPrefetching(this, &ITTService::preModuleEventPrefetching);
  registry.watchPostModuleEventPrefetching(this, &ITTService::postModuleEventPrefetching);

  // these signal pairs are guaranteed to be called within the same thread

  registry.watchPreOpenFile(this, &ITTService::preOpenFile);
  registry.watchPostOpenFile(this, &ITTService::postOpenFile);

  registry.watchPreCloseFile(this, &ITTService::preCloseFile);
  registry.watchPostCloseFile(this, &ITTService::postCloseFile);

  registry.watchPreSourceConstruction(this, &ITTService::preSourceConstruction);
  registry.watchPostSourceConstruction(this, &ITTService::postSourceConstruction);

  registry.watchPreSourceRun(this, &ITTService::preSourceRun);
  registry.watchPostSourceRun(this, &ITTService::postSourceRun);

  registry.watchPreSourceLumi(this, &ITTService::preSourceLumi);
  registry.watchPostSourceLumi(this, &ITTService::postSourceLumi);

  registry.watchPreSourceEvent(this, &ITTService::preSourceEvent);
  registry.watchPostSourceEvent(this, &ITTService::postSourceEvent);

  registry.watchPreModuleConstruction(this, &ITTService::preModuleConstruction);
  registry.watchPostModuleConstruction(this, &ITTService::postModuleConstruction);

  registry.watchPreModuleBeginJob(this, &ITTService::preModuleBeginJob);
  registry.watchPostModuleBeginJob(this, &ITTService::postModuleBeginJob);

  registry.watchPreModuleEndJob(this, &ITTService::preModuleEndJob);
  registry.watchPostModuleEndJob(this, &ITTService::postModuleEndJob);

  registry.watchPreModuleGlobalBeginRun(this, &ITTService::preModuleGlobalBeginRun);
  registry.watchPostModuleGlobalBeginRun(this, &ITTService::postModuleGlobalBeginRun);
  registry.watchPreModuleGlobalEndRun(this, &ITTService::preModuleGlobalEndRun);
  registry.watchPostModuleGlobalEndRun(this, &ITTService::postModuleGlobalEndRun);

  registry.watchPreModuleGlobalBeginLumi(this, &ITTService::preModuleGlobalBeginLumi);
  registry.watchPostModuleGlobalBeginLumi(this, &ITTService::postModuleGlobalBeginLumi);
  registry.watchPreModuleGlobalEndLumi(this, &ITTService::preModuleGlobalEndLumi);
  registry.watchPostModuleGlobalEndLumi(this, &ITTService::postModuleGlobalEndLumi);

  registry.watchPreModuleBeginStream(this, &ITTService::preModuleBeginStream);
  registry.watchPostModuleBeginStream(this, &ITTService::postModuleBeginStream);
  registry.watchPreModuleEndStream(this, &ITTService::preModuleEndStream);
  registry.watchPostModuleEndStream(this, &ITTService::postModuleEndStream);

  registry.watchPreModuleStreamBeginRun(this, &ITTService::preModuleStreamBeginRun);
  registry.watchPostModuleStreamBeginRun(this, &ITTService::postModuleStreamBeginRun);
  registry.watchPreModuleStreamEndRun(this, &ITTService::preModuleStreamEndRun);
  registry.watchPostModuleStreamEndRun(this, &ITTService::postModuleStreamEndRun);

  registry.watchPreModuleStreamBeginLumi(this, &ITTService::preModuleStreamBeginLumi);
  registry.watchPostModuleStreamBeginLumi(this, &ITTService::postModuleStreamBeginLumi);
  registry.watchPreModuleStreamEndLumi(this, &ITTService::preModuleStreamEndLumi);
  registry.watchPostModuleStreamEndLumi(this, &ITTService::postModuleStreamEndLumi);

  registry.watchPreEventReadFromSource(this, &ITTService::preEventReadFromSource);
  registry.watchPostEventReadFromSource(this, &ITTService::postEventReadFromSource);

  registry.watchPreModuleEvent(this, &ITTService::preModuleEvent);
  registry.watchPostModuleEvent(this, &ITTService::postModuleEvent);

  registry.watchPreModuleEventDelayedGet(this, &ITTService::preModuleEventDelayedGet);
  registry.watchPostModuleEventDelayedGet(this, &ITTService::postModuleEventDelayedGet);

  registry.preSourceEarlyTerminationSignal_.connect([this](edm::TerminationOrigin origin) {
      // early termination during source processing
  });

  registry.preGlobalEarlyTerminationSignal_.connect([this](edm::GlobalContext const& gc, edm::TerminationOrigin origin) {
    if (gc.luminosityBlockID().value() == 0) {
      // early termination during global run processing
      // run:    gc.luminosityBlockID().run()
    } else {
      // early termination during global lumisection processing
      // run:    gc.luminosityBlockID().run()
      // lumi:   gc.luminosityBlockID().luminosityBlock()
    }
  });

  registry.preStreamEarlyTerminationSignal_.connect([this](edm::StreamContext const& sc, edm::TerminationOrigin origin) {
    if (sc.eventID().luminosityBlock() == 0) {
      // early termination during stream run processing
      // stream: sc.streamID()
      // run:    sc.eventID().run()
    } else if (sc.eventID().event() == 0) {
      // early termination during stream lumisection processing
      // stream: sc.streamID()
      // run:    sc.eventID().run()
      // lumi:   sc.eventID().luminosityBlock()
    } else {
      // early termination during event processing
      // stream: sc.streamID()
      // run:    sc.eventID().run()
      // lumi:   sc.eventID().luminosityBlock()
      // event:  sc.eventID().event()
    }
  });

}

void
ITTService::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
  edm::ParameterSetDescription desc;
  descriptions.add("ITTService", desc);
  descriptions.setComment("This service monitors via ITT/SEAPI each phase the framework is processing, e.g. constructing a module,running a module, etc.");
}

void
ITTService::preallocate(service::SystemBounds const& bounds) {
  streamDomains_.resize(bounds.maxNumberOfStreams());
  for (unsigned int i = 0; i < bounds.maxNumberOfStreams(); ++i)
    streamDomains_[i] = __itt_domain_create((boost::format("cern.cms.stream %d") % i).str().c_str());

  runs_.resize(bounds.maxNumberOfConcurrentRuns());
  lumis_.resize(bounds.maxNumberOfConcurrentLuminosityBlocks());
  events_.resize(bounds.maxNumberOfStreams());

  std::stringstream out;
  out << "preallocate: " << bounds.maxNumberOfConcurrentRuns() << " concurrent runs, "
                         << bounds.maxNumberOfConcurrentLuminosityBlocks() << " concurrent luminosity sections, "
                         << bounds.maxNumberOfStreams() << " concurrent streams";
  std::string label = out.str();
  __itt_event event = __itt_event_create(label.c_str(), label.size());
  __itt_event_start(event);
}

void
ITTService::preSourceConstruction(ModuleDescription const& desc) {
  callgraph_.preSourceConstruction(desc);

  std::string label = (boost::format("constructitng source of type '%s'") % desc.moduleName()).str();
  __itt_event event = __itt_event_create(label.c_str(), label.size());
  ctor_events_.emplace(desc.id(), event);
  __itt_event_start(event);
}

void
ITTService::postSourceConstruction(ModuleDescription const& desc) {
  __itt_event_end(ctor_events_.at(desc.id()));
}

void
ITTService::preModuleConstruction(ModuleDescription const& desc) {
  std::string label = (boost::format("constructitng module '%s' of type '%s'") % desc.moduleLabel() % desc.moduleName()).str();
  __itt_event event = __itt_event_create(label.c_str(), label.size());
  ctor_events_.emplace(desc.id(), event);
  __itt_event_start(event);
}

void
ITTService::postModuleConstruction(ModuleDescription const& desc) {
  __itt_event_end(ctor_events_.at(desc.id()));
}

void
ITTService::preBeginJob(PathsAndConsumesOfModulesBase const& pathsAndConsumes, ProcessContext const& context) {
  callgraph_.preBeginJob(pathsAndConsumes, context);

  std::string label = (boost::format("job initialisation for process '%s'") % context.processName()).str();
  __itt_event event = __itt_event_create(label.c_str(), label.size());
  __itt_event_start(event);
}

void
ITTService::postBeginJob() {
  unsigned int size = callgraph_.size();
  tasks_.resize(size);
  for (unsigned int i = 0; i < size; ++i)
    tasks_[i] = itt_identifier(globalDomain_, callgraph_.module(i).moduleLabel());

  ctor_events_.clear();

  std::string label = "job initialisation done"s;
  __itt_event event = __itt_event_create(label.c_str(), label.size());
  __itt_event_start(event);
}

void
ITTService::postEndJob() {
  std::string label = "job done"s;
  __itt_event event = __itt_event_create(label.c_str(), label.size());
  __itt_event_start(event);
}

void
ITTService::preSourceEvent(StreamID sid) {
  //std::cerr << "starting: source event" << std::endl;
}

void
ITTService::postSourceEvent(StreamID sid) {
  //std::cerr << "finished: source event" << std::endl;
}

void
ITTService::preSourceLumi() {
  //std::cerr << "starting: source lumi" << std::endl;
}

void
ITTService::postSourceLumi() {
  //std::cerr << "finished: source lumi" << std::endl;
}

void
ITTService::preSourceRun() {
  //std::cerr << "starting: source run" << std::endl;
}

void
ITTService::postSourceRun() {
  //std::cerr << "finished: source run" << std::endl;
}

void
ITTService::preOpenFile(std::string const& lfn, bool b) {
  //std::cerr << "starting: open input file: lfn = " << lfn << std::endl;
}

void
ITTService::postOpenFile (std::string const& lfn, bool b) {
  //std::cerr << "finished: open input file: lfn = " << lfn << std::endl;
}

void
ITTService::preCloseFile(std::string const & lfn, bool b) {
  //std::cerr << "starting: close input file: lfn = " << lfn << std::endl;
}
void
ITTService::postCloseFile (std::string const& lfn, bool b) {
  //std::cerr << "finished: close input file: lfn = " << lfn << std::endl;
}

void
ITTService::preModuleBeginStream(StreamContext const& sc, ModuleCallingContext const& mcc) {
  ModuleDescription const& desc = * mcc.moduleDescription();
  //std::cerr << "starting: begin stream for module: stream = " << sc.streamID() << " label = '" << desc.moduleLabel() << "' id = " << desc.id() << std::endl;
}

void
ITTService::postModuleBeginStream(StreamContext const& sc, ModuleCallingContext const& mcc) {
  ModuleDescription const& desc = * mcc.moduleDescription();
  //std::cerr << "finished: begin stream for module: stream = " << sc.streamID() << " label = '" << desc.moduleLabel() << "' id = " << desc.id() << std::endl;
}

void
ITTService::preModuleEndStream(StreamContext const& sc, ModuleCallingContext const& mcc) {
  ModuleDescription const& desc = * mcc.moduleDescription();
  //std::cerr << "starting: end stream for module: stream = " << sc.streamID() << " label = '" << desc.moduleLabel() << "' id = " << desc.id() << std::endl;
}

void
ITTService::postModuleEndStream(StreamContext const& sc, ModuleCallingContext const& mcc) {
  ModuleDescription const& desc = * mcc.moduleDescription();
  //std::cerr << "finished: end stream for module: stream = " << sc.streamID() << " label = '" << desc.moduleLabel() << "' id = " << desc.id() << std::endl;
}

void
ITTService::preGlobalBeginRun(GlobalContext const& gc) {
  itt_identifier & region = runs_[gc.runIndex()];
  region = itt_identifier(globalDomain_, (boost::format("Run %d") % gc.luminosityBlockID().run()).str());
  __itt_region_begin(globalDomain_, region.id, __itt_null, region.label);
}

void
ITTService::postGlobalBeginRun(GlobalContext const& gc) {
  //std::cerr << "finished: global begin run " << gc.luminosityBlockID().run() << std::endl;
}

void
ITTService::preGlobalEndRun(GlobalContext const& gc) {
  //std::cerr << "starting: global end run " << gc.luminosityBlockID().run() << std::endl;
}

void
ITTService::postGlobalEndRun(GlobalContext const& gc) {
  itt_identifier & region = runs_[gc.runIndex()];
  __itt_region_end(globalDomain_, region.id);
  region.destroy();
}

void
ITTService::preStreamBeginRun(StreamContext const& sc) {
  StreamID sid = sc.streamID();
  itt_identifier const& region = runs_[sc.runIndex()];
  __itt_region_begin(streamDomains_[sid], region.id, __itt_null, region.label);
}

void
ITTService::postStreamBeginRun(StreamContext const& sc) {
  //std::cerr << "finished: begin run: stream = " << sc.streamID() << " run = " << sc.eventID().run() << std::endl;
}

void
ITTService::preStreamEndRun(StreamContext const& sc) {
  //std::cerr << "starting: end run: stream = " << sc.streamID() << " run = " << sc.eventID().run() << std::endl;
}

void
ITTService::postStreamEndRun(StreamContext const& sc) {
  StreamID sid = sc.streamID();
  itt_identifier const& region = runs_[sc.runIndex()];
  __itt_region_end(streamDomains_[sid], region.id);
}

void
ITTService::preGlobalBeginLumi(GlobalContext const& gc) {
  itt_identifier const& parent = runs_[gc.runIndex()];
  itt_identifier &      region = lumis_[gc.luminosityBlockIndex()];
  region = itt_identifier(globalDomain_, (boost::format("Lumi %d") % gc.luminosityBlockID().luminosityBlock()).str());
  __itt_region_begin(globalDomain_, region.id, parent.id, region.label);
}

void
ITTService::postGlobalBeginLumi(GlobalContext const& gc) {
  //std::cerr << "finished: global begin lumi: run = " << gc.luminosityBlockID().run()
  //    << " lumi = " << gc.luminosityBlockID().luminosityBlock() << std::endl;
}

void
ITTService::preGlobalEndLumi(GlobalContext const& gc) {
  //std::cerr << "starting: global end lumi: run = " << gc.luminosityBlockID().run()
  //    << " lumi = " << gc.luminosityBlockID().luminosityBlock() << std::endl;
}

void
ITTService::postGlobalEndLumi(GlobalContext const& gc) {
  itt_identifier & region = lumis_[gc.luminosityBlockIndex()];
  __itt_region_end(globalDomain_, region.id);
  region.destroy();
}

void
ITTService::preStreamBeginLumi(StreamContext const& sc) {
  StreamID sid = sc.streamID();
  itt_identifier const& parent = runs_[sc.runIndex()];
  itt_identifier const& region = lumis_[sc.luminosityBlockIndex()];
  __itt_region_begin(streamDomains_[sid], region.id, parent.id, region.label);
}

void
ITTService::postStreamBeginLumi(StreamContext const& sc) {
  //std::cerr << "finished: begin lumi: stream = " << sc.streamID() << " run = " << sc.eventID().run()
  //    << " lumi = " << sc.eventID().luminosityBlock() << std::endl;
}

void
ITTService::preStreamEndLumi(StreamContext const& sc) {
  //std::cerr << "starting: end lumi: stream = " << sc.streamID() << " run = " << sc.eventID().run()
  //    << " lumi = " << sc.eventID().luminosityBlock() << std::endl;
}

void
ITTService::postStreamEndLumi(StreamContext const& sc) {
  StreamID sid = sc.streamID();
  itt_identifier const& region = lumis_[sc.luminosityBlockIndex()];
  __itt_region_end(streamDomains_[sid], region.id);
}

void
ITTService::preEvent(StreamContext const& sc) {
  StreamID sid = sc.streamID();
  itt_identifier const& parent = lumis_[sc.luminosityBlockIndex()];
  itt_identifier &      region = events_[sid];
  region = itt_identifier(streamDomains_[sid], (boost::format("Event %d") % sc.eventID().event()).str());
  __itt_region_begin(streamDomains_[sid], region.id, parent.id, region.label);
}

void
ITTService::postEvent(StreamContext const& sc) {
  StreamID sid = sc.streamID();
  itt_identifier & region = events_[sid];
  __itt_region_end(streamDomains_[sid], region.id);
  region.destroy();
}

void
ITTService::prePathEvent(StreamContext const& sc, PathContext const& pc) {
  //std::cerr << "starting: processing path '" << pc.pathName() << "' : stream = " << sc.streamID() << std::endl;
}

void
ITTService::postPathEvent(StreamContext const& sc, PathContext const& pc, HLTPathStatus const& hlts) {
  //std::cerr << "finished: processing path '" << pc.pathName() << "' : stream = " << sc.streamID() << std::endl;
}

void
ITTService::preModuleBeginJob(ModuleDescription const& desc) {
  //std::cerr << "starting: begin job for module with label '" << desc.moduleLabel() << "' id = " << desc.id() << std::endl;
}

void
ITTService::postModuleBeginJob(ModuleDescription const& desc) {
  //std::cerr << "finished: begin job for module with label '" << desc.moduleLabel() << "' id = " << desc.id() << std::endl;
}

void
ITTService::preModuleEndJob(ModuleDescription const& desc) {
 // std::cerr << "starting: end job for module with label '" << desc.moduleLabel() << "' id = " << desc.id() << std::endl;
}

void
ITTService::postModuleEndJob(ModuleDescription const& desc) {
  //std::cerr << "finished: end job for module with label '" << desc.moduleLabel() << "' id = " << desc.id() << std::endl;
}

void
ITTService::preModuleEventPrefetching(StreamContext const& sc, ModuleCallingContext const& mcc) {
  //std::cerr << "starting: prefetching before processing event for module: stream = " << sc.streamID() << " label = '" << mcc.moduleDescription()->moduleLabel() << "' id = " << mcc.moduleDescription()->id() << std::endl;
}

void
ITTService::postModuleEventPrefetching(StreamContext const& sc, ModuleCallingContext const& mcc) {
  //std::cerr << "finished: prefetching before processing event for module: stream = " << sc.streamID() << " label = '" << mcc.moduleDescription()->moduleLabel() << "' id = " << mcc.moduleDescription()->id() << std::endl;
}

void
ITTService::preModuleEvent(StreamContext const& sc, ModuleCallingContext const& mcc) {
  StreamID sid = sc.streamID();
  itt_identifier const& parent = events_[sid];
  itt_identifier const& task   = tasks_[mcc.moduleDescription()->id()];
  __itt_task_begin_overlapped(streamDomains_[sid], task.id, parent.id, task.label);
}

void
ITTService::postModuleEvent(StreamContext const& sc, ModuleCallingContext const& mcc) {
  StreamID sid = sc.streamID();
  itt_identifier const& task   = tasks_[mcc.moduleDescription()->id()];
  __itt_task_end_overlapped(streamDomains_[sid], task.id);
}

void
ITTService::preModuleEventDelayedGet(StreamContext const& sc, ModuleCallingContext const& mcc) {
  //std::cerr << "starting: delayed processing event for module: stream = " << sc.streamID() << " label = '" << mcc.moduleDescription()->moduleLabel() << "' id = " << mcc.moduleDescription()->id() << std::endl;
}

void
ITTService::postModuleEventDelayedGet(StreamContext const& sc, ModuleCallingContext const& mcc) {
  //std::cerr << "finished: delayed processing event for module: stream = " << sc.streamID() << " label = '" << mcc.moduleDescription()->moduleLabel() << "' id = " << mcc.moduleDescription()->id() << std::endl;
}

void
ITTService::preEventReadFromSource(StreamContext const& sc, ModuleCallingContext const& mcc) {
  //std::cerr << "starting: event delayed read from source: stream = " << sc.streamID() << " label = '" << mcc.moduleDescription()->moduleLabel() << "' id = " << mcc.moduleDescription()->id() << std::endl;
}

void
ITTService::postEventReadFromSource(StreamContext const& sc, ModuleCallingContext const& mcc) {
  //std::cerr << "finished: event delayed read from source: stream = " << sc.streamID() << " label = '" << mcc.moduleDescription()->moduleLabel() << "' id = " << mcc.moduleDescription()->id() << std::endl;
}

void
ITTService::preModuleStreamBeginRun(StreamContext const& sc, ModuleCallingContext const& mcc) {
  //std::cerr << "starting: begin run for module: stream = " << sc.streamID() <<  " label = '" << mcc.moduleDescription()->moduleLabel() << "' id = " << mcc.moduleDescription()->id() << std::endl;
}

void
ITTService::postModuleStreamBeginRun(StreamContext const& sc, ModuleCallingContext const& mcc) {
  //std::cerr << "finished: begin run for module: stream = " << sc.streamID() <<  " label = '" << mcc.moduleDescription()->moduleLabel() << "' id = " << mcc.moduleDescription()->id() << std::endl;
}

void
ITTService::preModuleStreamEndRun(StreamContext const& sc, ModuleCallingContext const& mcc) {
  //std::cerr << "starting: end run for module: stream = " << sc.streamID() << " label = '" << mcc.moduleDescription()->moduleLabel() << "' id = " << mcc.moduleDescription()->id() << std::endl;
}

void
ITTService::postModuleStreamEndRun(StreamContext const& sc, ModuleCallingContext const& mcc) {
  //std::cerr << "finished: end run for module: stream = " << sc.streamID() << " label = '" << mcc.moduleDescription()->moduleLabel() << "' id = " << mcc.moduleDescription()->id() << std::endl;
}

void
ITTService::preModuleStreamBeginLumi(StreamContext const& sc, ModuleCallingContext const& mcc) {
  //std::cerr << "starting: begin lumi for module: stream = " << sc.streamID() << " label = '" << mcc.moduleDescription()->moduleLabel() << "' id = " << mcc.moduleDescription()->id() << std::endl;
}

void
ITTService::postModuleStreamBeginLumi(StreamContext const& sc, ModuleCallingContext const& mcc) {
  //std::cerr << "finished: begin lumi for module: stream = " << sc.streamID() << " label = '" << mcc.moduleDescription()->moduleLabel() << "' id = " << mcc.moduleDescription()->id() << std::endl;
}

void
ITTService::preModuleStreamEndLumi(StreamContext const& sc, ModuleCallingContext const& mcc) {
  //std::cerr << "starting: end lumi for module: stream = " << sc.streamID() << " label = '"<< mcc.moduleDescription()->moduleLabel() << "' id = " << mcc.moduleDescription()->id() << std::endl;
}

void
ITTService::postModuleStreamEndLumi(StreamContext const& sc, ModuleCallingContext const& mcc) {
  //std::cerr << "finished: end lumi for module: stream = " << sc.streamID() << " label = '"<< mcc.moduleDescription()->moduleLabel() << "' id = " << mcc.moduleDescription()->id() << std::endl;
}

void
ITTService::preModuleGlobalBeginRun(GlobalContext const& gc, ModuleCallingContext const& mcc) {
  //std::cerr << "starting: global begin run for module: label = '" << mcc.moduleDescription()->moduleLabel() << "' id = " << mcc.moduleDescription()->id() << std::endl;
}

void
ITTService::postModuleGlobalBeginRun(GlobalContext const& gc, ModuleCallingContext const& mcc) {
  //std::cerr << "finished: global begin run for module: label = '" << mcc.moduleDescription()->moduleLabel() << "' id = " << mcc.moduleDescription()->id() << std::endl;
}

void
ITTService::preModuleGlobalEndRun(GlobalContext const& gc, ModuleCallingContext const& mcc) {
  //std::cerr << "starting: global end run for module: label = '" << mcc.moduleDescription()->moduleLabel() << "' id = " << mcc.moduleDescription()->id() << std::endl;
}

void
ITTService::postModuleGlobalEndRun(GlobalContext const& gc, ModuleCallingContext const& mcc) {
  //std::cerr << "finished: global end run for module: label = '" << mcc.moduleDescription()->moduleLabel() << "' id = " << mcc.moduleDescription()->id() << std::endl;
}

void
ITTService::preModuleGlobalBeginLumi(GlobalContext const& gc, ModuleCallingContext const& mcc) {
  //std::cerr << "starting: global begin lumi for module: label = '" << mcc.moduleDescription()->moduleLabel() << "' id = " << mcc.moduleDescription()->id() << std::endl;
}

void
ITTService::postModuleGlobalBeginLumi(GlobalContext const& gc, ModuleCallingContext const& mcc) {
  //std::cerr << "finished: global begin lumi for module: label = '" << mcc.moduleDescription()->moduleLabel() << "' id = " << mcc.moduleDescription()->id() << std::endl;
}

void
ITTService::preModuleGlobalEndLumi(GlobalContext const& gc, ModuleCallingContext const& mcc) {
  //std::cerr << "starting: global end lumi for module: label = '" << mcc.moduleDescription()->moduleLabel() << "' id = " << mcc.moduleDescription()->id() << std::endl;
}

void
ITTService::postModuleGlobalEndLumi(GlobalContext const& gc, ModuleCallingContext const& mcc) {
  //std::cerr << "finished: global end lumi for module: label = '" << mcc.moduleDescription()->moduleLabel() << "' id = " << mcc.moduleDescription()->id() << std::endl;
}

using edm::service::ITTService;
DEFINE_FWK_SERVICE(ITTService);
