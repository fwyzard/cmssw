#include <vector>
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/TCDS/interface/TCDSEventRecord.h"

namespace {
  struct dictionary {
    tcds::TCDSEventRecord               tcds_;
    edm::Wrapper<tcds::TCDSEventRecord> w_tcds_;
  };
}
