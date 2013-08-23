#include <vector>
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Examples/interface/SampleProduct.h"

namespace {
  struct dictionary {

    // SampleProduct
    example::SampleProduct                                        sp;
    example::SampleProductCollection                              sp_c;
    example::SampleProductRef                                     sp_r;
    example::SampleProductFwdRef                                  sp_fr;
    example::SampleProductRefProd                                 sp_rp;
    example::SampleProductRefVector                               sp_rv;
    edm::Wrapper<example::SampleProductCollection>                sp_wc;

  };
}
