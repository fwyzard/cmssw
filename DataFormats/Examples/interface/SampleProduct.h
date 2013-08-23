#ifndef SampleProduct_h
#define SampleProduct_h

#include <string>
#include "DataFormats/BTauReco/interface/RefMacros.h"

namespace example {

class SampleProduct
{
public:
  SampleProduct() noexcept :
    m_data()
  { }

  SampleProduct(std::string const & data) noexcept :
    m_data(data)
  { }

  SampleProduct(std::string && data) noexcept :
    m_data(data)
  { }

  std::string const & data() const
  {
    return m_data;
  }

private:
  std::string m_data;

};

DECLARE_EDM_REFS(SampleProduct)

} // namespace

#endif // SampleProduct_h
