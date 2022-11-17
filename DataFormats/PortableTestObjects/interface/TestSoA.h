#ifndef DataFormats_PortableTestObjects_interface_TestSoA_h
#define DataFormats_PortableTestObjects_interface_TestSoA_h

#include <Eigen/Core>
#include <Eigen/Dense>

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoAView.h"

namespace portabletest {

  using Matrix = Eigen::Matrix<double, 3, 6>;
  enum class Quality : uint8_t { bad = 0, edup, dup, loose, strict, tight, highPurity, notQuality };
  // SoA layout with x, y, z, id fields
  GENERATE_SOA_LAYOUT(TestSoALayout,
                      // columns: one value per element
                      SOA_COLUMN(double, x),
                      SOA_COLUMN(double, y),
                      SOA_COLUMN(double, z),
                      SOA_COLUMN(int32_t, id),
                      // scalars: one value for the whole structure
                      SOA_SCALAR(double, r),
                      // Eigen columns
                      // the typedef is needed because commas confuse macros
                      SOA_EIGEN_COLUMN(Matrix, m),
                      // Auxiliary array of doubles of size 5 * n + 3
                      SOA_AUX_COLUMN(SOA_AUX_TYPE(double, 5, 3), aux),
                      SOA_COLUMN(Quality, quality))

  using TestSoA = TestSoALayout<>;

  inline std::ostream& operator<<(std::ostream& os, Quality const& q) {
    using Q = Quality;
    switch (q) {
      case Q::bad:
        os << "bad";
        break;
      case Q::edup:
        os << "edup";
        break;
      case Q::dup:
        os << "dup";
        break;
      case Q::loose:
        os << "loose";
        break;
      case Q::strict:
        os << "strict";
        break;
      case Q::tight:
        os << "tight";
        break;
      case Q::highPurity:
        os << "highPurity";
        break;
      case Q::notQuality:
        os << "notQuality";
        break;
      default:
        os << "UnknownQuality(" << static_cast<uint8_t>(q) << ")";
        break;
    }
    return os;
  }

}  // namespace portabletest
#endif  // DataFormats_PortableTestObjects_interface_TestSoA_h
