#ifndef DataFormats_PortableTestObjects_interface_TestSoA_h
#define DataFormats_PortableTestObjects_interface_TestSoA_h

#include <Eigen/Core>
#include <Eigen/Dense>

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoAView.h"

namespace portabletest {

  using M36d = Eigen::Matrix<double, 3, 6>;
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
                      // Typedef is needed as comas confuse macros
                      SOA_EIGEN_COLUMN(M36d, m))

  using TestSoA = TestSoALayout<>;

}  // namespace portabletest

#endif  // DataFormats_PortableTestObjects_interface_TestSoA_h
