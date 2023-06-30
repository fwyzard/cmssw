#ifndef DataFormats_HcalDigi_HcalDigiSoA_h
#define DataFormats_HcalDigi_HcalDigiSoA_h

//TODO: Use Eigen column for data(?)
//#include <Eigen/Core>
//#include <Eigen/Dense>
#include <array>

#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"

namespace hcal {

  using QIE11dataArray = std::array<uint16_t, QIE11DigiCollection::MAXSAMPLES>;
  using QIE10dataArray = std::array<uint16_t, HBHEDataFrame::MAXSAMPLES>;

  //using QIE11dataVector = Eigen::Matrix<uint16_t,  QIE11DigiCollection::MAXSAMPLES, 1>;
  //using QIE10dataVector = Eigen::Matrix<uint16_t,  HBHEDataFrame::MAXSAMPLES, 1>;

  GENERATE_SOA_LAYOUT(HcalPhase1DigiSoALayout,
                      SOA_COLUMN(uint32_t, ids),
                      //SOA_EIGEN_COLUMN(QIE11dataVector, data),
                      SOA_COLUMN(QIE11dataArray, data),
                      SOA_SCALAR(uint32_t, stride),
                      SOA_SCALAR(uint32_t, size))
  GENERATE_SOA_LAYOUT(HcalPhase0DigiSoALayout,
                      SOA_COLUMN(uint32_t, ids),
                      SOA_COLUMN(uint32_t, npresamples),
                      //SOA_EIGEN_COLUMN(QIE10dataVector, data),
                      SOA_COLUMN(QIE10dataArray, data),
                      SOA_SCALAR(uint32_t, stride),
                      SOA_SCALAR(uint32_t, size))

  using HcalPhase1DigiSoA = HcalPhase1DigiSoALayout<>;
  using HcalPhase0DigiSoA = HcalPhase0DigiSoALayout<>;

}  // namespace hcal

#endif
