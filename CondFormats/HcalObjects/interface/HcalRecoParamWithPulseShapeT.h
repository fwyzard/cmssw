#ifndef CondFormats_HcalObjects_interface_HcalRecoParamWithPulseShapeHostT_h
#define CondFormats_HcalObjects_interface_HcalRecoParamWithPulseShapeHostT_h

#include "CondFormats/HcalObjects/interface/HcalRecoParamWithPulseShapeSoA.h"
#include "DataFormats/Portable/interface/PortableCollection.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"

template <typename TDev>
class HcalRecoParamWithPulseShapeT {
public:
  using RecoParamCollection = PortableCollection<HcalRecoParamSoA, TDev>;
  using PulseShapeCollection = PortableCollection<HcalPulseShapeSoA, TDev>;

  class ConstView {
  public:
    using RecoParamConstView = typename RecoParamCollection::ConstView;
    using PulseShapeConstView = typename PulseShapeCollection::ConstView;
    constexpr ConstView(RecoParamConstView recoView, PulseShapeConstView psView)
        : recoParamView_{recoView}, pulseShapeView_{psView} {};

    constexpr float compute_pulse_shape_value(uint32_t const hashedId,
                                              float const pulse_time,
                                              int const sample,
                                              int const shift) {

        auto const recoPulseShapeId = this->recoParamView().ids()[hashedId];
        auto const& acc25nsVec = this->pulseShapeView().acc25nsVec()[recoPulseShapeId];
        auto const& diff25nsItvlVec = this->pulseShapeView().diff25nsItvlVec()[recoPulseShapeId];
        auto const& accVarLenIdxMinusOneVec =
            this->pulseShapeView().accVarLenIdxMinusOneVec()[recoPulseShapeId];
        auto const& diffVarItvlIdxMinusOneVec =
            this->pulseShapeView().diffVarItvlIdxMinusOneVec()[recoPulseShapeId];
        auto const& accVarLenIdxZeroVec = this->pulseShapeView().accVarLenIdxZEROVec()[recoPulseShapeId];
        auto const& diffVarItvlIdxZeroVec = this->pulseShapeView().diffVarItvlIdxZEROVec()[recoPulseShapeId];

        // constants
        constexpr float slew = 0.f;
        constexpr auto ns_per_bx = ::hcal::constants::nsPerBX;

        // FIXME: clean up all the rounding... this is coming from original cpu version
        float const i_start_float = -::hcal::constants::iniTimeShift - pulse_time - slew > 0.f
                                        ? 0.f
                                        : std::abs(-::hcal::constants::iniTimeShift - pulse_time - slew) + 1.f;
        int i_start = static_cast<int>(i_start_float);
        float offset_start = static_cast<float>(i_start) - ::hcal::constants::iniTimeShift - pulse_time - slew;

        // boundary
        if (offset_start == 1.0f) {
          offset_start = 0.f;
          i_start -= 1;
        }

        int const bin_start = static_cast<int>(offset_start);
        auto const bin_start_up = static_cast<float>(bin_start) + 0.5f;
        int const bin_0_start = offset_start < bin_start_up ? bin_start - 1 : bin_start;
        int const its_start = i_start / ns_per_bx;
        int const distTo25ns_start = ::hcal::constants::nsPerBX - 1 - i_start % ns_per_bx;
        auto const factor = offset_start - static_cast<float>(bin_0_start) - 0.5;

        auto const sample_over10ts = sample + shift;
        float value = 0.0f;
        if (sample_over10ts == its_start) {
          value = bin_0_start == -1
                      ? accVarLenIdxMinusOneVec[distTo25ns_start] + factor * diffVarItvlIdxMinusOneVec[distTo25ns_start]
                      : accVarLenIdxZeroVec[distTo25ns_start] + factor * diffVarItvlIdxZeroVec[distTo25ns_start];
        } else if (sample_over10ts > its_start) {
          int const bin_idx = distTo25ns_start + 1 + (sample_over10ts - its_start - 1) * ns_per_bx + bin_0_start;
          value = acc25nsVec[bin_idx] + factor * diff25nsItvlVec[bin_idx];
        }
        return value;
      }

    constexpr typename RecoParamCollection::ConstView recoParamView() { return recoParamView_; }
    constexpr typename PulseShapeCollection::ConstView pulseShapeView() { return pulseShapeView_; }

  private:
    typename RecoParamCollection::ConstView recoParamView_;
    typename PulseShapeCollection::ConstView pulseShapeView_;
  };

  HcalRecoParamWithPulseShapeT(size_t recoSize, size_t pulseSize, TDev const& dev)
      : recoParam_(recoSize, dev), pulseShape_(pulseSize, dev) {}
  template <typename TQueue, typename = std::enable_if_t<alpaka::isQueue<TQueue>>>
  HcalRecoParamWithPulseShapeT(size_t recoSize, size_t pulseSize, TQueue const& queue)
      : recoParam_(recoSize, queue), pulseShape_(pulseSize, queue) {}
  HcalRecoParamWithPulseShapeT(RecoParamCollection reco, PulseShapeCollection pulse)
      : recoParam_(std::move(reco)), pulseShape_(std::move(pulse)) {}

  const RecoParamCollection& recoParam() const { return recoParam_; }
  const PulseShapeCollection& pulseShape() const { return pulseShape_; }

  typename RecoParamCollection::View recoParamView() { return recoParam_.view(); }
  typename PulseShapeCollection::View pulseShapeView() { return pulseShape_.view(); }

  ConstView const_view() const { return ConstView(recoParam_.view(), pulseShape_.view()); }

private:
  RecoParamCollection recoParam_;
  PulseShapeCollection pulseShape_;
};
#endif
