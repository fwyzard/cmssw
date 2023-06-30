#ifndef CondFormats_HcalObjects_interface_HcalRecoParamWithPulseShapeHost_h
#define CondFormats_HcalObjects_interface_HcalRecoParamWithPulseShapeHost_h

#include "CondFormats/HcalObjects/interface/HcalRecoParamWithPulseShapeSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "DataFormats/Portable/interface/PortableCollection.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"

template <typename TDev>
class HcalRecoParamWithPulseShapeT {
public:
  using RecoParamCollection = PortableCollection<HcalRecoParamSoA, TDev>;
  using PulseShapeCollection = PortableCollection<HcalPulseShapeSoA, TDev>;

  class ConstView {
  public:
    constexpr ConstView(typename RecoParamCollection::ConstView recoView,
                        typename PulseShapeCollection::ConstView psView)
        : recoParamView_{recoView}, pulseShapeView_{psView} {};
    constexpr float pulseShape(int id) const { return 0; }

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

  ALPAKA_FN_HOST_ACC typename RecoParamCollection::View recoParamView() { return recoParam_.view(); }
  ALPAKA_FN_HOST_ACC typename PulseShapeCollection::View pulseShapeView() { return pulseShape_.view(); }

  ConstView const_view() const { return ConstView(recoParam_.view(), pulseShape_.view()); }

private:
  RecoParamCollection recoParam_;
  PulseShapeCollection pulseShape_;
};

using HcalRecoParamWithPulseShapeHost = HcalRecoParamWithPulseShapeT<alpaka::DevCpu>;

#endif
