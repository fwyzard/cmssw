#include <cassert>

#include "DataFormats/PortableTestObjects/interface/TestHostCollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"

namespace {

  template <typename T>
  class Column {
  public:
    Column(T const* data, size_t size) : data_(data), size_(size) {}

    void print(std::ostream& out) const {
      std::stringstream buffer;
      buffer << "{ ";
      if (size_ > 0) {
        buffer << data_[0];
      }
      if (size_ > 1) {
        buffer << ", " << data_[1];
      }
      if (size_ > 2) {
        buffer << ", " << data_[2];
      }
      if (size_ > 3) {
        buffer << ", ...";
      }
      buffer << '}';
      out << buffer.str();
    }

  private:
    T const* const data_;
    size_t const size_;
  };

  template <typename T>
  std::ostream& operator<<(std::ostream& out, Column<T> const& column) {
    column.print(out);
    return out;
  }

  template <typename T>
  void checkViewAddresses(T const& view) {
    assert(view.metadata().addressOf_x() == view.x());
    assert(view.metadata().addressOf_x() == &view.x(0));
    assert(view.metadata().addressOf_x() == &view[0].x());
    assert(view.metadata().addressOf_y() == view.y());
    assert(view.metadata().addressOf_y() == &view.y(0));
    assert(view.metadata().addressOf_y() == &view[0].y());
    assert(view.metadata().addressOf_z() == view.z());
    assert(view.metadata().addressOf_z() == &view.z(0));
    assert(view.metadata().addressOf_z() == &view[0].z());
    assert(view.metadata().addressOf_id() == view.id());
    assert(view.metadata().addressOf_id() == &view.id(0));
    assert(view.metadata().addressOf_id() == &view[0].id());
    assert(view.metadata().addressOf_m() == view.m());
    assert(view.metadata().addressOf_m() == &view.m(0).coeffRef(0, 0));
    assert(view.metadata().addressOf_m() == &view[0].m().coeffRef(0, 0));
    assert(view.metadata().addressOf_r() == &view.r());
    //assert(view.metadata().addressOf_r() == &view.r(0));                  // cannot access a scalar with an index
    //assert(view.metadata().addressOf_r() == &view[0].r());                // cannot access a scalar via a SoA row-like accessor
  }

  template <typename T>
  void checkViewAddresses2(T const& view) {
    assert(view.metadata().addressOf_x2() == view.x2());
    assert(view.metadata().addressOf_x2() == &view.x2(0));
    assert(view.metadata().addressOf_x2() == &view[0].x2());
    assert(view.metadata().addressOf_y2() == view.y2());
    assert(view.metadata().addressOf_y2() == &view.y2(0));
    assert(view.metadata().addressOf_y2() == &view[0].y2());
    assert(view.metadata().addressOf_z2() == view.z2());
    assert(view.metadata().addressOf_z2() == &view.z2(0));
    assert(view.metadata().addressOf_z2() == &view[0].z2());
    assert(view.metadata().addressOf_id2() == view.id2());
    assert(view.metadata().addressOf_id2() == &view.id2(0));
    assert(view.metadata().addressOf_id2() == &view[0].id2());
    assert(view.metadata().addressOf_m2() == view.m2());
    assert(view.metadata().addressOf_m2() == &view.m2(0).coeffRef(0, 0));
    assert(view.metadata().addressOf_m2() == &view[0].m2().coeffRef(0, 0));
    assert(view.metadata().addressOf_r2() == &view.r2());
    //assert(view.metadata().addressOf_r2() == &view.r2(0));                  // cannot access a scalar with an index
    //assert(view.metadata().addressOf_r2() == &view[0].r2());                // cannot access a scalar via a SoA row-like accessor
  }

}  // namespace

class TestAlpakaAnalyzer : public edm::stream::EDAnalyzer<> {
public:
  TestAlpakaAnalyzer(edm::ParameterSet const& config)
      : source_{config.getParameter<edm::InputTag>("source")},
        token_{consumes(source_)},
        tokenMulti_{consumes(source_)} {}

  void analyze(edm::Event const& event, edm::EventSetup const&) override {
    portabletest::TestHostCollection const& product = event.get(token_);
    auto const& view = product.const_view();
    auto& mview = product.view();
    auto const& cmview = product.view();

    {
      edm::LogInfo msg("TestAlpakaAnalyzer");
      msg << source_.encode() << ".size() = " << view.metadata().size() << '\n';
      msg << "  data @ " << product.buffer().data() << ",\n"
          << "  x    @ " << view.metadata().addressOf_x() << " = " << Column(view.x(), view.metadata().size()) << ",\n"
          << "  y    @ " << view.metadata().addressOf_y() << " = " << Column(view.y(), view.metadata().size()) << ",\n"
          << "  z    @ " << view.metadata().addressOf_z() << " = " << Column(view.z(), view.metadata().size()) << ",\n"
          << "  id   @ " << view.metadata().addressOf_id() << " = " << Column(view.id(), view.metadata().size())
          << ",\n"
          << "  r    @ " << view.metadata().addressOf_r() << " = " << view.r() << '\n'
          << "  m    @ " << view.metadata().addressOf_m() << " = { ... {" << view[1].m()(1, Eigen::all)
          << " } ... } \n";
      msg << std::hex << "  [y - x] = 0x"
          << reinterpret_cast<intptr_t>(view.metadata().addressOf_y()) -
                 reinterpret_cast<intptr_t>(view.metadata().addressOf_x())
          << "  [z - y] = 0x"
          << reinterpret_cast<intptr_t>(view.metadata().addressOf_z()) -
                 reinterpret_cast<intptr_t>(view.metadata().addressOf_y())
          << "  [id - z] = 0x"
          << reinterpret_cast<intptr_t>(view.metadata().addressOf_id()) -
                 reinterpret_cast<intptr_t>(view.metadata().addressOf_z())
          << "  [r - id] = 0x"
          << reinterpret_cast<intptr_t>(view.metadata().addressOf_r()) -
                 reinterpret_cast<intptr_t>(view.metadata().addressOf_id())
          << "  [m - r] = 0x"
          << reinterpret_cast<intptr_t>(view.metadata().addressOf_m()) -
                 reinterpret_cast<intptr_t>(view.metadata().addressOf_r());
    }

    checkViewAddresses(view);
    checkViewAddresses(mview);
    checkViewAddresses(cmview);

    const portabletest::Matrix matrix{{1, 2, 3, 4, 5, 6}, {2, 4, 6, 8, 10, 12}, {3, 6, 9, 12, 15, 18}};
    assert(view.r() == 1.);
    for (int32_t i = 0; i < view.metadata().size(); ++i) {
      auto vi = view[i];
      assert(vi.x() == 0.);
      assert(vi.y() == 0.);
      assert(vi.z() == 0.);
      assert(vi.id() == i);
      assert(vi.m() == matrix * i);
    }

    portabletest::TestHostMultiCollection const& productMulti = event.get(tokenMulti_);
    auto const& viewMulti0 = productMulti.const_view<0>();
    auto& mviewMulti0 = productMulti.view<0>();
    auto const& cmviewMulti0 = productMulti.view<0>();
    auto const& viewMulti1 = productMulti.const_view<1>();
    auto& mviewMulti1 = productMulti.view<1>();
    auto const& cmviewMulti1 = productMulti.view<1>();

    checkViewAddresses(viewMulti0);
    checkViewAddresses(mviewMulti0);
    checkViewAddresses(cmviewMulti0);
    checkViewAddresses2(viewMulti1);
    checkViewAddresses2(mviewMulti1);
    checkViewAddresses2(cmviewMulti1);

    assert(viewMulti0.r() == 1.);
    for (int32_t i = 0; i < viewMulti0.metadata().size(); ++i) {
      auto vi = viewMulti0[i];
      assert(vi.x() == 0.);
      assert(vi.y() == 0.);
      assert(vi.z() == 0.);
      assert(vi.id() == i);
      assert(vi.m() == matrix * i);
    }
    assert(viewMulti1.r2() == 1.);
    for (int32_t i = 0; i < viewMulti1.metadata().size(); ++i) {
      auto vi = viewMulti1[i];
      assert(vi.x2() == 0.);
      assert(vi.y2() == 0.);
      assert(vi.z2() == 0.);
      assert(vi.id2() == i);
      assert(vi.m2() == matrix * i);
    }
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("source");
    descriptions.addWithDefaultLabel(desc);
  }

private:
  const edm::InputTag source_;
  const edm::EDGetTokenT<portabletest::TestHostCollection> token_;
  const edm::EDGetTokenT<portabletest::TestHostMultiCollection> tokenMulti_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TestAlpakaAnalyzer);
