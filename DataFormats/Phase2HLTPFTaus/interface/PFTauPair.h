#ifndef DataFormats_Phase2HLTPFTaus_PFTauPair_h
#define DataFormats_Phase2HLTPFTaus_PFTauPair_h

#include "DataFormats/Candidate/interface/LeafCandidate.h"  // reco::LeafCandidate
#include "DataFormats/TauReco/interface/PFTau.h"            // reco::PFTau
#include "DataFormats/TauReco/interface/PFTauFwd.h"         // reco::PFTauRef

namespace reco {
  class PFTauPair : public reco::LeafCandidate {
  public:
    PFTauPair() = default;
    PFTauPair(const reco::PFTauRef& leadPFTau,
              double leadPFTauSumChargedIso,
              const reco::PFTauRef& subleadPFTau,
              double subleadPFTauSumChargedIso,
              double dz);
    ~PFTauPair() = default;

    const reco::PFTauRef& leadPFTau() const { return leadPFTau_; }
    double leadPFTauSumChargedIso() const { return leadPFTauSumChargedIso_; }

    const reco::PFTauRef& subleadPFTau() const { return subleadPFTau_; }
    double subleadPFTauSumChargedIso() const { return subleadPFTauSumChargedIso_; }

    double dz() const { return dz_; }

  private:
    reco::PFTauRef leadPFTau_;
    double leadPFTauSumChargedIso_;

    reco::PFTauRef subleadPFTau_;
    double subleadPFTauSumChargedIso_;

    double dz_;
  };

  std::ostream& operator<<(std::ostream& out, const PFTauPair& pfTauPair);

}  // end namespace reco

#endif
