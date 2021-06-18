#include "DataFormats/Phase2HLTPFTaus/interface/PFTauPair.h"

#include "DataFormats/Math/interface/deltaR.h"  // reco::deltaR

namespace reco {

  PFTauPair::PFTauPair(const reco::PFTauRef& leadPFTau,
                       double leadPFTauSumChargedIso,
                       const reco::PFTauRef& subleadPFTau,
                       double subleadPFTauSumChargedIso,
                       double dz)
      : LeafCandidate(leadPFTau->charge() + subleadPFTau->charge(),
                      leadPFTau->p4() + subleadPFTau->p4(),
                      reco::LeafCandidate::Point(0.5 * (leadPFTau->vertex().x() + subleadPFTau->vertex().x()),
                                                 0.5 * (leadPFTau->vertex().y() + subleadPFTau->vertex().y()),
                                                 0.5 * (leadPFTau->vertex().z() + subleadPFTau->vertex().z()))),
        leadPFTau_(leadPFTau),
        leadPFTauSumChargedIso_(leadPFTauSumChargedIso),
        subleadPFTau_(subleadPFTau),
        subleadPFTauSumChargedIso_(subleadPFTauSumChargedIso),
        dz_(dz) {
    assert(leadPFTau_->pt() >= subleadPFTau_->pt());
    assert(reco::deltaR(leadPFTau_->p4(), subleadPFTau_->p4()) > 1.e-3);
  }

  std::ostream& operator<<(std::ostream& out, const PFTauPair& pfTauPair) {
    //out << "PFTauPair:"
    //    << " pT = " << pfTauPair.pt() << ","
    out << "pT = " << pfTauPair.pt() << ","
        << " eta = " << pfTauPair.eta() << ","
        << " phi = " << pfTauPair.phi() << ","
        << " charge = " << pfTauPair.charge() << ","
        << " mass = " << pfTauPair.mass() << std::endl;
    out << " vertex: x = " << pfTauPair.vertex().x() << ", y = " << pfTauPair.vertex().y()
        << ", z = " << pfTauPair.vertex().z() << std::endl;

    const reco::PFTauRef& leadPFTau = pfTauPair.leadPFTau();
    out << "leadingPFTau:"
        << " pT = " << leadPFTau->pt() << ","
        << " eta = " << leadPFTau->eta() << ","
        << " phi = " << leadPFTau->phi() << ","
        << " decayMode = " << leadPFTau->decayMode() << ","
        << " mass = " << leadPFTau->mass() << std::endl;
    out << " vertex: x = " << leadPFTau->vertex().x() << ", y = " << leadPFTau->vertex().y()
        << ", z = " << leadPFTau->vertex().z() << std::endl;
    const reco::PFCandidatePtr leadPFTauLeadPFChargedHadrCand = leadPFTau->leadPFChargedHadrCand();
    out << " leadChargedHadrCand:";
    if (leadPFTauLeadPFChargedHadrCand.isNonnull() && leadPFTauLeadPFChargedHadrCand.isAvailable()) {
      out << " pT = " << leadPFTauLeadPFChargedHadrCand->pt() << ","
          << " eta = " << leadPFTauLeadPFChargedHadrCand->eta() << ","
          << " phi = " << leadPFTauLeadPFChargedHadrCand->phi() << ","
          << " pdgId = " << leadPFTauLeadPFChargedHadrCand->pdgId() << std::endl;
      out << "  vertex:";
      const reco::Track* leadPFTauLeadTrack = leadPFTauLeadPFChargedHadrCand->bestTrack();
      if (leadPFTauLeadTrack) {
        out << " x = " << leadPFTauLeadTrack->vertex().x() << ", y = " << leadPFTauLeadTrack->vertex().y()
            << ", z = " << leadPFTauLeadTrack->vertex().z() << " (dz = " << leadPFTauLeadTrack->dz(pfTauPair.vertex())
            << ")" << std::endl;
      } else {
        out << " N/A" << std::endl;
      }
    } else {
      out << " N/A" << std::endl;
    }
    out << " sumChargedIso = " << pfTauPair.leadPFTauSumChargedIso() << std::endl;

    const reco::PFTauRef& subleadPFTau = pfTauPair.subleadPFTau();
    out << "subleadingPFTau:"
        << " pT = " << subleadPFTau->pt() << ","
        << " eta = " << subleadPFTau->eta() << ","
        << " phi = " << subleadPFTau->phi() << ","
        << " decayMode = " << subleadPFTau->decayMode() << ","
        << " mass = " << subleadPFTau->mass() << std::endl;
    out << " vertex: x = " << subleadPFTau->vertex().x() << ", y = " << subleadPFTau->vertex().y()
        << ", z = " << subleadPFTau->vertex().z() << std::endl;
    const reco::PFCandidatePtr subleadPFTauLeadPFChargedHadrCand = subleadPFTau->leadPFChargedHadrCand();
    out << " leadChargedHadrCand:";
    if (subleadPFTauLeadPFChargedHadrCand.isNonnull() && subleadPFTauLeadPFChargedHadrCand.isAvailable()) {
      out << " pT = " << subleadPFTauLeadPFChargedHadrCand->pt() << ","
          << " eta = " << subleadPFTauLeadPFChargedHadrCand->eta() << ","
          << " phi = " << subleadPFTauLeadPFChargedHadrCand->phi() << ","
          << " pdgId = " << subleadPFTauLeadPFChargedHadrCand->pdgId() << std::endl;
      out << "  vertex:";
      const reco::Track* subleadPFTauLeadTrack = subleadPFTauLeadPFChargedHadrCand->bestTrack();
      if (subleadPFTauLeadTrack) {
        out << " x = " << subleadPFTauLeadTrack->vertex().x() << ", y = " << subleadPFTauLeadTrack->vertex().y()
            << ", z = " << subleadPFTauLeadTrack->vertex().z()
            << " (dz = " << subleadPFTauLeadTrack->dz(pfTauPair.vertex()) << ")" << std::endl;
      } else {
        out << " N/A" << std::endl;
      }
    } else {
      out << " N/A" << std::endl;
    }
    out << " sumChargedIso = " << pfTauPair.subleadPFTauSumChargedIso() << std::endl;

    out << "dz = " << pfTauPair.dz() << std::endl;

    return out;
  }

}  // namespace reco
