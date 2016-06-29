#ifndef DataFormats_TCDS_interface_BSTMessage_h
#define DataFormats_TCDS_interface_BSTMessage_h

#include <array>
#include <bitset>

#include "DataFormats/TCDS/interface/BSTMessage.h"

// For the description of the TCDS Event Record, see
// https://twiki.cern.ch/twiki/bin/viewauth/CMS/TcdsEventRecord .
//
// As of 2016.06.01, this implementation is up to date with the
// TCDS Data Block Definition implementation dated 2015.03.09 .
//
// The S-LINK header and footer are describet at
// http://cmsdoc.cern.ch/cms/TRIDAS/horizontal/RUWG/DAQ_IF_guide/DAQ_IF_guide.html .
//
// The LHC BST message is described at
// https://edms.cern.ch/ui/file/638899/2.0/LHC-BOB-ES-0001-20-00.pdf .

namespace tcds {

enum class ParticleType : uint8_t {
  PROTON        =  1,
  LEAD          =  2
};

enum class BeamMode : uint8_t {
  NOMODE        =  1,                   // No mode, data is not available, not set
  SETUP         =  2,                   // Setup
  INJPILOT      =  3,                   // Pilot injection
  INJINTER      =  4,                   // Intermediate injection
  INJNOMIN      =  5,                   // Nominal injection
  PRERAMP       =  6,                   // Before ramp
  RAMP          =  7,                   // Ramp
  FLATTOP       =  8,                   // Flat top
  SQUEEZE       =  9,                   // Squeeze
  ADJUST        = 10,                   // Adjust beam on flat top
  STABLE        = 11,                   // Stable beam for physics
  UNSTABLE      = 12,                   // Unstable beam
  BEAMDUMP      = 13,                   // Beam dump
  RAMPDOWN      = 14,                   // Ramp down
  RECOVERY      = 15,                   // Recovering
  INJDUMP       = 16,                   // Inject and dump
  CIRCDUMP      = 17,                   // Circulate and dump
  ABORT         = 18,                   // Recovery after a beam permit flag drop
  CYCLING       = 19,                   // Pre-cycle before injection, no beam
  WBDUMP        = 20,                   // Warning beam dump
  NOBEAM        = 21                    // No beam or preparation for beam
};

// This structure is not a bitwise description of the BST message,
// because it avoids the "reserved" or otherwise empty fields, and
// rounds up the size where necessary (e.g. 48 bits --> 64 bits).

struct BSTMessage {
  timeval       gps_time;               // seconds since epoch and microseconds
  char          reserved_part1[9];      // reserved for BI, part 1 (bytes 8-16)
  uint8_t       beam;                   // beam 1 or 2
  uint32_t      turn;                   // turn count number (11.2 kHz)
  uint32_t      fill;                   // fill number
  BeamMode      beam_mode;              // STABLE, etc
  ParticleType  beam1_type;             // proton or lead
  ParticleType  beam2_type;             // proton or lead
  uint16_t      beam_momentum;          // GeV/c
  uint32_t      beam1_intensity;        // in unit of 10e10 charges
  uint32_t      beam2_intensity;        // in unit of 10e10 charges
  char          reserved_part2[24];     // reserved for BI, part 2 (bytes 40-63)
};

} // namespace tcds

#endif // not defined DataFormats_TCDS_interface_BSTMessage_h
