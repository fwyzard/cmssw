#ifndef DataFormats_TCDS_interface_TCDSEventRecord_h
#define DataFormats_TCDS_interface_TCDSEventRecord_h

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

// This structure is not a bitwise description of the L1A History entries,
// because it avoids the "reserved" or otherwise empty fields, and
// rounds up the size where necessary (e.g. 48 bits --> 64 bits).

struct L1AHistoryEntry {
    uint8_t  index;
    uint8_t  event_type;
    uint16_t bx;
    uint64_t orbit;
};


// This structure is not a bitwise description of the TCDS data block,
// because it avoids the "reserved" or otherwise empty fields, and
// rounds up the size where necessary (e.g. 48 bits --> 64 bits).

struct TCDSEventRecord {
    uint64_t                        slink_header;       // word 0
    uint8_t                         size_header;        // word 1
    uint8_t                         size_summary;
    uint8_t                         size_l1_history;
    uint8_t                         size_bst;
    uint8_t                         size_bgo_history;
    uint64_t                        mac_address;        // word 2
    uint32_t                        version_software;   // word 3
    uint32_t                        version_firmware;
    uint32_t                        version_record;     // word 4
    uint32_t                        run_number;         // word 5
    uint32_t                        bst_status;         // words 6..7
    std::bitset<96>                 active_partitions;
    uint32_t                        lumi_nibble;        // word 8
    uint32_t                        lumi_section;
    uint16_t                        nibble_per_ls;      // word 9
    uint16_t                        trigger_word;
    uint8_t                         in0;
    uint8_t                         in1;
    uint16_t                        bcid;               // word 10
    uint64_t                        orbit;
    uint64_t                        event_number;       // word 11
    uint64_t                        total_event_number; // word 12
    std::array<L1AHistoryEntry, 16> l1a_history;        // words 13..44
    BSTMessage                      bst_message;        // words 45..52
    uint32_t                        bgo_history_header; // word 53
    std::array<uint64_t, 64>        bgo_history;        // words 54..117
    uint64_t                        slink_trailer;      // word 118
};

} // namespace tcds

#endif // not defined DataFormats_TCDS_interface_TCDSEventRecord_h
