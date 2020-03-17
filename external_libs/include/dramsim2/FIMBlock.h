#ifndef __ALU__HPP__
#define __ALU__HPP__

#include "dramsim2/burst.h"
#include <cmath>
#include <cstring>
#include <iostream>
#include <string>

using namespace std;


namespace DRAMSim {

class fim_block_t {
  public:
    BurstType srf;
    BurstType grfa[8];
    BurstType grfb[8];
    BurstType m_out;
    BurstType a_out;

    void add(BurstType& dst_bst, BurstType& src0_bst, BurstType& src1_bst);
    void mac(BurstType& dst_bst, BurstType& src0_bst, BurstType& src1_bst);
    void mad(BurstType& dst_bst, BurstType& src0_bst, BurstType& src1_bst, BurstType& src2_bst);

    std::string print();
};

// #if 0
// class fim_block {
//   public:
//     uint8_t* input_buffer_;
//     uint8_t* weight_buffer_;
//     uint8_t* accum_buffer_;
//     uint8_t* output_buffer_;
    
//     int width_;
//     int numelem_;

//     fim_block();
//     virtual ~fim_block() {
//         delete[] input_buffer_;
//         delete[] weight_buffer_;
//         delete[] accum_buffer_;
//         delete[] output_buffer_;
//     }
    
//     void initialize_buffer();
//     void clear_buffer();
    
//     void load_input(void* input);
//     void load_weight(void* weight);

//     void execute();
// };
// #endif

} //namespace DRAMSim
#endif
