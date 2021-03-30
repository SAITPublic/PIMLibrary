/*********************************************************************************
 *  Copyright (c) 2010-2011, Elliott Cooper-Balis
 *                             Paul Rosenfeld
 *                             Bruce Jacob
 *                             University of Maryland
 *                             dramninjas [at] gmail [dot] com
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright notice,
 *        this list of conditions and the following disclaimer.
 *
 *     * Redistributions in binary form must reproduce the above copyright
 *notice,
 *        this list of conditions and the following disclaimer in the
 *documentation
 *        and/or other materials provided with the distribution.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *AND
 *  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 *  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 *  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 *  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 *  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *********************************************************************************/
#ifndef __ALU__HPP__
#define __ALU__HPP__

#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
#include "burst.h"

using namespace std;

namespace DRAMSim
{
class fim_block_t
{
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

#if 0
class fim_block {
  public:
    uint8_t* input_buffer_;
    uint8_t* weight_buffer_;
    uint8_t* accum_buffer_;
    uint8_t* output_buffer_;
    
    int width_;
    int numelem_;

    fim_block();
    virtual ~fim_block() {
        delete[] input_buffer_;
        delete[] weight_buffer_;
        delete[] accum_buffer_;
        delete[] output_buffer_;
    }
    
    void initialize_buffer();
    void clear_buffer();
    
    void load_input(void* input);
    void load_weight(void* weight);

    void execute();
};
#endif

}  // namespace DRAMSim
#endif
