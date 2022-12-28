/*
 * Copyright (C) 2022 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 * (Use of the Software is restricted to non-commercial, personal or academic, research purpose only)
 */

#include <assert.h>
#include <gtest/gtest.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <iostream>
#include "utility/pim_debug.hpp"
#include "utility/pim_profile.h"
#include "half.hpp"
#include "pim_runtime_api.h"
#include "test_util.h"

using namespace std;
using half_float::half;

void inline calculate_relu(half* input, half* output, int len)
{
    for (int i = 0; i < len; i++) {
        output[i] = (input[i] > 0) ? input[i] : (half)0;
    }
}

class PimReluTest : public Testing
{
   public:
    PimReluTest(unsigned in_length) : n_(in_length)
    {
        PimDesc* pim_desc = PimCreateDesc(1, 1, 1, n_, PIM_FP16);

        /* __PIM_API__ call : Create PIM Buffer Object */
        host_input_ = PimCreateBo(pim_desc, MEM_TYPE_HOST);
        host_out_ = PimCreateBo(pim_desc, MEM_TYPE_HOST);
        ref_out_ = PimCreateBo(pim_desc, MEM_TYPE_HOST);
        device_input_ = PimCreateBo(pim_desc, MEM_TYPE_PIM);
        device_output_ = PimCreateBo(pim_desc, MEM_TYPE_PIM);
    }

    ~PimReluTest(void)
    {
        PimDestroyBo(host_input_);
        PimDestroyBo(host_out_);
        PimDestroyBo(ref_out_);
        PimDestroyBo(device_output_);
        PimDestroyBo(device_input_);
    }

    virtual void prepare(float alpha = 1.0f, float beta = 0.0f, float variation = 0.2f) override
    {
        set_rand_half_data((half*)host_input_->data, (half)0.5, n_);
        calculate_relu((half*)host_input_->data, (half*)ref_out_->data, n_);

        PimCopyMemory(device_input_, host_input_, HOST_TO_PIM);
    }

    virtual void run(bool block = true, unsigned niter = 1) override
    {
        for (unsigned i = 0; i < niter; ++i) {
            PimExecuteRelu(device_output_, device_input_, nullptr, true);
            PimCopyMemory(host_out_, device_output_, PIM_TO_HOST);
        }
    }

    virtual int validate(float epsilon = 0.1f) override
    {
        return compare_half_relative((half*)host_out_->data, (half*)ref_out_->data, n_);
    }

   private:
    unsigned n_;

    PimBo *host_input_, *host_out_;
    PimBo *device_input_, *device_output_;
    PimBo* ref_out_;
};

class PimReluTestFixture : public ::testing::Test
{
   protected:
    void SetUp(PimRuntimeType plt)
    {
        PimInitialize(plt, PIM_FP16);
        PimExecuteDummy();
    }
    virtual void TearDown(void) override { PimDeinitialize(); }
    int ExecuteTest(unsigned n, bool block = true)
    {
        PimReluTest pimReluTest = PimReluTest(n);
        pimReluTest.prepare();
        pimReluTest.run(block);
        return pimReluTest.validate();
    }
};
