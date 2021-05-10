/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#include <iostream>
#include "pim_runtime_api.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "utility/pim_log.h"

using namespace tensorflow;  // NOLINT(build/namespaces)

void KernelLauncher()
{
    DLOG(INFO) << "Launcher for PIM_Init";

    /* __PIM_API__ call : Initialize PimRuntime */
    PimInitialize(RT_TYPE_HIP, PIM_FP16);
}

class PimInitOp : public OpKernel
{
   public:
    explicit PimInitOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override { KernelLauncher(); }
};

REGISTER_KERNEL_BUILDER(Name("PimInit").Device(DEVICE_GPU), PimInitOp);
