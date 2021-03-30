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
#include "fim_runtime_api.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "utility/fim_log.h"

using namespace tensorflow;  // NOLINT(build/namespaces)

void KernelLauncher()
{
    DLOG(INFO) << "Launcher for FIM_Init";

    /* __FIM_API__ call : Initialize FimRuntime */
    FimInitialize(RT_TYPE_HIP, FIM_FP16);
}

class FimInitOp : public OpKernel
{
   public:
    explicit FimInitOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override { KernelLauncher(); }
};

REGISTER_KERNEL_BUILDER(Name("FimInit").Device(DEVICE_GPU), FimInitOp);
