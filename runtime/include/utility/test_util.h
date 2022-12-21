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

#include <iostream>
#include <random>
#include "half.hpp"
#include "manager/PimInfo.h"
#include "pim_data_types.h"
#include "stdio.h"

class Testing
{
   public:
    // virtual void prepare();
    virtual void run(bool block = true, unsigned niter = 1) {}
    virtual int validate(float epsilon = 1e-5) {}
    virtual void prepare(float alpha = 1.0f, float beta = 0.0f, float variation = 0.01f) {}
};
