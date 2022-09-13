/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#ifndef _PIM_MEMORY_MANAGER_H_
#define _PIM_MEMORY_MANAGER_H_

#include "pim_data_types.h"

namespace pim
{
namespace runtime
{
namespace manager
{
class IPimMemoryManager
{
   public:
    virtual int initialize(void) = 0;
    virtual int deinitialize(void) = 0;
    virtual int alloc_memory(void** ptr, size_t size, PimMemType mem_type) = 0;
    virtual int alloc_memory(PimBo* pim_bo) = 0;
    virtual int free_memory(void* ptr, PimMemType mem_type) = 0;
    virtual int free_memory(PimBo* pim_bo) = 0;
    virtual int copy_memory(void* dst, void* src, size_t size, PimMemCpyType cpy_type) = 0;
    virtual int copy_memory(PimBo* dst, PimBo* src, PimMemCpyType cpy_type) = 0;
    virtual int copy_memory_3d(const PimCopy3D* copy_params) = 0;
    virtual int convert_data_layout(PimBo* dst, PimBo* src) = 0;
};
} /* namespace manager */
} /* namespace runtime */
} /* namespace pim */

#endif /* _PIM_MEMORY_MANAGER_H_ */
