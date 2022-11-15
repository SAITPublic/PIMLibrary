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

#ifndef _PIM_MANAGER_H_
#define _PIM_MANAGER_H_

#include "manager/IPimMemoryManager.h"
#include "manager/PimDevice.h"
#include "manager/PimInfo.h"
#include "pim_data_types.h"

namespace pim
{
namespace runtime
{
namespace manager
{
class PimManager
{
   public:
    virtual ~PimManager(void);

    static PimManager* get_instance(PimRuntimeType rt_type, PimPrecision precision);

    int initialize(void);
    int deinitialize(void);
    int alloc_memory(void** ptr, size_t size, PimMemType mem_type);
    int alloc_memory(PimBo* pim_bo);
    int free_memory(void* ptr, PimMemType mem_type);
    int free_memory(PimBo* pim_bo);
    int copy_memory(void* dst, void* src, size_t size, PimMemCpyType cpy_type);
    int copy_memory(PimBo* dst, PimBo* src, PimMemCpyType);
    int copy_memory_3d(const PimCopy3D* copy_params);
    int convert_data_layout(PimBo* dst, PimBo* src, bool reorder_on_device);
    void set_gemm_order(PimGemmOrder gemm_order);

    uint8_t* get_crf_binary(void);
    int get_crf_size(void);
    std::shared_ptr<PimDevice> get_pim_device(void) { return pim_device_; }
    std::shared_ptr<IPimMemoryManager> get_pim_manager(void) { return pim_memory_manager_; }

   private:
    PimManager(PimRuntimeType rt_type, PimPrecision precision);
    std::shared_ptr<IPimMemoryManager> pim_memory_manager_;
    std::shared_ptr<PimDevice> pim_device_;

    PimRuntimeType rt_type_;
    PimPrecision precision_;

    uint8_t h_binary_buffer_[128] = {
        0,
    };
    int crf_size_;
};
} /* namespace manager */
} /* namespace runtime */
} /* namespace pim */

#endif /* _PIM_MANAGER_H_ */
