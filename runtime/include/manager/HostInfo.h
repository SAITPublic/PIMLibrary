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

#ifndef _HOST_INFO_H_
#define _HOST_INFO_H_

#define MAX_NUM_GPUS 10

typedef enum __HostType {
    AMDGPU,
    CPU,
    FPGA,
} HostType;

typedef struct __HostInfo {
    HostType host_type;
    uint32_t node_id;
    uint32_t host_id;
    uint32_t base_address;
} HostInfo;

#endif /* _HOST_INFO_H_ */
