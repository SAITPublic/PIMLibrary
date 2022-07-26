/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#ifndef _PIM_PROFILE_H_
#define _PIM_PROFILE_H_

#include "utility/pim_log.h"

long long getTickCount(void);
double getTickFrequency(void);

#ifdef PROFILE
#define PIM_PROFILE_TICK(name) long long __tick_##name = getTickCount()

#if CONSOLE
#define PIM_PROFILE_TOCK(name)                                                                                     \
    std::cout << #name                                                                                             \
              << " time (ms) : " << ((double(getTickCount() - __tick_##name) / (double)getTickFrequency())) * 1000 \
              << std::endl;
#define PIM_PROFILE_TOCK_ITER(name, iter_cnt)                                                                  \
    std::cout << #name << " time (ms) : "                                                                      \
              << ((((double(getTickCount() - __tick_##name) / (double)getTickFrequency())) * 1000) / iter_cnt) \
              << std::endl;
#else
#define PIM_PROFILE_TOCK(name) \
    DLOG(INFO) << #name        \
               << " time (ms) : " << ((double(getTickCount() - __tick_##name) / (double)getTickFrequency())) * 1000;
#define PIM_PROFILE_TOCK_ITER(name, iter_cnt)                                                                   \
    DLOG(INFO) << #name << " time (ms) : "                                                                      \
               << ((((double(getTickCount() - __tick_##name) / (double)getTickFrequency())) * 1000) / iter_cnt) \
               << std::endl;
#endif
#else /* !PROFILE */

#define PIM_PROFILE_TICK(name)
#define PIM_PROFILE_TOCK(name)
#define PIM_PROFILE_TOCK_ITER(name, iter_cnt)
#endif /* PROFILE */

/* Always Profile in Console */
#define PIM_PROFILE_TICK_A(name) long long __tick_##name = getTickCount()
#define PIM_PROFILE_TOCK_A(name)                                                                                   \
    std::cout << #name                                                                                             \
              << " time (ms) : " << ((double(getTickCount() - __tick_##name) / (double)getTickFrequency())) * 1000 \
              << std::endl;
#define PIM_PROFILE_TOCK_ITER_A(name, iter_cnt)                                                                \
    std::cout << #name << " time (ms) : "                                                                      \
              << ((((double(getTickCount() - __tick_##name) / (double)getTickFrequency())) * 1000) / iter_cnt) \
              << std::endl;
#endif /* _PIM_PROFILE_H_ */
