/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#ifndef _FIM_PROFILE_H_
#define _FIM_PROFILE_H_

#include "utility/fim_log.h"

long long getTickCount(void);
double getTickFrequency(void);

#ifdef PROFILE
#define FIM_PROFILE_TICK(name) long long __tick_##name = getTickCount()

#if CONSOLE
#define FIM_PROFILE_TOCK(name)                                                                                     \
    std::cout << #name                                                                                             \
              << " time (ms) : " << ((double(getTickCount() - __tick_##name) / (double)getTickFrequency())) * 1000 \
              << std::endl;
#else
#define FIM_PROFILE_TOCK(name) \
    DLOG(INFO) << #name        \
               << " time (ms) : " << ((double(getTickCount() - __tick_##name) / (double)getTickFrequency())) * 1000;
#endif
#else /* !PROFILE */

#define FIM_PROFILE_TICK(name)
#define FIM_PROFILE_TOCK(name)
#endif /* PROFILE */

/* Always Profile in Console */
#define FIM_PROFILE_TICK_A(name) long long __tick_##name = getTickCount()
#define FIM_PROFILE_TOCK_A(name)                                                                                   \
    std::cout << #name                                                                                             \
              << " time (ms) : " << ((double(getTickCount() - __tick_##name) / (double)getTickFrequency())) * 1000 \
              << std::endl;

#endif /* _FIM_PROFILE_H_ */
