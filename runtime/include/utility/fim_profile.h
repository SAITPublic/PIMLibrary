#ifndef _FIM_PROFILE_H_
#define _FIM_PROFILE_H_

#include "utility/fim_log.h"

long long getTickCount(void);
double getTickFrequency(void);

#define CONSOL 0

#ifdef PROFILE
#define FIM_PROFILE_TICK(name) long long __tick_##name = getTickCount()

#if CONSOL
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
