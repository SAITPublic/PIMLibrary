#ifndef _FIM_PROFILE_H_
#define _FIM_PROFILE_H_

#include "utility/fim_log.h"

long long getTickCount(void);
double getTickFrequency(void);

#ifdef PROFILE
#define FIM_PROFILE_TICK(name) long long __tick_##name = getTickCount()
#define FIM_PROFILE_TOCK(name)                                                                                      \
    DLOG(INFO) << #name                                                                                             \
               << " time (ms) : " << ((double(getTickCount() - __tick_##name) / (double)getTickFrequency())) * 1000 \
               << std::endl;
#else /* !PROFILE */

#define FIM_PROFILE_TICK(name)
#define FIM_PROFILE_TOCK(name)
#endif /* PROFILE */

#endif /* _FIM_PROFILE_H_ */
