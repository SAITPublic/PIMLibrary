#ifndef _FIM_PROFILE_H_
#define _FIM_PROFILE_H_

#include "utility/fim_log.h"

long long getTickCount(void);
double getTickFrequency(void);

#ifdef PROFILE
#define FIM_PROFILE_TICK(name) long long __tick_##name = getTickCount()
#define FIM_PROFILE_TOCK(name) \
    DLOG(INFO) << #name        \
               << " time (ms) : " << (double(getTickCount() - __tick_##name) / (double)getTickFrequency()) * 1000;

#else /* !PROFILE */

#define FIM_PROFILE_TICK(name) long long __tick_##name = getTickCount()
#define FIM_PROFILE_TOCK(name) \
    LOGD(FIM_PROFILE, "%s : %f ", #name, double(getTickCount() - __tick_##name) / (double)getTickFrequency())
#endif /* PROFILE */

#endif /* _FIM_PROFILE_H_ */
