#include <time.h>

long long getTickCount(void)
{
    struct timespec tp;
    clock_gettime(CLOCK_MONOTONIC, &tp);
    return (long long)tp.tv_sec * 1000000000 + tp.tv_nsec;
}

double getTickFrequency(void) { return 1e9; }
