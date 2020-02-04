#ifndef _FIM_LOG_H_
#define _FIM_LOG_H_

#include <stdio.h>

enum __log_level {
    FIM_LOG_ERROR = 1,
    FIM_LOG_WARNING = 2,
    FIM_LOG_DEBUG = 3,
    FIM_LOG_INFO = 4,
};

/* fim log tags */
#define FIM_API "FIM_API"
#define FIM_RT "FIM_RT"
#define FIM_MG "FIM_MG"
#define FIM_MEM_MG "FIM_MEM_MG"
#define FIM_CON_MG "FIM_CON_MG"
#define FIM_DEV "FIM_DEV"
#define FIM_EXE "FIM_EXE"
#define FIM_EMUL "FIM_EMUL"

static int FIM_LOG_LEVEL = FIM_LOG_INFO;

#define CHECK_LOG_LEVEL(LOG_LEVEL) (((FIM_LOG_LEVEL) >= (LOG_LEVEL)) != 0)

#ifdef DEBUG

#define LOGI(LOG_TAG, ...)                                                                  \
    {                                                                                       \
        if (!CHECK_LOG_LEVEL(FIM_LOG_INFO))                                                 \
            ;                                                                               \
        else {                                                                              \
            fprintf(stdout, "[FIMLib][%s][INFO] %s:%d: ", LOG_TAG, __FUNCTION__, __LINE__); \
            fprintf(stdout, __VA_ARGS__);                                                   \
            fprintf(stdout, "\n");                                                          \
        }                                                                                   \
    }

#define LOGD(LOG_TAG, ...)                                                                   \
    {                                                                                        \
        if (!CHECK_LOG_LEVEL(FIM_LOG_DEBUG))                                                 \
            ;                                                                                \
        else {                                                                               \
            fprintf(stdout, "[FIMLib][%s][DEBUG] %s:%d: ", LOG_TAG, __FUNCTION__, __LINE__); \
            fprintf(stdout, __VA_ARGS__);                                                    \
            fprintf(stdout, "\n");                                                           \
        }                                                                                    \
    }

#define LOGW(LOG_TAG, ...)                                                                     \
    {                                                                                          \
        if (!CHECK_LOG_LEVEL(FIM_LOG_WARNING))                                                 \
            ;                                                                                  \
        else {                                                                                 \
            fprintf(stdout, "[FIMLib][%s][WARNING] %s:%d: ", LOG_TAG, __FUNCTION__, __LINE__); \
            fprintf(stdout, __VA_ARGS__);                                                      \
            fprintf(stdout, "\n");                                                             \
        }                                                                                      \
    }

#define LOGE(LOG_TAG, ...)                                                                   \
    {                                                                                        \
        if (!CHECK_LOG_LEVEL(FIM_LOG_ERROR))                                                 \
            ;                                                                                \
        else {                                                                               \
            fprintf(stdout, "[FIMLib][%s][ERROR] %s:%d: ", LOG_TAG, __FUNCTION__, __LINE__); \
            fprintf(stdout, __VA_ARGS__);                                                    \
            fprintf(stdout, "\n");                                                           \
        }                                                                                    \
    }

#else /* !DEBUG */

#define LOGD(FLAG, ...) (void)0
#define LOGI(FLAG, ...) (void)0
#define LOGE(FLAG, ...) (void)0

#endif /* DEBUG */

#endif /* _FIM_LOG_H_ */
