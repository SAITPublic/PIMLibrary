#ifndef _FIM_DUMP_HPP_
#define _FIM_DUMP_HPP_

#include "fim_data_types.h"
#include "stdio.h"

template <typename T>
int load_fp16_data(const char* filename, T* data, size_t size)
{
    FILE* fp = fopen(filename, "r");
    int cnt = size / sizeof(T);

    if (fp == nullptr) {
        printf("fopen error : %s\n", filename);
        return -1;
    }

    for (int i = 0; i < cnt; i++) {
        fscanf(fp, "%d", &data[i]);
    }
    fclose(fp);

    return 0;
}

template <typename T>
int dump_fp16_data(const char* filename, T* data, size_t size)
{
    FILE* fp = fopen(filename, "wb");
    int cnt = size / sizeof(T);

    if (fp == nullptr) {
        printf("fopen error : %s\n", filename);
        return -1;
    }

    for (int i = 0; i < cnt; i++) {
        fprintf(fp, "%d ", data[i]);
    }
    fclose(fp);

    return 0;
}

int dump_fmtd16(const char* filename, FimMemTraceData* fmtd16, size_t fmtd16_size)
{
    FILE* fp = fopen(filename, "wb");
    int cnt = fmtd16_size;

    if (fp == nullptr) {
        printf("fopen error : %s\n", filename);
        return -1;
    }

    for (int i = 0; i < cnt; i++) {
        fprintf(fp, "%d\t%d\t%d\t%c\t0x%lX\t", i, fmtd16[i].block_id, fmtd16[i].thread_id, fmtd16[i].cmd,
                fmtd16[i].addr);
        if (fmtd16[i].cmd == 'W') {
            for (int d = 0; d < 16; d++) fprintf(fp, "%c ", fmtd16[i].data[d]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);

    return 0;
}

#endif /* _FIM_DUMP_HPP_ */
