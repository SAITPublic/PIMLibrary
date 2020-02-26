#ifndef _FIM_DUMP_HPP_
#define _FIM_DUMP_HPP_

#include "fim_data_types.h"
#include "stdio.h"

int load_data(const char* filename, char* data, size_t size)
{
    FILE* fp = fopen(filename, "r");

    if (fp == nullptr) {
        printf("fopen error : %s\n", filename);
        return -1;
    }

    for (int i = 0; i < size; i++) {
        fscanf(fp, "%c", &data[i]);
    }
    fclose(fp);

    return 0;
}

int dump_data(const char* filename, char* data, size_t size)
{
    FILE* fp = fopen(filename, "wb");

    if (fp == nullptr) {
        printf("fopen error : %s\n", filename);
        return -1;
    }

    for (int i = 0; i < size; i++) {
        fprintf(fp, "%c", data[i]);
    }
    fclose(fp);

    return 0;
}

int dump_fmtd16(const char* filename, FimMemTraceData* fmtd16, size_t fmtd16_size)
{
    FILE* fp = fopen(filename, "wb");

    if (fp == nullptr) {
        printf("fopen error : %s\n", filename);
        return -1;
    }

    for (int i = 0; i < fmtd16_size; i++) {
        fprintf(fp, "%d\t%d\t%d\t%c\t0x%lX", i, fmtd16[i].block_id, fmtd16[i].thread_id, fmtd16[i].cmd, fmtd16[i].addr);
        if (fmtd16[i].cmd == 'W') {
            fprintf(fp, "\t");
            for (int d = 0; d < 2; d++) fprintf(fp, "%016lX", fmtd16[i].data[d]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);

    return 0;
}

#endif /* _FIM_DUMP_HPP_ */
