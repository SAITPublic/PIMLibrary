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

#endif /* _FIM_DUMP_HPP_ */
