#ifndef _FIM_DUMP_HPP_
#define _FIM_DUMP_HPP_

#include "fim_data_types.h"
#include "stdio.h"

inline const char* get_fim_op_string(FimOpType op_type)
{
    const char* op_str;

    switch (op_type) {
        case OP_GEMV:
            op_str = "gemv";
            break;
        case OP_ELT_ADD:
            op_str = "elt_add";
            break;
        case OP_ELT_MUL:
            op_str = "elt_mul";
            break;
        case OP_BN:
            op_str = "bn";
            break;
        case OP_RELU:
            op_str = "relu";
            break;
    }

    return op_str;
}

inline int load_data(const char* filename, char* data, size_t size)
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

inline int dump_data(const char* filename, char* data, size_t size)
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

inline int dump_hexa_array(const char* filename, char* data, size_t size)
{
    FILE* fp = fopen(filename, "wb");

    if (fp == nullptr) {
        printf("fopen error : %s\n", filename);
        return -1;
    }

    for (int i = 0; i < size; i++) {
        fprintf(fp, "0x%X, ", (unsigned char)data[i]);
    }
    fclose(fp);

    return 0;
}

template <int block_size>
inline int dump_fmtd(const char* filename, FimMemTraceData* fmtd, size_t fmtd_size)
{
    FILE* fp = fopen(filename, "wb");

    if (fp == nullptr) {
        printf("fopen error : %s\n", filename);
        return -1;
    }

    for (int i = 0; i < fmtd_size; i++) {
        fprintf(fp, "%d\t%d\t%d\t%c\t0x%lX", i, fmtd[i].block_id, fmtd[i].thread_id, fmtd[i].cmd, fmtd[i].addr);
        if (fmtd[i].cmd == 'W') {
            fprintf(fp, "\t");
            for (int d = 0; d < block_size; d++) fprintf(fp, "%02X", fmtd[i].data[d]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);

    return 0;
}

inline int compare_data(char* data_a, char* data_b, size_t size)
{
    int ret = 0;

    for (int i = 0; i < size; i++) {
        if (data_a[i] != data_b[i]) {
            ret = -1;
            break;
        }
    }
    return ret;
}

inline int compare_data_round_off(half* data_a, half* data_b, size_t size, double epsilon = 0.001)
{
    for (int i = 0; i < size; i++) {
        if (!(std::abs(data_a[i] - data_b[i]) < epsilon)) {
            return -1;
        }
    }
    return 0;
}

#endif /* _FIM_DUMP_HPP_ */
