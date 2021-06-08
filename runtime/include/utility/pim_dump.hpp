/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#ifndef _PIM_DUMP_HPP_
#define _PIM_DUMP_HPP_

#include "half.hpp"
#include "manager/PimInfo.h"
#include "pim_data_types.h"
#include "stdio.h"

inline const char* get_pim_op_string(PimOpType op_type)
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
        case OP_DUMMY:
            op_str = "dummy";
            break;
        default:
            op_str = "None";
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
inline int dump_fmtd(const char* filename, PimMemTraceData* fmtd, size_t fmtd_size)
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
        // std::cout << (int)data_a[i] << " : " << (int)data_b[i] << std::endl;
        if (data_a[i] != data_b[i]) {
            ret = -1;
            break;
        }
    }
    return ret;
}

inline int compare_data_round_off(half_float::half* data_a, half_float::half* data_b, int size, float epsilon = 0.001)
{
    int pass_cnt = 0;
    int fail_cnt = 0;
    int ret = 0;
    float abs_diff;
    float max_diff = 0.0;
    float avg_diff = 0.0;

    for (int i = 0; i < size; i++) {
        abs_diff = abs(float(data_a[i]) - float(data_b[i]));
        if (abs_diff < epsilon) {
            pass_cnt++;
        } else {
            fail_cnt++;
            if (max_diff < abs_diff) max_diff = abs_diff;
            avg_diff += abs_diff;
            ret = 1;
        }
    }
    avg_diff /= fail_cnt;

    if (ret) {
        printf("pass_cnt : %d, fail_cnt : %d, pass ratio : %f\n", pass_cnt, fail_cnt,
               ((float)pass_cnt / ((float)fail_cnt + (float)pass_cnt) * 100.));
        printf("max_diff : %f, avg_diff : %f\n", max_diff, avg_diff);
    }

    return ret;
}

inline int compare_half_Ulps(half_float::half data_a, half_float::half data_b, int maxUlpsDiff)
{
    uint16_t Ai = *((uint16_t*)&data_a);
    uint16_t Bi = *((uint16_t*)&data_b);

    if ((Ai & (1 << 15)) != (Bi & (1 << 15))) {
        if (Ai == Bi) return true;
        return false;
    }

    // Find the difference in ULPs.
    int ulpsDiff = abs(Ai - Bi);
    if (ulpsDiff <= maxUlpsDiff) return true;

    return false;
}

inline int compare_half_relative(half_float::half* data_a, half_float::half* data_b, int size)
{
    int pass_cnt = 0;
    int warning_cnt = 0;
    int fail_cnt = 0;
    int ret = 0;

    float max_diff = 0.0;

    for (int i = 0; i < size; i++) {
        if (compare_half_Ulps(data_a[i], data_b[i], 4)) {
            pass_cnt++;
        } else if (compare_half_Ulps(data_a[i], data_b[i], 256)) {
            warning_cnt++;
        } else {
            if (abs(float(data_a[i]) - float(data_b[i])) > max_diff) {
                max_diff = abs(float(data_a[i]) - float(data_b[i]));
            }

            fail_cnt++;
        }
    }

    int quasi_cnt = pass_cnt + warning_cnt;

    printf("pass_cnt : %d, warning_cnt : %d, fail_cnt : %d, pass ratio : %f\n", pass_cnt, warning_cnt, fail_cnt,
           ((float)quasi_cnt / ((float)fail_cnt + (float)warning_cnt + (float)pass_cnt) * 100));

    printf("max diff : %f\n", max_diff);

    return ret;
}

#endif /* _PIM_DUMP_HPP_ */
