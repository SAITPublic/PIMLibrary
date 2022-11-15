/*
 * Copyright (C) 2022 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 * (Use of the Software is restricted to non-commercial, personal or academic, research purpose only)
 */

#include <assert.h>
#include <gtest/gtest.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <iostream>
#include "half.hpp"
#include "hip/hip_runtime.h"
#include "pim_runtime_api.h"
#include "utility/pim_debug.hpp"

#define NUM (1024)
#define KNRM "\x1B[0m"
#define KRED "\x1B[31m"
#define KBLUE "\x1b[34m"

#define HIPCHECK(error)                                                                                               \
    {                                                                                                                 \
        hipError_t localError = error;                                                                                \
        if ((localError != hipSuccess) && (localError != hipErrorPeerAccessAlreadyEnabled) &&                         \
            (localError != hipErrorPeerAccessNotEnabled)) {                                                           \
            printf("%serror: '%s'(%d) from %s at %s:%d%s\n", KRED, hipGetErrorString(localError), localError, #error, \
                   __FILE__, __LINE__, KNRM);                                                                         \
            printf("%sAPI returned error code.%s\n", KRED, KNRM);                                                     \
        }                                                                                                             \
    }

#define PIMCHECK(error)                                                                      \
    {                                                                                        \
        if (error != 0) {                                                                    \
            printf("%serror: from %s at %s:%d%s\n", KRED, #error, __FILE__, __LINE__, KNRM); \
            printf("%sAPI returned error code.%s\n", KRED, KNRM);                            \
        }                                                                                    \
    }

using namespace std;
using half_float::half;

void checkPeer2PeerSupport()
{
    int gpuCount;
    int canAccessPeer;

    HIPCHECK(hipGetDeviceCount(&gpuCount));
    cout << " gpu count : " << gpuCount << endl;

    for (int currentGpu = 0; currentGpu < gpuCount; currentGpu++) {
        HIPCHECK(hipSetDevice(currentGpu));

        for (int peerGpu = 0; peerGpu < currentGpu; peerGpu++) {
            if (currentGpu != peerGpu) {
                HIPCHECK(hipDeviceCanAccessPeer(&canAccessPeer, currentGpu, peerGpu));
                printf("currentGpu#%d canAccessPeer: peerGpu#%d=%d\n", currentGpu, peerGpu, canAccessPeer);
            }

            HIPCHECK(hipSetDevice(peerGpu));
            HIPCHECK(hipDeviceReset());
        }
        HIPCHECK(hipSetDevice(currentGpu));
        HIPCHECK(hipDeviceReset());
    }
}

void enablePeer2Peer(int currentGpu, int peerGpu)
{
    int canAccessPeer;

    // Must be on a multi-gpu system:
    assert(currentGpu != peerGpu);

    PIMCHECK(PimSetDevice(currentGpu));
    hipDeviceCanAccessPeer(&canAccessPeer, currentGpu, peerGpu);
    if (canAccessPeer == 1) {
        HIPCHECK(hipDeviceEnablePeerAccess(peerGpu, 0));
    } else
        printf("peer2peer transfer not possible between the selected gpu devices");
}

void disablePeer2Peer(int currentGpu, int peerGpu)
{
    int canAccessPeer;

    // Must be on a multi-gpu system:
    assert(currentGpu != peerGpu);

    PIMCHECK(PimSetDevice(currentGpu));
    hipDeviceCanAccessPeer(&canAccessPeer, currentGpu, peerGpu);

    if (canAccessPeer == 1) {
        HIPCHECK(hipDeviceDisablePeerAccess(peerGpu));
    } else
        printf("peer2peer disable not required");
}

int pim_d2d_test()
{
    int gpuCount;
    int errors = 0;
    double eps = 1.0E-6;

    half* device0_data;
    half* device1_data;
    half* randArray;
    half* output_data;

    PimInitialize(RT_TYPE_HIP, PIM_FP16);
    HIPCHECK(hipGetDeviceCount(&gpuCount));

    if (gpuCount < 2) {
        printf("Peer2Peer application requires at least 2 gpu devices");
        return 0;
    }

    randArray = (half*)malloc(NUM * sizeof(half));
    output_data = (half*)malloc(NUM * sizeof(half));

    for (int i = 0; i < NUM; i++) {
        randArray[i] = (half)i * 1.0f;
    }

    for (int currentGpu = 0; currentGpu < gpuCount; currentGpu++) {
        for (int peerGpu = currentGpu + 1; peerGpu < gpuCount; peerGpu++) {
            errors = 0;
            enablePeer2Peer(currentGpu, peerGpu);
            PimAllocMemory((void**)&device0_data, NUM * sizeof(half), MEM_TYPE_PIM);
            PimCopyMemory(device0_data, randArray, NUM * sizeof(half), HOST_TO_PIM);

            PIMCHECK(PimSetDevice(peerGpu));
            PimAllocMemory((void**)&device1_data, NUM * sizeof(half), MEM_TYPE_PIM);

            PimCopyMemory(device1_data, device0_data, NUM * sizeof(half), PIM_TO_PIM);
            PimCopyMemory(output_data, device1_data, NUM * sizeof(half), PIM_TO_HOST);
            hipDeviceSynchronize();

            disablePeer2Peer(currentGpu, peerGpu);

            // verify the results
            for (int i = 0; i < NUM; i++) {
                if (std::abs(randArray[i] - output_data[i]) > eps) {
                    // printf("failed : %d cpu: %f gpu peered data  %f\n", i, (float)randArray[i],
                    //(float)output_data[i]);
                    errors++;
                } else {
                    // printf("passed : %d cpu: %f gpu peered data  %f\n", i, (float)randArray[i],
                    // (float)output_data[i]);
                }
            }

            PIMCHECK(PimSetDevice(currentGpu));
            PimFreeMemory(device0_data, MEM_TYPE_PIM);

            PIMCHECK(PimSetDevice(peerGpu));
            PimFreeMemory(device1_data, MEM_TYPE_PIM);

            if (errors != 0) {
                printf("%sfailed%s, current_gpu : %d , peer_gpu : %d\n", KRED, KNRM, currentGpu, peerGpu);
            }
        }
    }

    free(randArray);
    free(output_data);
    PimDeinitialize();

    return 0;
}

TEST(HIPIntegrationTest, pimD2DTest) { EXPECT_TRUE(pim_d2d_test() == 0); }
