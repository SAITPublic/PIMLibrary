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
#include "gtest/gtest.h"
#include "hip/hip_runtime.h"

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    int numDevices = 1;
    int result = 0;

    if (argc == 2) {
        std::string device_str = (argv[1]);
        std::string delimiter = "=";
        size_t pos = 0;
        std::string token;
        if ((pos = device_str.find(delimiter)) != std::string::npos) {
            token = device_str.substr(0, pos);
            if (token == "--pim-device") {
                std::string device = device_str.substr(pos + 1);
                int device_id = stoi((device));
                hipError_t deviceSet = hipSetDevice(device_id);
                if (hipSuccess != deviceSet) {
                    std::cout << "Failed to set device " << deviceSet << "Device ID: " << device_id << std::endl;
                    return result;
                }
                std::cout << "Executing on Device" << device_id << std::endl;
                result |= RUN_ALL_TESTS();
            }  //--pim-device
        }
    } else {
        hipGetDeviceCount(&numDevices);
        std::cout << "Running on Available GPU devices " << numDevices << std::endl;
        for (int device_id = 0; device_id < numDevices; device_id++) {
            hipError_t device_set = hipSetDevice(device_id);
            if (hipSuccess != device_set) {
                std::cout << "Failed to set device " << device_set << std::endl;
                return 0;
            }
            std::cout << "Executing on Device" << device_id << std::endl;
            result |= RUN_ALL_TESTS();
        }
    }
    return result;
}
