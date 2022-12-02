#include <gtest/gtest.h>
#include "hip/hip_runtime.h"
#include "iostream"

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    int result = 0;
    int numDevices = 1;
    int device_id;
    hipGetDeviceCount(&numDevices);
    // Check atleast for 1 device
    if (numDevices < 1) {
        std::cout << "No device found ... exiting" << std::endl;
        return -1;
    }
    if (argc == 2) {
        std::string device_str = (argv[1]);
        std::string delimiter = "=";
        size_t pos = 0;
        std::string token;
        if ((pos = device_str.find(delimiter)) != std::string::npos) {
            token = device_str.substr(0, pos);
            if (token == "--pim-device") {
                std::string device = device_str.substr(pos + 1);
                device_id = stoi((device));
                hipError_t deviceSet = hipSetDevice(device_id);
                if (hipSuccess != deviceSet) {
                    std::cout << "Failed to set device " << deviceSet << "Device ID: " << device_id << std::endl;
                    return result;
                }
                std::cout << "Executing on Device : " << device_id << std::endl;
                result |= RUN_ALL_TESTS();
            }  //--pim-device
        }
    } else {
        device_id = numDevices - 1;
        hipError_t device_set = hipSetDevice(device_id);
        if (hipSuccess != device_set) {
            std::cout << "Failed to set device " << device_set << std::endl;
            return 0;
        }
        std::cout << "Executing on Device : " << device_id << std::endl;
        result |= RUN_ALL_TESTS();
    }
    return result;
}
