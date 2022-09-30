#include "gtest/gtest.h"
#include "hip/hip_runtime.h"

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    int result = 0;
    int numDevices = 1;
    hipGetDeviceCount(&numDevices);
    // Check atleast for 1 device
    if(numDevices < 1){
        std::cout << "No device found ... exiting" << std::endl;
        return -1;
    }
    // Always default to 0 for now
    hipError_t deviceSet = hipSetDevice(0);
    if (hipSuccess != deviceSet) {
        std::cout << "Failed to set device " << deviceSet << "Device ID: " << device_id << std::endl;
        return result;
    }
    std::cout << "Executing on Device" << device_id << std::endl;
    result |= RUN_ALL_TESTS();

    return result;
}
