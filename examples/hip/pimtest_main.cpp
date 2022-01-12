#include "gtest/gtest.h"
#include "hip/hip_runtime.h"

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    int numDevices = 0;
    int result = 0;
    hipGetDeviceCount(&numDevices);
    std::cout << "Available GPU devices " << numDevices << std::endl;

    for (int deviceID = 0; deviceID < numDevices; deviceID++) {
        hipError_t deviceSet = hipSetDevice(deviceID);
        if (hipSuccess != deviceSet) {
            std::cout << "Failed to set device " << deviceSet << std::endl;
            return 0;
        }
        std::cout << "Executing on Device" << deviceID << std::endl;
        result |= RUN_ALL_TESTS();
    }
    return result;
}
