#include "gtest/gtest.h"
#include "hip/hip_runtime.h"

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    int numDevices = 1;
    int result = 0;

    const char* env_pim_device = std::getenv("PIM_DEVICE_ID");
    if (env_pim_device != nullptr) {
        std::cout << "User Set Device to " << *env_pim_device << std::endl;
        uint32_t device_id = (*env_pim_device) - '0';
        hipError_t deviceSet = hipSetDevice(device_id);
        if (hipSuccess != deviceSet) {
            std::cout << "Failed to set device " << deviceSet << "Device ID: " << device_id << std::endl;
            return result;
        }
        std::cout << "Executing on Device" << device_id << std::endl;
        result |= RUN_ALL_TESTS();
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
