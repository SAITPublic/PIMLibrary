#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.hpp>
#include <iostream>
#include <string>

std::vector<cl::Device> devices;
int find_devices()
{
    std::vector<cl::Platform> platforms;  // get all platforms
    std::vector<cl::Device> devices_available;
    cl_device_id *device_ids;
    int n = 0;  // number of available devices
    cl::Platform::get(&platforms);
    for (int i = 0; i < (int)platforms.size(); i++) {
        devices_available.clear();
        platforms[i].getDevices(CL_DEVICE_TYPE_GPU, &devices_available);
        if (devices_available.size() == 0) continue;  // no device found in plattform i
        for (int j = 0; j < (int)devices_available.size(); j++) {
            n++;
            devices.push_back(devices_available[j]);
        }
    }
    if (platforms.size() == 0 || devices.size() == 0) {
        std::cout << "Error: There are no OpenCL devices available!" << std::endl;
        return -1;
    }
    for (int i = 0; i < n; i++)
        std::cout << "ID: " << i << ", Device: " << devices[i].getInfo<CL_DEVICE_NAME>() << std::endl;
    // clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, device_ids, NULL);

    return n;  // return number of available devices
}

int main()
{
    cl_int num_devices = find_devices();
    std::cout << "num_Devices : " << num_devices << std::endl;
    return 0;
}
