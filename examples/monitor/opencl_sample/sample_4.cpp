#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.h>
#include <iostream>
#include <string>

using std::cout;
using std::cerr;
using std::endl;
using std::string;

void print(cl_float* array, int length)
{
    for (int i = 0; i < length; i++) {
        cout << array[i] << " ";
    }
    cout << endl;
}

cl_int err;
int length = 10;
size_t bytes = length * sizeof(cl_float);
cl_float* host_input = NULL;
cl_float* host_output = NULL;

cl_platform_id cpPlatform;  // OpenCL platform
cl_device_id device_id;     // device ID
cl_context context;         // context
cl_command_queue queue;     // command queue
cl_uint num_gpu_devices;    // num gpu devices

void init_host()
{
    host_input = (cl_float*)malloc(bytes);
    if (host_input == NULL) {
        throw(string("allocation for input failed\n"));
    }
    host_output = (cl_float*)malloc(bytes);
    if (host_output == NULL) {
        throw(string("allocation for output failed\n"));
    }

    for (int i = 0; i < length; i++) {
        host_input[i] = cl_float(i);
    }
}

int validate_result(cl_float* host_input, cl_float* host_output)
{
    for (int i = 0; i < length; i++) {
        if (host_input[i] != host_output[i]) {
            return 0;
        }
    }
    return 1;
}

int main()
{
    init_host();
    clGetPlatformIDs(1, &cpPlatform, NULL);
    clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 0, NULL, &num_gpu_devices);
    clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    queue = clCreateCommandQueueWithProperties(context, device_id, 0, NULL);

    cl_mem device_buffer1 = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, NULL, &err);
    cl_mem device_buffer2 = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, NULL, &err);
    clEnqueueWriteBuffer(queue, device_buffer1, CL_TRUE, 0, bytes, host_input, 0, NULL, NULL);
    clEnqueueCopyBuffer(queue, device_buffer1, device_buffer2, 0, 0, bytes, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, device_buffer2, CL_TRUE, 0, bytes, host_output, 0, NULL, NULL);

    cout << "result : " << validate_result(host_input, host_output) << endl;
    return 0;
}
