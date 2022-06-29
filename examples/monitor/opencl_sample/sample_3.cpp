#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.hpp>
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

std::vector<cl::Platform> platforms;
cl::Context context;
std::vector<cl::Device> devices;
cl::CommandQueue queue;

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
    cl::Platform::get(&platforms);
    std::vector<cl::Platform>::iterator iter;
    for (iter = platforms.begin(); iter != platforms.end(); ++iter) {
        if (!strcmp((*iter).getInfo<CL_PLATFORM_VENDOR>().c_str(), "Advanced Micro Devices, Inc.")) {
            break;
        }
    }

    cl_context_properties cps[3] = {CL_CONTEXT_PLATFORM, (cl_context_properties)(*iter)(), 0};
    context = cl::Context(CL_DEVICE_TYPE_GPU, cps);
    devices = context.getInfo<CL_CONTEXT_DEVICES>();
    queue = cl::CommandQueue(context, devices[0]);
    cl::Buffer host_inp_buffer = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, bytes, host_input, &err);
    cl::Buffer device_buffer = cl::Buffer(context, CL_MEM_READ_WRITE, bytes, NULL, &err);
    // host_out_buffer = clCreateBuffer(context , CL_MEM_READ_WRITE | CL_MEM_ALLOC_USE_HOST_PTR , bytes , NULL , &err);

    queue.enqueueWriteBuffer(device_buffer, CL_TRUE, 0, bytes, host_input);
    queue.enqueueReadBuffer(device_buffer, CL_TRUE, 0, bytes, host_output);

    cout << "result : " << validate_result(host_input, host_output) << endl;
    return 0;
}
