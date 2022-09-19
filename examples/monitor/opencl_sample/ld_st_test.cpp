#include <CL/opencl.h>
#include <stdio.h>
#include <fstream>
#include <iostream>

int length = 4;
cl_half* inp1 = NULL;
cl_half* out = NULL;
cl_half* ref = NULL;

const char* source =
    "#pragma OPENCL EXTENSION cl_khr_fp16 : enable              \n"
    "__kernel void gpuAdd(__global half* a,                     \n"
    "                     __global half* output,                \n"
    "                     const unsigned int n)                 \n"
    "{                                                          \n"
    "   half4 array = vload4(0 , a);                            \n"
    "   vstore4(array , 0 , output);                            \n"
    "}                                                          \n";

void initHost()
{
    size_t sizeInBytes = length * sizeof(cl_half);
    inp1 = (cl_half*)malloc(sizeInBytes);
    ref = (cl_half*)malloc(sizeInBytes);
    out = (cl_half*)malloc(sizeInBytes);

    for (int i = 0; i < length; i++) {
        inp1[i] = i;
    }
}

int compare_results()
{
    int ret = 1;
    for (int i = 0; i < length; i++) {
        if (inp1[i] != out[i]) {
            std::cout << "ref : " << (cl_half)ref[i] << " :::  observed : " << (cl_half)out[i] << "\n";
            ret = -1;
        }
    }
    return ret;
}

int main(int argc, char** argv)
{
    initHost();
    cl_int err;
    // 1. Get a platform.
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);

    // 2. Find a gpu device.
    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    // 3. Create a context and command queue on that device.
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);

    cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL);

    // 4. Perform runtime source compilation, and obtain kernel entry point.
    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source, NULL, &err);
    if (err != CL_SUCCESS) {
        std::cout << "program creation unsuccessfull unsuccessfull" << err << "\n";
    }

    std::string options = "-cl-opt-disable";
    clBuildProgram(program, 1, &device, options.c_str(), NULL, &err);
    if (err != CL_SUCCESS) {
        std::cout << "program building unsuccessfull" << err << "\n";
    }

    cl_kernel kernel = clCreateKernel(program, "gpuAdd", &err);
    if (err != CL_SUCCESS) {
        std::cout << "kernel creation unsuccessfull" << err << "\n";
    }

    cl_mem buffer_1 = clCreateBuffer(context, CL_MEM_READ_WRITE, length * sizeof(cl_half), NULL, NULL);
    cl_mem out_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, length * sizeof(cl_half), NULL, NULL);

    clEnqueueWriteBuffer(queue, buffer_1, CL_TRUE, 0, length * sizeof(cl_half), inp1, 0, NULL, NULL);

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&buffer_1);
    if (err != CL_SUCCESS) {
        std::cout << "setting kernel argument unsuccessfull" << err << "\n";
    }
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&out_buffer);
    if (err != CL_SUCCESS) {
        std::cout << "setting kernel argument unsuccessfull" << err << "\n";
    }
    clSetKernelArg(kernel, 2, sizeof(unsigned int), (void*)&length);
    if (err != CL_SUCCESS) {
        std::cout << "setting kernel argument unsuccessfull" << err << "\n";
    }

    size_t local_work_size = 32;
    size_t global_work_size = ceil(length / (float)local_work_size) * local_work_size;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        std::cout << "kernel execution unsuccessfull" << err << "\n";
    }
    clFinish(queue);
    err = clEnqueueReadBuffer(queue, out_buffer, CL_TRUE, 0, length * sizeof(cl_half), out, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        std::cout << "output buffer read unsuccessfull" << err << "\n";
    }
    int ret = compare_results();
    if (ret == -1) {
        std::cout << "output dont match \n";
    } else {
        std::cout << "outputs match \n";
    }
    return 0;
}
