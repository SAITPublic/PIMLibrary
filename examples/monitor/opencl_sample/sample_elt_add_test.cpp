#include <CL/opencl.h>
#include <stdio.h>
#include <iostream>
#include "executor/pim_opencl_kernels/gpu_add_kernel.pimk"

int length = 256;
cl_half* inp1 = NULL;
cl_half* inp2 = NULL;
cl_half* out = NULL;
cl_half* ref = NULL;

cl_half* temp1 = NULL;
cl_half* temp2 = NULL;

void add_vector()
{
    for (int i = 0; i < length; i++) {
        ref[i] = inp1[i] + inp2[i];
    }
}

void initHost()
{
    size_t sizeInBytes = length * sizeof(cl_half);
    inp1 = (cl_half*)malloc(sizeInBytes);
    inp2 = (cl_half*)malloc(sizeInBytes);
    ref = (cl_half*)malloc(sizeInBytes);
    out = (cl_half*)malloc(sizeInBytes);

    temp1 = (cl_half*)malloc(sizeInBytes);
    temp2 = (cl_half*)malloc(sizeInBytes);

    for (int i = 0; i < length; i++) {
        inp1[i] = i;
        inp2[i] = i;
    }
    add_vector();
}

int compare_results()
{
    int ret = 1;
    for (int i = 0; i < length; i++) {
        if (ref[i] != out[i]) {
            std::cout << "ref : " << (cl_half)ref[i] << " :::  observed : " << (cl_half)out[i] << "\n";
            ret = -1;
        }
    }
    return ret;
}

int compare_vectors(cl_half* a, cl_half* b)
{
    for (int i = 0; i < length; i++) {
        if (a[i] != b[i]) {
            return -1;
        }
    }
    return 0;
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

    clBuildProgram(program, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        std::cout << "program building unsuccessfull" << err << "\n";
    }

    cl_kernel kernel = clCreateKernel(program, "gpuAdd", &err);
    if (err != CL_SUCCESS) {
        std::cout << "kernel creation unsuccessfull" << err << "\n";
    }

    cl_mem buffer_1 = clCreateBuffer(context, CL_MEM_READ_WRITE, length * sizeof(cl_half), NULL, NULL);
    cl_mem buffer_2 = clCreateBuffer(context, CL_MEM_READ_WRITE, length * sizeof(cl_half), NULL, NULL);
    cl_mem out_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, length * sizeof(cl_half), NULL, NULL);

    clEnqueueWriteBuffer(queue, buffer_1, CL_TRUE, 0, length * sizeof(cl_half), inp1, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, buffer_2, CL_TRUE, 0, length * sizeof(cl_half), inp2, 0, NULL, NULL);

    clEnqueueReadBuffer(queue, buffer_1, CL_TRUE, 0, length * sizeof(cl_half), temp1, 0, NULL, NULL);
    if (compare_vectors(temp1, inp1) != 0) {
        std::cout << "mem copy unsuccessfull\n";
    }

    clEnqueueReadBuffer(queue, buffer_2, CL_TRUE, 0, length * sizeof(cl_half), temp2, 0, NULL, NULL);
    if (compare_vectors(temp2, inp2) != 0) {
        std::cout << "mem copy unsuccessfull\n";
    }

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&buffer_1);
    if (err != CL_SUCCESS) {
        std::cout << "setting kernel argument unsuccessfull" << err << "\n";
    }
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&buffer_2);
    if (err != CL_SUCCESS) {
        std::cout << "setting kernel argument unsuccessfull" << err << "\n";
    }
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&out_buffer);
    if (err != CL_SUCCESS) {
        std::cout << "setting kernel argument unsuccessfull" << err << "\n";
    }
    clSetKernelArg(kernel, 3, sizeof(unsigned int), (void*)&length);
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
