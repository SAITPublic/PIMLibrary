#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.hpp>
#include <iostream>
#include <string>

using std::cerr;
using std::cout;
using std::endl;
using std::string;

/////////////////////////////////////////////////////////////////
// Helper function to print vector elements
/////////////////////////////////////////////////////////////////
void printVector(const std::string arrayName, const cl_float *arrayData, const unsigned int length)
{
    int numElementsToPrint = (256 < length) ? 256 : length;
    cout << endl << arrayName << ":" << endl;
    for (int i = 0; i < numElementsToPrint; ++i) cout << arrayData[i] << " ";
    cout << endl;
}

/////////////////////////////////////////////////////////////////
// Globals
/////////////////////////////////////////////////////////////////
int length = 256;
cl_float *pX = NULL;
cl_float *pY = NULL;
cl_float a = 2.f;

std::vector<cl::Platform> platforms;
cl::Context context;
std::vector<cl::Device> devices;
cl::CommandQueue queue;
cl::Program program;

cl::Kernel kernel;
cl::Buffer bufX;
cl::Buffer bufY;

/////////////////////////////////////////////////////////////////
// The saxpy kernel
/////////////////////////////////////////////////////////////////
string kernelStr =
    "__kernel void saxpy(const global float * x,\n"
    "                    __global float * y,\n"
    "                    const float a)\n"
    "{\n"
    "   uint gid = get_global_id(0);\n"
    "   y[gid] = a* x[gid] + y[gid];\n"
    "}\n";

/////////////////////////////////////////////////////////////////
// Allocate and initialize memory on the host
/////////////////////////////////////////////////////////////////
void initHost()
{
    size_t sizeInBytes = length * sizeof(cl_float);
    pX = (cl_float *)malloc(sizeInBytes);
    if (pX == NULL) throw(string("Error: Failed to allocate input memory on host\n"));

    pY = (cl_float *)malloc(sizeInBytes);
    if (pY == NULL) throw(string("Error: Failed to allocate input memory on host\n"));

    for (int i = 0; i < length; i++) {
        pX[i] = cl_float(i);
        pY[i] = cl_float(length - 1 - i);
    }

    printVector("X", pX, length);
    printVector("Y", pY, length);
}

/////////////////////////////////////////////////////////////////
// Release host memory
/////////////////////////////////////////////////////////////////
void cleanupHost()
{
    if (pX) {
        free(pX);
        pX = NULL;
    }
    if (pY != NULL) {
        free(pY);
        pY = NULL;
    }
}

int main(int argc, char *argv[])
{
    try {
        /////////////////////////////////////////////////////////////////
        // Allocate and initialize memory on the host
        /////////////////////////////////////////////////////////////////
        initHost();

        /////////////////////////////////////////////////////////////////
        // Find the platform
        /////////////////////////////////////////////////////////////////
        cl::Platform::get(&platforms);
        std::vector<cl::Platform>::iterator iter;
        for (iter = platforms.begin(); iter != platforms.end(); ++iter) {
            if (!strcmp((*iter).getInfo<CL_PLATFORM_VENDOR>().c_str(), "Advanced Micro Devices, Inc.")) {
                break;
            }
        }

        /////////////////////////////////////////////////////////////////
        // Create an OpenCL context
        /////////////////////////////////////////////////////////////////
        cl_context_properties cps[3] = {CL_CONTEXT_PLATFORM, (cl_context_properties)(*iter)(), 0};
        context = cl::Context(CL_DEVICE_TYPE_GPU, cps);

        /////////////////////////////////////////////////////////////////
        // Detect OpenCL devices
        /////////////////////////////////////////////////////////////////
        devices = context.getInfo<CL_CONTEXT_DEVICES>();

        /////////////////////////////////////////////////////////////////
        // Create an OpenCL command queue
        /////////////////////////////////////////////////////////////////
        queue = cl::CommandQueue(context, devices[0]);

        /////////////////////////////////////////////////////////////////
        // Create OpenCL memory buffers
        /////////////////////////////////////////////////////////////////
        bufX = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * length, pX);
        bufY = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * length, pY);

        /////////////////////////////////////////////////////////////////
        // Load CL file, build CL program object, create CL kernel object
        /////////////////////////////////////////////////////////////////
        cl::Program::Sources sources(1, std::make_pair(kernelStr.c_str(), kernelStr.length()));
        program = cl::Program(context, sources);
        program.build(devices);
        kernel = cl::Kernel(program, "saxpy");

        /////////////////////////////////////////////////////////////////
        // Set the arguments that will be used for kernel execution
        /////////////////////////////////////////////////////////////////
        kernel.setArg(0, bufX);
        kernel.setArg(1, bufY);
        kernel.setArg(2, a);

        /////////////////////////////////////////////////////////////////
        // Enqueue the kernel to the queue
        // with appropriate global and local work sizes
        /////////////////////////////////////////////////////////////////
        queue.enqueueNDRangeKernel(kernel, cl::NDRange(), cl::NDRange(length), cl::NDRange(64));

        /////////////////////////////////////////////////////////////////
        // Enqueue blocking call to read back buffer Y
        /////////////////////////////////////////////////////////////////
        queue.enqueueReadBuffer(bufY, CL_TRUE, 0, length * sizeof(cl_float), pY);

        printVector("Y", pY, length);

        /////////////////////////////////////////////////////////////////
        // Release host resources
        /////////////////////////////////////////////////////////////////
        cleanupHost();
    } catch (cl::Error err) {
        /////////////////////////////////////////////////////////////////
        // Catch OpenCL errors and print log if it is a build error
        /////////////////////////////////////////////////////////////////
        cerr << "ERROR: " << err.what() << "(" << err.err() << ")" << endl;
        if (err.err() == CL_BUILD_PROGRAM_FAILURE) {
            string str = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
            cout << "Program Info: " << str << endl;
        }
        cleanupHost();
    } catch (string msg) {
        cerr << "Exception caught in main(): " << msg << endl;
        cleanupHost();
    }
}
