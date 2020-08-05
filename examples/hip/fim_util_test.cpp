#include <gtest/gtest.h>
#include "hip/hip_fp16.h"
#include "hip/hip_runtime.h"
//#include "utility/fim_util.h"

__global__ void integral_sum(void* out, void* in, int out_size, int reduce_size)
{
    int bcnt = hipGridDim_x;
    int tcnt = hipBlockDim_x;
    int bid = hipBlockIdx_x;
    int tid = hipThreadIdx_x;
    int outcnt_per_thread = out_size / (bcnt * tcnt);
    half* t_out = (half*)out;
    half* t_in = (half*)in;

    int in_idx = bid * tcnt * reduce_size * outcnt_per_thread + tid * reduce_size * outcnt_per_thread;
    int out_idx = bid * tcnt * outcnt_per_thread + tid * outcnt_per_thread;

    for (int outcnt = 0; outcnt < outcnt_per_thread; outcnt++) {
        for (int i = 0; i < reduce_size; i++) {
            t_out[out_idx] += t_in[in_idx + i];
        }
        out_idx++;
        t_in += reduce_size;
    }
}

int gpu_integral_sum(void)
{
    int ret = -1;
    int in_size = 16 * 4096;
    int out_size = 4096;
    int reduce_size = 16;
    half* in;
    half* out;

    hipMalloc((void**)&in, in_size * sizeof(half));
    hipMalloc((void**)&out, out_size * sizeof(half));

    unsigned max_threads = 64;
    unsigned blocks = 64;

    for (int threads = 1; threads <= max_threads; threads *= 2) {
        for (int i = 0; i < in_size; i++) in[i] = 1;
        for (int i = 0; i < out_size; i++) out[i] = 0;

        hipLaunchKernelGGL(integral_sum, dim3(blocks), dim3(threads), 0, 0, out, in, out_size, reduce_size);
        hipStreamSynchronize(NULL);

        for (int i = 0; i < out_size; i++) {
            if (16.0 != __half2float(out[i])) {
                return -1;
            }
        }
    }
#if 0
    for (int i = 0; i < out_size; i++) {
        printf("%f ", __half2float(out[i]));
    }
#endif

    hipFree(in);
    hipFree(out);

    return 0;
}

TEST(HIPIntegrationTest, GpuIntegralSum) { EXPECT_TRUE(gpu_integral_sum() == 0); }
