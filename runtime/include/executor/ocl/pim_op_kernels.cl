
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void pim_add_test(__global half* a,
                           __global half* b,
                           __global half* output,
                           const unsigned int n)
{
   int gid = get_global_id(0);
   if (gid < n)
       output[gid] = a[gid] + b[gid];
}

