#ifndef _UTILS_H_
#define _UTILS_H_

#include <iostream>

inline void print_help()
{
    std::cout << "PimBenchmark tool usage : ./pimbench <args>\n";
    std::cout << "args : \n";
    std::cout << "-device : sets the device for PIM execution in multi gpu scenario (default : 0)\n";
    std::cout << "-plt (hip / opencl) : set the platform for PIM execution (default : hip)\n";
    std::cout << "-pscn (fp16) : sets the precision for PIM operations (default : fp16)\n";
    std::cout << "-op (add / mul / relu / gemm) : indicates which operation is to be run on PIM\n";
    std::cout << "-n : set the number of batch dimension. \n";
    std::cout << "-c : sets the number of channel dimension.\n";
    std::cout << "-i_h : sets the input height dimension.\n";
    std::cout << "i_w : sets the input width.\n";
    std::cout << "o_h : sets the output height.\n";
    std::cout << "o_w : sets the output width.\n";
    std::cout << "-order(i_x_w / w_x_i) : sets the order for gemm operation. \n";
    std::cout << "-act (relu / none) : set the activation function for gemm operation (default : relu).\n";
    std::cout << "-bias (0 / 1) : sets if the gemm operation has bias addition (default : 1).\n";
    std::cout << "-i : sets the number of iterations to be executed for a particualr operation. (default : 2)\n";
}

#endif
