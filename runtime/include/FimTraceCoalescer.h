#ifndef _FIM_TRACE_COALESCER_H_
#define _FIM_TRACE_COALESCER_H_

#include <string.h>
#include <iostream>
#include <sstream>
#include <vector>

namespace fim
{
namespace runtime
{

// TODO: Define Stride for each FIM configuration
#define TRANS_SIZE 0x10

enum TraceType { MemRead, MemWrite, Barrier };
typedef int64_t DATA;

struct Cmd {
    TraceType type_;
    DATA data_;
    char write_data_[65];
    Cmd(TraceType t, DATA d, char *w_data)
    {
        type_ = t;
        data_ = d;
        if (w_data != NULL) {
            int i = 0;
            for (; w_data[i] != '\0'; i++) {
                write_data_[i] = w_data[i];
            }
            write_data_[i] = '\0';
        }
    }
};

class TraceParser
{
   public:
    void parse();
    void coalesce_traces();
    DATA hex_to_int(char *str);
    void print_hex_base(DATA addr);

  private:
    std::vector<Cmd> cur_vec_;
};

} /* namespace runtime */
} /* namespace fim */

#endif
