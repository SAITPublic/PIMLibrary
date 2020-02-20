#include "FimTraceCoalescer.h"

namespace fim
{
namespace runtime
{

void TraceParser::parse()
{
    char type_str[10];
    // TODO: bar_id for each channel
    int bar_id = 0;
    char addr[33], write_data[65];
    DATA int_addr;
    while (scanf("%s", type_str) > 0) {
        if (!strcmp(type_str, "BAR")) {
            cur_vec_.push_back(Cmd(Barrier, bar_id, NULL));
        } else if (!strcmp(type_str, "READ")) {
            std::cin >> addr;
            int_addr = hex_to_int(addr);
            cur_vec_.push_back(Cmd(MemRead, int_addr, NULL));
        } else if (!strcmp(type_str, "WRITE")) {
            std::cin >> std::hex >> addr;
            int_addr = hex_to_int(addr);
            scanf("%s", write_data);
            cur_vec_.push_back(Cmd(MemWrite, int_addr, write_data));
        }
    }
}

void TraceParser::coalesce_traces()
{
    DATA prev_addr;
    TraceType prev_cmd = Barrier;
    for (int trace_it = 0; trace_it < (int)cur_vec_.size(); trace_it++) {
        switch (cur_vec_[trace_it].type_) {
            case Barrier:
                std::cout << "BAR " << cur_vec_[trace_it].data_ << "\n";
                prev_addr = -1;
                prev_cmd = Barrier;
                break;
            case MemRead:
                if (prev_addr != -1 && prev_cmd == MemRead) {
                    if ((prev_addr + TRANS_SIZE) == cur_vec_[trace_it].data_) {
                        ;
                    } else {
                        std::cout << "READ  ";
                        print_hex_base(cur_vec_[trace_it].data_);
                        std::cout << "\n";
                        prev_addr = cur_vec_[trace_it].data_;
                    }
                } else {
                    std::cout << "READ  ";
                    print_hex_base(cur_vec_[trace_it].data_);
                    std::cout << "\n";
                    prev_addr = cur_vec_[trace_it].data_;
                }
                prev_cmd = MemRead;
                break;
            case MemWrite:
                if (prev_addr != -1 && prev_cmd == MemWrite) {
                    if ((prev_addr + TRANS_SIZE) == cur_vec_[trace_it].data_) {
                        // TODO: Which data to print?
                        for (int i = 0; cur_vec_[trace_it].write_data_[i] != '\0'; i++)
                            std::cout << cur_vec_[trace_it].write_data_[i];
                        std::cout << "\n";
                    } else {
                        std::cout << "WRITE  ";
                        print_hex_base(cur_vec_[trace_it].data_);
                        std::cout << " ";
                        // TODO: Which data to print?
                        for (int i = 0; cur_vec_[trace_it].write_data_[i] != '\0'; i++)
                            std::cout << cur_vec_[trace_it].write_data_[i];
                        prev_addr = cur_vec_[trace_it].data_;
                    }
                } else {
                    std::cout << "WRITE  ";
                    print_hex_base(cur_vec_[trace_it].data_);
                    std::cout << " ";
                    // TODO: Which data to print?
                    for (int i = 0; cur_vec_[trace_it].write_data_[i] != '\0'; i++) std::cout << cur_vec_[trace_it].write_data_[i];
                    prev_addr = cur_vec_[trace_it].data_;
                }
                prev_cmd = MemWrite;
                break;
        }
    }
}

DATA TraceParser::hex_to_int(char *str)
{
    if (str == NULL || str[0] == '\0' || str[1] == '\0' || str[2] == '\0') return 0;
    if (str[0] != '0' || str[1] != 'x') return 0;
    int i = 2;
    DATA ret = 0;
    while (str[i] != '\0') {
        if (str[i] >= '0' && str[i] <= '9') {
            ret = (ret * 16) + (str[i] - '0');
        } else if (str[i] >= 'a' && str[i] <= 'f') {
            ret = (ret * 16) + (str[i] - 'a' + 10);
        }
        i++;
    }
    return ret;
}

void TraceParser::print_hex_base(DATA addr)
{
    static char hex_print[16];
    std::cout << "0x";
    if (addr == 0) {
        std::cout << "0";
        return;
    }
    int reminder = 0;
    int index = 0;
    while (addr != 0) {
        reminder = addr % 16;
        addr = addr >> 4;
        if (reminder >= 10) {
            hex_print[index] = 'a' + (reminder - 10);
        } else {
            hex_print[index] = '0' + reminder;
        }
        index++;
    }
    index--;
    while (index >= 0) {
        std::cout << hex_print[index];
        index--;
    }
}

} /* namespace runtime */
} /* namespace fim */
