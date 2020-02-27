#include "emulator/FimTraceCoalescer.h"

namespace fim
{
namespace runtime
{
namespace emulator
{
void TraceParser::parse(std::string file_name)
{
    FILE *file_ptr = fopen(file_name.c_str(), "r");
    if (file_ptr == NULL) {
        std::cout << "Trace file" << file_name << " not found\n";
        exit(1);
    }
    char type_str[10];
    // TODO: bar_id for each channel
    int bar_id = 0;
    char addr[33], write_data[65];
    DATA int_addr;
    while (fscanf(file_ptr, "%s", type_str) > 0) {
        if (!strcmp(type_str, "BAR")) {
            cur_vec_.push_back(Cmd(Barrier, bar_id, NULL));
        } else if (!strcmp(type_str, "READ")) {
            fscanf(file_ptr, "%s", addr);
            int_addr = hex_to_int(addr);
            cur_vec_.push_back(Cmd(MemRead, int_addr, NULL));
        } else if (!strcmp(type_str, "WRITE")) {
            fscanf(file_ptr, "%s", addr);
            int_addr = hex_to_int(addr);
            fscanf(file_ptr, "%s", write_data);
            cur_vec_.push_back(Cmd(MemWrite, int_addr, write_data));
        }
    }
}

void TraceParser::append_data(uint8_t *dst, uint8_t *src, int size)
{
    memcpy(dst, src, size);
}

void TraceParser::coalesce_trace(FimMemTraceData *fmtd32, FimMemTraceData *fmtd16, int fmtd16_size)
{
    uint64_t prev_addr = -1;
    char prev_cmd = 'B';
    int coalesced_trace_it = 0;
    for (int trace_it = 0; trace_it < fmtd16_size; trace_it++) {
        switch (fmtd16[trace_it].cmd) {
            case 'B':
                std::cout << "BAR " << fmtd16[trace_it].block_id << "\n";
                fmtd32[coalesced_trace_it].cmd = 'B';
                fmtd32[coalesced_trace_it].block_id = fmtd16[trace_it].block_id;
                fmtd32[coalesced_trace_it].thread_id = fmtd16[trace_it].thread_id;
                fmtd32[coalesced_trace_it].addr = 0;
                coalesced_trace_it++;
                prev_addr = -1;
                prev_cmd = 'B';
                break;
            case 'R':
                if (prev_addr != -1 && prev_cmd == 'R') {
                    if ((prev_addr + TRANS_SIZE) == fmtd16[trace_it].addr) {
                        ;
                    } else {
                        std::cout << "READ  ";
                        print_hex_base(fmtd16[trace_it].addr);
                        std::cout << "\n";
                        fmtd32[coalesced_trace_it].cmd = 'R';
                        fmtd32[coalesced_trace_it].block_id = fmtd16[trace_it].block_id;
                        fmtd32[coalesced_trace_it].thread_id = fmtd16[trace_it].thread_id;
                        fmtd32[coalesced_trace_it].addr = fmtd16[trace_it].addr;
                        coalesced_trace_it++;
                        prev_addr = fmtd16[trace_it].addr;
                    }
                } else {
                    std::cout << "READ  ";
                    print_hex_base(fmtd16[trace_it].addr);
                    std::cout << "\n";
                    fmtd32[coalesced_trace_it].cmd = 'R';
                    fmtd32[coalesced_trace_it].block_id = fmtd16[trace_it].block_id;
                    fmtd32[coalesced_trace_it].thread_id = fmtd16[trace_it].thread_id;
                    fmtd32[coalesced_trace_it].addr = fmtd16[trace_it].addr;
                    coalesced_trace_it++;
                    prev_addr = fmtd16[trace_it].addr;
                }
                prev_cmd = 'R';
                break;
            case 'W':
                if (prev_addr != -1 && prev_cmd == 'W') {
                    if ((prev_addr + TRANS_SIZE) == fmtd16[trace_it].addr) {
                        for (int i = 0; i < 16; i++)
                            std::cout << fmtd16[trace_it].data[i];
                        std::cout << "\n";
                        append_data(fmtd32[coalesced_trace_it].data + 16, fmtd16[trace_it].data, 16);
                    } else {
                        std::cout << "WRITE  ";
                        print_hex_base(fmtd16[trace_it].addr);
                        std::cout << " ";
                        for (int i = 0; i < 16; i++)
                            std::cout << fmtd16[trace_it].data[i];

                        fmtd32[coalesced_trace_it].cmd = 'W';
                        fmtd32[coalesced_trace_it].block_id = fmtd16[trace_it].block_id;
                        fmtd32[coalesced_trace_it].thread_id = fmtd16[trace_it].thread_id;
                        fmtd32[coalesced_trace_it].addr = fmtd16[trace_it].addr;
                        memcpy(fmtd32[coalesced_trace_it].data, fmtd16[trace_it].data, 16);
                        coalesced_trace_it++;
                        prev_addr = fmtd16[trace_it].addr;
                    }
                } else {
                    std::cout << "WRITE  ";
                    print_hex_base(fmtd16[trace_it].addr);
                    std::cout << " ";
                    for (int i = 0; i < 16; i++)
                        std::cout << fmtd16[trace_it].data[i];

                    fmtd32[coalesced_trace_it].cmd = 'W';
                    fmtd32[coalesced_trace_it].block_id = fmtd16[trace_it].block_id;
                    fmtd32[coalesced_trace_it].thread_id = fmtd16[trace_it].thread_id;
                    fmtd32[coalesced_trace_it].addr = fmtd16[trace_it].addr;
                    memcpy(fmtd32[coalesced_trace_it].data, fmtd16[trace_it].data, 16);
                    coalesced_trace_it++;
                    prev_addr = fmtd16[trace_it].addr;
                }
                prev_cmd = 'W';
                break;
        }
    }
}

void TraceParser::coalesce_traces()
{
    coalesced_mem_trace_.clear();
    DATA prev_addr;
    TraceType prev_cmd = Barrier;
    for (int trace_it = 0; trace_it < (int)cur_vec_.size(); trace_it++) {
        switch (cur_vec_[trace_it].type_) {
            case Barrier:
                std::cout << "BAR " << cur_vec_[trace_it].data_ << "\n";
                coalesced_mem_trace_.push_back(Cmd(cur_vec_[trace_it]));
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
                        coalesced_mem_trace_.push_back(Cmd(cur_vec_[trace_it]));
                        prev_addr = cur_vec_[trace_it].data_;
                    }
                } else {
                    std::cout << "READ  ";
                    print_hex_base(cur_vec_[trace_it].data_);
                    std::cout << "\n";
                    coalesced_mem_trace_.push_back(Cmd(cur_vec_[trace_it]));
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
                        coalesced_mem_trace_.back().append_data(cur_vec_[trace_it].write_data_);
                    } else {
                        std::cout << "WRITE  ";
                        print_hex_base(cur_vec_[trace_it].data_);
                        std::cout << " ";
                        // TODO: Which data to print?
                        for (int i = 0; cur_vec_[trace_it].write_data_[i] != '\0'; i++)
                            std::cout << cur_vec_[trace_it].write_data_[i];

                        coalesced_mem_trace_.push_back(Cmd(cur_vec_[trace_it]));
                        prev_addr = cur_vec_[trace_it].data_;
                    }
                } else {
                    std::cout << "WRITE  ";
                    print_hex_base(cur_vec_[trace_it].data_);
                    std::cout << " ";
                    // TODO: Which data to print?
                    for (int i = 0; cur_vec_[trace_it].write_data_[i] != '\0'; i++)
                        std::cout << cur_vec_[trace_it].write_data_[i];

                    coalesced_mem_trace_.push_back(Cmd(cur_vec_[trace_it]));
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

std::vector<Cmd> &TraceParser::get_trace_data() { return cur_vec_; }

bool TraceParser::verify_coalesced_trace(std::vector<Cmd> verified_trace)
{
    if (coalesced_mem_trace_.size() != verified_trace.size()) {
        std::cout << "Size mismatch\n";
        return false;
    }

    for (int i = 0; i < (int)coalesced_mem_trace_.size(); i++) {
        if (coalesced_mem_trace_[i].type_ != verified_trace[i].type_) {
            std::cout << "Trace file Cmd  mismatched at point. Expected " << verified_trace[i].type_ << " got "
                      << coalesced_mem_trace_[i].type_ << "\n";
            return false;
        }
        if (coalesced_mem_trace_[i].type_ == Barrier) {
            if (coalesced_mem_trace_[i].data_ != verified_trace[i].data_) {
                std::cout << "Trace file Barrier Channel mismatched at point. Expected " << verified_trace[i].data_
                          << " got " << coalesced_mem_trace_[i].data_ << "\n";
                return false;
            }
        } else {
            if (coalesced_mem_trace_[i].data_ != verified_trace[i].data_) {
                std::cout << "Trace file MemRead/Write Address mismatched at point. Expected "
                          << verified_trace[i].data_ << " got " << coalesced_mem_trace_[i].data_ << "\n";
                return false;
            }
            if (coalesced_mem_trace_[i].type_ == MemWrite) {
                for (int dataIt = 0; coalesced_mem_trace_[i].write_data_[dataIt] != '\0'; dataIt++) {
                    if (coalesced_mem_trace_[i].write_data_[dataIt] != verified_trace[i].write_data_[dataIt]) {
                        std::cout << "Trace file MemWrite data mismatched at point. Expected "
                                  << verified_trace[i].write_data_[dataIt] << " got "
                                  << coalesced_mem_trace_[i].write_data_[dataIt] << "\n";
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

} /* namespace emulator */
} /* namespace runtime */
} /* namespace fim */
