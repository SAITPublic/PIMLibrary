#include <gtest/gtest.h>

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

    void append_data(const char *w_data)
    {
        int index = 0;
        for (; write_data_[index] != '\0' && index < 32; index++) {
            write_data_[index + 32] = write_data_[index];
        }
        write_data_[index + 32] = '\0';

        if (w_data != NULL) {
            int i = 0;
            for (; w_data[i] != '\0'; i++) {
                write_data_[i] = w_data[i];
            }
        }
    }
};

class TraceParser
{
   public:
    void parse(std::string file_name);
    void coalesce_traces();
    DATA hex_to_int(char *str);
    void print_hex_base(DATA addr);
    std::vector<Cmd> &get_trace_data();
    bool verify_coalesced_trace(std::vector<Cmd> verified_trace);

    void append_data(uint8_t *dst, uint8_t *src, int size);

   private:
    std::vector<Cmd> cur_vec_;
    std::vector<Cmd> coalesced_mem_trace_;
};

std::string input_file_name, verify_file_name;
void set_input_file(const std::string f) { input_file_name = f; }
void set_verify_file(const std::string f) { verify_file_name = f; }
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

void TraceParser::coalesce_traces()
{
    coalesced_mem_trace_.clear();
    DATA prev_addr;
    TraceType prev_cmd = Barrier;
    for (int trace_it = 0; trace_it < (int)cur_vec_.size(); trace_it++) {
        switch (cur_vec_[trace_it].type_) {
            case Barrier:
                coalesced_mem_trace_.push_back(Cmd(cur_vec_[trace_it]));
                prev_addr = -1;
                prev_cmd = Barrier;
                break;
            case MemRead:
                if (prev_addr != -1 && prev_cmd == MemRead) {
                    if ((prev_addr + TRANS_SIZE) == cur_vec_[trace_it].data_) {
                        ;
                    } else {
                        coalesced_mem_trace_.push_back(Cmd(cur_vec_[trace_it]));
                        prev_addr = cur_vec_[trace_it].data_;
                    }
                } else {
                    coalesced_mem_trace_.push_back(Cmd(cur_vec_[trace_it]));
                    prev_addr = cur_vec_[trace_it].data_;
                }
                prev_cmd = MemRead;
                break;
            case MemWrite:
                if (prev_addr != -1 && prev_cmd == MemWrite) {
                    if ((prev_addr + TRANS_SIZE) == cur_vec_[trace_it].data_) {
                        coalesced_mem_trace_.back().append_data(cur_vec_[trace_it].write_data_);
                    } else {
                        coalesced_mem_trace_.push_back(Cmd(cur_vec_[trace_it]));
                        prev_addr = cur_vec_[trace_it].data_;
                    }
                } else {
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

int test_fim_trace_parser()
{
    TraceParser p;
    p.parse(input_file_name);
    p.coalesce_traces();

    TraceParser verify;
    verify.parse(verify_file_name);

    if (!p.verify_coalesced_trace(verify.get_trace_data())) {
        std::cout << "Verification failed\n\n";
    }

    return 0;
}

TEST(FIMRuntimeUnitTest, Coalesce_1Thread_1Channel)
{
    // Path to the memory trace file
    set_input_file("../runtime/unit-tests/test-traces/16byte_1channel_input.txt");
    set_verify_file("../runtime/unit-tests/test-traces/32byte_1channel_output.txt");
    EXPECT_TRUE(test_fim_trace_parser() == 0);
}
