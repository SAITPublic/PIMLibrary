#include <emulator/FimTraceCoalescer.h>
#include <gtest/gtest.h>

std::string input_file_name, verify_file_name;
void set_input_file(const std::string f) { input_file_name = f; }
void set_verify_file(const std::string f) { verify_file_name = f; }
int test_fim_trace_parser()
{
    fim::runtime::emulator::TraceParser p;
    p.parse(input_file_name);
    p.coalesce_traces();

    fim::runtime::emulator::TraceParser verify;
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
