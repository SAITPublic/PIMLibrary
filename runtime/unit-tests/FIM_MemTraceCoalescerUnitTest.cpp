#include <FimTraceCoalescer.h>
#include <gtest/gtest.h>


int test_fim_trace_parser()
{
    fim::runtime::TraceParser p;
    p.parse();
    p.coalesce_traces();
    return 0;
}
TEST(FIMRuntimeUnitTest, Coalesce_1Thread_1Channel) { EXPECT_TRUE(test_fim_trace_parser() == 0); }
