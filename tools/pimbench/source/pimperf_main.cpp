#include "elt_perf.h"
#include "gemm_perf.h"
#include "utils.h"

int main(int argc, char* argv[])
{
    PerformanceAnalyser* analyser = NULL;
    int ret;
    for (int i = 0; i < argc; i++) {
        if (argv[i] == std::string("-help") || argv[i] == std::string("-h")) {
            print_help();
            exit(0);
        }
        if (argv[i] == std::string("-op")) {
            string op = argv[i + 1];
            if (op == "add" || op == "mul") {
                analyser = new PimEltTestFixture();
                continue;
            } else if (op == "gemm") {
                analyser = new PimGemmTestFixture();
                continue;
            } else if (op == "relu") {
                analyser = new PimReluTestFixture();
                continue;
            } else {
                DLOG(ERROR) << "Pim doesnt support provided operation\n";
                return -1;
            }
        }
    }

    if (analyser != NULL) {
        ret = analyser->SetUp(argc, argv);
        if (ret == 0) {
            int success = analyser->ExecuteTest();
            analyser->print_analytical_data();
        }
    } else {
        print_help();
    }
    delete analyser;
}
