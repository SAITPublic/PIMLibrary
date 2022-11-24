#include <boost/program_options.hpp>
#include "elt_perf.h"
#include "gemm_perf.h"
#include "parser.h"

using namespace boost::program_options;

int main(int argc, char* argv[])
{
    PerformanceAnalyser* analyser = NULL;
    int ret = 1;
    Parser* parser = new Parser();
    variables_map vm = parser->parse_args(argc, argv);

    if (vm.count("help")) 
    {
      ret = parser->print_help();
      return ret;
    }
    if (vm.count("operation")) {
        string op = vm["operation"].as<string>();
        if (op == "add" || op == "mul") {
            analyser = new PimEltTestFixture();
        } else if (op == "gemm") {
            analyser = new PimGemmTestFixture();
        } else if (op == "relu") {
            analyser = new PimReluTestFixture();
        } else {
            DLOG(ERROR) << "Pim doesnt support provided operation\n";
            parser->print_help();
            return -1;
        }
    }
    if (analyser != NULL) {
        ret = analyser->SetUp(parser);
        if (ret == 0) {
            int success = analyser->ExecuteTest();
            analyser->print_analytical_data();
        }
    }
    else{
      parser->print_help();
      return -1;
    }

    delete analyser;
    delete parser;
    return 1;
}
