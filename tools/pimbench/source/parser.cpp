#include "parser.h"
#include "utils.h"

int Parser::parse_args(int argc, char* argv[])
{
    if (argc < 2) {
        print_help();
        return -1;
    }
    if (argc < 6) {
        DLOG(ERROR) << "plt(hip/opencl) , op , n_b , n_c , i_h , i_w are minimum require args\n";
        return -1;
    }

    std::vector<std::string> args(argv, argv + argc);
    for (size_t i = 1; i < args.size(); ++i) {
        string option = args[i];
        if (option == "-plt") {
            platform = args[i + 1];
        } else if (option == "pcsn") {
            precision = args[i + 1];
        } else if (option == "-op") {
            operation = args[i + 1];
        } else if (option == "-order") {
            order = args[i + 1];
        } else if (option == "-n") {
            num_batch = stoi(args[i + 1]);
        } else if (option == "-c") {
            num_channels = stoi(args[i + 1]);
        } else if (option == "-i_h") {
            input_height = stoi(args[i + 1]);
        } else if (option == "-i_w") {
            input_width = stoi(args[i + 1]);
        } else if (option == "-o_h") {
            output_height = stoi(args[i + 1]);
        } else if (option == "-o_w") {
            output_width = stoi(args[i + 1]);
        } else if (option == "-bias") {
            has_bias = stoi(args[i + 1]);
        } else if (option == "-act") {
            act_function = args[i + 1];
        } else if (option == "-device") {
            device_id = stoi(args[i + 1]);
        } else if (option == "-help" or option == "-h") {
            print_help();
            return -1;
        } else if (option == "-i") {
            num_iter = stoi(args[i + 1]);
        }
    }
    if (!check_validity()) {
        DLOG(ERROR) << "Error encountered while parsing the arguments\n";
        print_help();
        return -1;
    }
    return 1;
}

bool Parser::check_validity()
{
    if (platform != "hip" && platform != "opencl") {
        std::cout << "platform : " << platform << std::endl;
        DLOG(ERROR) << "invalid platform provided\n";
        return false;
    }

    if (precision != "fp16") {
        DLOG(ERROR) << "PIM doesnt support precision compute provided\n";
        return false;
    }

    if (operation == "") {
        DLOG(ERROR) << "op (operation) argument is missing\n";
        return false;
    }

    if (num_batch == -1 || num_channels == -1 || input_height == -1 || input_width == -1) {
        DLOG(ERROR) << "input information is missing\n";
        return false;
    }

    if (operation == "gemm") {
        if (order == "") {
            DLOG(ERROR) << "compute order is missing\n";
            return false;
        }
        if (output_height == -1 || output_width == -1) {
            DLOG(ERROR) << "output information is missing\n";
            return false;
        }
    }
    return true;
}
