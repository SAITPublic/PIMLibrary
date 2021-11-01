/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#include <iostream>
#include "PimCmd.h"

int main(int argc, char* argv[])
{
    std::cout << "crf decoder" << std::endl;

    if (argc != 2) {
        printf("./crfdecoder <hex_value>\n");
        exit(1);
    }
    unsigned int val = std::stoul(argv[1], nullptr, 16);

    crfgen_offline::PimCommand cmd;

    cmd.from_int((uint32_t)val);
    std::cout << cmd.to_str() << std::endl;

    return 0;
}
