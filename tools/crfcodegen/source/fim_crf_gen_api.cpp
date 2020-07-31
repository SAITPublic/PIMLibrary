#include "fim_crf_gen_api.h"
#include "FimCrfBinGen.h"
#include "FimValidationChecker.h"
using namespace crfgen_offline;

FimCrfBinGen fim_gen;
FimValidationChecker vc;

int ConvertToBinary(std::vector<FimCommand>& fim_cmd_vec, uint32_t* crf_binary)
{
    int ret = 0;

    if (ret = vc.validation_check(fim_cmd_vec)) {
        fim_gen.convert_to_binary(fim_cmd_vec, crf_binary);

        return ret;
    }

    return ret;
}

int ConvertToBinary(FimCommand fim_cmd, uint32_t* crf_binary)
{
    int ret = 0;

    if (ret = vc.check_validate_pair(fim_cmd)) {
        fim_gen.convert_to_binary(fim_cmd, crf_binary);
        return ret;
    }

    return ret;
}
