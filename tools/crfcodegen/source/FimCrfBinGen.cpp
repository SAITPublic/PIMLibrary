#include "FimCrfBinGen.h"

namespace crfgen_offline
{
FimCrfBinGen::FimCrfBinGen() {}

void FimCrfBinGen::convert_to_binary(std::vector<FimCommand>& fim_cmd_vec, uint32_t* crf_binary)
{
    //*crf_size = fim_cmd_vec.size();

    for (int i = 0; i < fim_cmd_vec.size(); i++) {
        crf_binary[i] = fim_cmd_vec[i].to_int();
    }
}

void FimCrfBinGen::convert_to_binary(FimCommand fim_cmd, uint32_t* crf_binary) { *crf_binary = fim_cmd.to_int(); }

}  // namespace crfgen_offline
