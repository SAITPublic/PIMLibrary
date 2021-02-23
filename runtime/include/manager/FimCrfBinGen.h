#ifndef _FIM_CRF_BIN_GEN_H_
#define _FIM_CRF_BIN_GEN_H_

#include <vector>
#include "fim_data_types.h"
#include "manager/FimCommand.h"
#include "manager/FimInfo.h"

namespace fim
{
namespace runtime
{
namespace manager
{
class FimCrfBinGen
{
   public:
    FimCrfBinGen();

    void create_fim_cmd(FimOpType op_type, int lc);
    void change_to_binary(uint8_t* crf_binary, int* crf_size);
    void gen_binary_with_loop(FimOpType op_type, int lc, uint8_t* bin_buf, int* crf_sz);

   private:
    std::vector<FimCommand> cmds_;
};

} /* namespace manager */
} /* namespace runtime */
} /* namespace fim */

#endif /* _CRF_CODE_GENERATOR_H_ */
