#include "executor/PimCompilerDriver.h"
#include "executor/hip/PimCompilerHipDriver.hpp"

namespace pim
{
namespace runtime
{
namespace pimc_driver
{
#if PIM_COMPILER_ENABLE == 1
std::shared_ptr<Driver> Driver::create_driver(PimTarget* target)
{
    if (target->runtime == PimRuntimeType::RT_TYPE_HIP) {
        return std::make_shared<HIPDriver>();
    } else {
        // else  TODO: OCL
    }
}
#endif
}  // namespace pimc_driver
}  // namespace runtime
}  // namespace pim
