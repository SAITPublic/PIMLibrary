#ifndef FIM_DEVICE_H_
#define FIM_DEVICE_H_

#include "fim_data_types.h"

namespace fim {
namespace runtime {
namespace manager {

class FimDevice {
public:
    FimDevice(FimPrecision precision);
    virtual ~FimDevice(void);

    int Initialize(void);
    int Deinitialize(void);
    int RequestIoctl(void);

private:
    int OpenDevice(void);
    int CloseDevice(void);

    static constexpr char fimDrvName_[] = "/dev/fim_drv";
    int fim_fd_;
    size_t grf_size_;
    size_t grf_cnt_;
    size_t channel_cnt_;
    FimPrecision precision_;
};

} /* namespace manager */
} /* namespace runtime */
} /* namespace fim */

#endif
