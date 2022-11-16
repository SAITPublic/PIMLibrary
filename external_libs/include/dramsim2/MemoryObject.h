/***************************************************************************************************
* Copyright (C) 2021 Samsung Electronics Co. LTD
*
* This software is a property of Samsung Electronics.
* No part of this software, either material or conceptual may be copied or distributed, transmitted,
* transcribed, stored in a retrieval system, or translated into any human or computer language in
* any form by any means,electronic, mechanical, manual or otherwise, or disclosed
* to third parties without the express written permission of Samsung Electronics.
* (Use of the Software is restricted to non-commercial, personal or academic, research purpose only)
***************************************************************************************************/

#ifndef MEMORYOBJECT_H
#define MEMORYOBJECT_H

#include <stdint.h>
#include <string>
#include "Transaction.h"
#include "SimulatorObject.h"

namespace DRAMSim
{

class MemoryObject : public SimulatorObject
{
  public:
    MemoryObject() {}
    virtual ~MemoryObject() {}
    virtual bool addTransaction(Transaction* trans) = 0;
    virtual bool addTransaction(bool isWrite, uint64_t addr, BurstType* data) = 0;
    virtual bool addTransaction(bool isWrite, uint64_t addr, const std::string& tag,
                                BurstType* data) = 0;
};

} // namespace DRAMSim

#endif
