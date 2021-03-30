/*********************************************************************************
 *  Copyright (c) 2010-2011, Elliott Cooper-Balis
 *                             Paul Rosenfeld
 *                             Bruce Jacob
 *                             University of Maryland
 *                             dramninjas [at] gmail [dot] com
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright notice,
 *        this list of conditions and the following disclaimer.
 *
 *     * Redistributions in binary form must reproduce the above copyright
 *notice,
 *        this list of conditions and the following disclaimer in the
 *documentation
 *        and/or other materials provided with the distribution.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *AND
 *  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 *  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 *  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 *  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 *  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *********************************************************************************/

#ifndef __CLOCKDOMAIN__
#define __CLOCKDOMAIN__

#include <stdint.h>
#include <cmath>
#include <iostream>

namespace ClockDomain
{
template <typename ReturnT>
class CallbackBase
{
   public:
    virtual ReturnT operator()() = 0;
    virtual ~CallbackBase() {}
};

template <typename ConsumerT, typename ReturnT>
class Callback : public CallbackBase<ReturnT>
{
   private:
    typedef ReturnT (ConsumerT::*PtrMember)();

   public:
    Callback(ConsumerT* const object, PtrMember member) : object(object), member(member) {}

    Callback(const Callback<ConsumerT, ReturnT>& e) : object(e.object), member(e.member) {}

    virtual ~Callback() {}

    ReturnT operator()() { return (const_cast<ConsumerT*>(object)->*member)(); }

   private:
    ConsumerT* const object;
    const PtrMember member;
};

typedef CallbackBase<void> ClockUpdateCB;

class ClockDomainCrosser
{
   public:
    ClockUpdateCB* callback;
    uint64_t clock1, clock2;
    uint64_t counter1, counter2;
    ClockDomainCrosser(ClockUpdateCB* _callback);
    // ClockDomainCrosser(uint64_t _clock1, uint64_t _clock2, ClockUpdateCB
    // *_callback);
    // ClockDomainCrosser(double ratio, ClockUpdateCB *_callback);
    void update();
    virtual ~ClockDomainCrosser()
    {
        if (callback) {
            delete callback;
        }
    }
};

/*
    class TestObj
    {
            public:
            TestObj() {}
            void cb();
            int test();
    };
*/
}  // namespace ClockDomain

#endif
