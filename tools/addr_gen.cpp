#include <iostream>

using namespace std;

typedef union addr_t {
#pragma pack(push, 1)
    struct {
        unsigned int offset:5;
        unsigned int c0:1;
        unsigned int pch:1;
        unsigned int c1:1;
        unsigned int ch:5;
        unsigned int ba0:1;
        unsigned int bg0:1;
        unsigned int bg1:1;
        unsigned int ba1:1;
        unsigned int c4to2:3;
        unsigned int ra:14;
        unsigned int sid:1;
    } format;
#pragma pack(pop);
    uint64_t value;
    void to_str(){
        cout << " sid " << format.sid << " ch "<<format.ch << " pch " << format.pch << " ba " << format.bg1*8 +format.bg0*4+format.ba1*2+format.ba0
             << " ra " << format.ra << " ca " << format.c4to2*4 + format.c1*2 + format.c0 << endl;
    }
};

uint64_t encode(int sid, int ch, int pch, int ba, int ra, int ca){
    addr_t addr;
    addr.format.sid = sid;
    addr.format.ch = ch;
    addr.format.pch = pch;
    addr.format.ra = ra;
    addr.format.bg1 = (ba&0x8)>0?1:0 ;
    addr.format.bg0 = (ba&0x4)>0?1:0 ;
    addr.format.ba1 = (ba&0x2)>0?1:0 ;
    addr.format.ba0 = (ba&0x1)>0?1:0 ;
    cout <<     addr.format.bg1<<   addr.format.bg0 <<  addr.format.ba1 <<  addr.format.ba0 << endl;
    addr.format.c4to2 = (ca >> 2)&0x7;
    addr.format.c1 = (ca >> 1)&0x1;
    addr.format.c0 = ca&0x1;
    return addr.value;

}

void decode(uint64_t value){
    addr_t addr;
    addr.value  = value;
    addr.to_str();
}

int main(){
    uint64_t addr = encode(0,4,0,0xF,0x3fff,1);
    cout << hex << addr << endl;
    decode(addr);
}
