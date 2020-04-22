/* default access unit for dram */
#ifndef __BURST__HPP__
#define __BURST__HPP__

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>

#include "FP16.h"
#include "npy.h"

using namespace std;

namespace DRAMSim
{
union BurstType {
    BurstType()
    {
        for (int i = 0; i < 16; i++) {
            fp16_data_[i] = convert_f2h(0.0f);
        }
    }
    BurstType(float* x) { memcpy(u8_data_, x, 32); }
    BurstType(fp16* x) { memcpy(u8_data_, x, 32); }

    BurstType(float x0, float x1, float x2, float x3, float x4, float x5, float x6, float x7)
    {
        set(x0, x1, x2, x3, x4, x5, x6, x7);
    }

    BurstType(fp16 x0, fp16 x1, fp16 x2, fp16 x3, fp16 x4, fp16 x5, fp16 x6, fp16 x7, fp16 x8, fp16 x9, fp16 x10,
              fp16 x11, fp16 x12, fp16 x13, fp16 x14, fp16 x15)
    {
        set(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15);
    }

    BurstType(uint32_t x0, uint32_t x1, uint32_t x2, uint32_t x3, uint32_t x4, uint32_t x5, uint32_t x6, uint32_t x7)
    {
        u32_data_[0] = x0;
        u32_data_[1] = x1;
        u32_data_[2] = x2;
        u32_data_[3] = x3;
        u32_data_[4] = x4;
        u32_data_[5] = x5;
        u32_data_[6] = x6;
        u32_data_[7] = x7;
    }

    BurstType(uint16_t x0, uint16_t x1, uint16_t x2, uint16_t x3, uint16_t x4, uint16_t x5, uint16_t x6, uint16_t x7,
              uint16_t x8, uint16_t x9, uint16_t x10, uint16_t x11, uint16_t x12, uint16_t x13, uint16_t x14,
              uint16_t x15)
    {
        u16_data_[0] = x0;
        u16_data_[1] = x1;
        u16_data_[2] = x2;
        u16_data_[3] = x3;
        u16_data_[4] = x4;
        u16_data_[5] = x5;
        u16_data_[6] = x6;
        u16_data_[7] = x7;
        u16_data_[8] = x8;
        u16_data_[9] = x9;
        u16_data_[10] = x10;
        u16_data_[11] = x11;
        u16_data_[12] = x12;
        u16_data_[13] = x13;
        u16_data_[14] = x14;
        u16_data_[15] = x15;
    }

    void set(float x) { set(x, x, x, x, x, x, x, x); }

    void set(fp16 x) { set(x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x); }

    void set(uint32_t x) { set(x, x, x, x, x, x, x, x); }

    void set(float x0, float x1, float x2, float x3, float x4, float x5, float x6, float x7)
    {
        fp32_data_[0] = x0;
        fp32_data_[1] = x1;
        fp32_data_[2] = x2;
        fp32_data_[3] = x3;
        fp32_data_[4] = x4;
        fp32_data_[5] = x5;
        fp32_data_[6] = x6;
        fp32_data_[7] = x7;
    }

    void set(uint32_t x0, uint32_t x1, uint32_t x2, uint32_t x3, uint32_t x4, uint32_t x5, uint32_t x6, uint32_t x7)
    {
        u32_data_[0] = x0;
        u32_data_[1] = x1;
        u32_data_[2] = x2;
        u32_data_[3] = x3;
        u32_data_[4] = x4;
        u32_data_[5] = x5;
        u32_data_[6] = x6;
        u32_data_[7] = x7;
    }

    void set(fp16 x0, fp16 x1, fp16 x2, fp16 x3, fp16 x4, fp16 x5, fp16 x6, fp16 x7, fp16 x8, fp16 x9, fp16 x10,
             fp16 x11, fp16 x12, fp16 x13, fp16 x14, fp16 x15)
    {
        fp16_data_[0] = x0;
        fp16_data_[1] = x1;
        fp16_data_[2] = x2;
        fp16_data_[3] = x3;
        fp16_data_[4] = x4;
        fp16_data_[5] = x5;
        fp16_data_[6] = x6;
        fp16_data_[7] = x7;
        fp16_data_[8] = x8;
        fp16_data_[9] = x9;
        fp16_data_[10] = x10;
        fp16_data_[11] = x11;
        fp16_data_[12] = x12;
        fp16_data_[13] = x13;
        fp16_data_[14] = x14;
        fp16_data_[15] = x15;
    }

    void set(uint16_t x0, uint16_t x1, uint16_t x2, uint16_t x3, uint16_t x4, uint16_t x5, uint16_t x6, uint16_t x7,
             uint16_t x8, uint16_t x9, uint16_t x10, uint16_t x11, uint16_t x12, uint16_t x13, uint16_t x14,
             uint16_t x15)
    {
        u16_data_[0] = x0;
        u16_data_[1] = x1;
        u16_data_[2] = x2;
        u16_data_[3] = x3;
        u16_data_[4] = x4;
        u16_data_[5] = x5;
        u16_data_[6] = x6;
        u16_data_[7] = x7;
        u16_data_[8] = x8;
        u16_data_[9] = x9;
        u16_data_[10] = x10;
        u16_data_[11] = x11;
        u16_data_[12] = x12;
        u16_data_[13] = x13;
        u16_data_[14] = x14;
        u16_data_[15] = x15;
    }

    void set(BurstType& b) { memcpy(u8_data_, b.u8_data_, 32); }

    void set_random()
    {
        std::random_device rd;   // Will be used to obtain a seed for the random number engine
        std::mt19937 gen(rd());  // Standard mersenne_twister_engine seeded with rd()
        std::uniform_int_distribution<uint16_t> dis;
        for (int i = 0; i < 16; i++) {
            u16_data_[i] = dis(gen);
        }
    }

    string bin_tostr() const
    {
        stringstream ss;
        ss << "[";
        for (int i = 0; i < 16; i++) {
            ss << bitset<16>(u16_data_[i]);
        }
        ss << "]";
        return ss.str();
    }

    string hex_tostr() const
    {
        stringstream ss;
        // ss << "[";
        for (int i = 0; i < 16; i++) {
            ss << setfill('0') << setw(4) << hex << u16_data_[i];
        }
        // ss << "]" << dec;
        return ss.str();
    }

    string hex_tostr_u8() const
    {
        stringstream ss;
        // ss << "[";
        for (int i = 0; i < 32; i++) {
            ss << setfill('0') << setw(2) << hex << (int)u8_data_[i];
        }
        // ss << "]" << dec;
        return ss.str();
    }

    string hex_tostr2() const
    {
        stringstream ss;
        // ss << "[";
        for (int i = 15; i >= 0; i--) {
            ss << setfill('0') << setw(4) << hex << u16_data_[i];
        }
        // ss << "]" << dec;
        return ss.str();
    }

    string hex_tostr_reverse(int start, int end) const
    {
        stringstream ss;
        // ss << "[";
        for (int i = start; i <= end; i++) {
            ss << setfill('0') << setw(4) << hex << u16_data_[i];
        }
        // ss << "]" << dec;
        return ss.str();
    }

    string hex_tostr_reverse_u8(int start, int end) const
    {
        stringstream ss;
        // ss << "[";
        for (int i = start; i <= end; i++) {
            ss << setfill('0') << setw(2) << hex << (int)u8_data_[i];
        }
        // ss << "]" << dec;
        return ss.str();
    }

    string fp32_tostr() const
    {
        stringstream ss;
        ss << "[ ";
        for (int i = 0; i < 8; i++) {
            ss << fp32_data_[i] << " ";
        }
        ss << "]";
        return ss.str();
    }

    string fp16_tostr() const
    {
        stringstream ss;
        ss << "[ ";
        for (int i = 0; i < 16; i++) {
            ss << convert_h2f(fp16_data_[i]) << " ";
        }
        ss << "]";
        return ss.str();
    }

    bool fp16_similar(const BurstType& rhs, float epsilon)
    {
        for (int i = 0; i < 16; i++) {
            if ((convert_h2f(fp16_data_[i]) - convert_h2f(rhs.fp16_data_[i])) / convert_h2f(fp16_data_[i]) > epsilon) {
                return false;
            }
        }
        return true;
    }

    fp16 fp16_reducesum()
    {
        fp16 sum(0.0f);
        for (int i = 0; i < 16; i++) {
            sum += fp16_data_[i];
        }
        return sum;
    }

    float fp32_reducesum()
    {
        float sum = 0.0;
        for (int i = 0; i < 8; i++) {
            sum += fp32_data_[i];
        }
        return sum;
    }

    bool operator==(const BurstType& rhs) const { return !(memcmp(this, &rhs, 32)); }

    bool operator!=(const BurstType& rhs) const { return (memcmp(this, &rhs, 32)); }

    BurstType operator+(const BurstType& rhs) const
    {
        BurstType ret;
        for (int i = 0; i < 16; i++) {
            ret.fp16_data_[i] = fp16_data_[i] + rhs.fp16_data_[i];
        }
        return ret;
    }
    BurstType operator*(const BurstType& rhs) const
    {
        BurstType ret;
        for (int i = 0; i < 16; i++) {
            ret.fp16_data_[i] = fp16_data_[i] * rhs.fp16_data_[i];
        }
        return ret;
    }

    fp16 fp16_data_[16];
    uint8_t u8_data_[32];
    float fp32_data_[8];
    uint32_t u32_data_[8];
    uint16_t u16_data_[16];
};

struct NumpyBurstType {
    vector<unsigned long> shape;
    vector<float> data;
    vector<uint16_t> u16_data;
    vector<unsigned long> bshape;
    vector<BurstType> bdata;
    enum precision { FP32, FP16 };

    BurstType& get_burst(int x, int y) { return bdata[y * bshape[1] + x]; }
    BurstType& get_burst(int x) { return bdata[x]; }

    void load_fp32(string filename)
    {
        npy::LoadArrayFromNumpy(filename, shape, data);
        for (int i = 0; i < shape.size(); i++) {
            if (i == shape.size() - 1)
                bshape.push_back(ceil(shape[i] / (double)8));
            else
                bshape.push_back(shape[i]);
        }
        for (int i = 0; i < data.size(); i += 8) {
            BurstType burst(data[i], data[i + 1], data[i + 2], data[i + 3], data[i + 4], data[i + 5], data[i + 6],
                            data[i + 7]);
            bdata.push_back(burst);
        }
    }

    void load_fp16(string filename)
    {
        npy::LoadArrayFromNumpy(filename, shape, u16_data);
        for (int i = 0; i < shape.size(); i++) {
            if (i == shape.size() - 1)
                bshape.push_back(ceil(shape[i] / (double)16));
            else
                bshape.push_back(shape[i]);
        }

        for (int i = 0; i < u16_data.size(); i += 16) {
            BurstType burst((u16_data[i]), (u16_data[i + 1]), (u16_data[i + 2]), (u16_data[i + 3]), (u16_data[i + 4]),
                            (u16_data[i + 5]), (u16_data[i + 6]), (u16_data[i + 7]), (u16_data[i + 8]),
                            (u16_data[i + 9]), (u16_data[i + 10]), (u16_data[i + 11]), (u16_data[i + 12]),
                            (u16_data[i + 13]), (u16_data[i + 14]), (u16_data[i + 15]));
            bdata.push_back(burst);
        }
    }

    void load_fp16_from_fp32(string filename)
    {
        npy::LoadArrayFromNumpy(filename, shape, data);

        for (int i = 0; i < shape.size(); i++) {
            if (i == shape.size() - 1)
                bshape.push_back(ceil(shape[i] / (double)16));
            else
                bshape.push_back(shape[i]);
        }
        for (int i = 0; i < data.size(); i += 16) {
            BurstType burst(convert_f2h(data[i]), convert_f2h(data[i + 1]), convert_f2h(data[i + 2]),
                            convert_f2h(data[i + 3]), convert_f2h(data[i + 4]), convert_f2h(data[i + 5]),
                            convert_f2h(data[i + 6]), convert_f2h(data[i + 7]), convert_f2h(data[i + 8]),
                            convert_f2h(data[i + 9]), convert_f2h(data[i + 10]), convert_f2h(data[i + 11]),
                            convert_f2h(data[i + 12]), convert_f2h(data[i + 13]), convert_f2h(data[i + 14]),
                            convert_f2h(data[i + 15]));
            bdata.push_back(burst);
        }
    }

    void copy_from_buffer(uint16_t* buffer, vector<unsigned long> buf_shape)
    {
        int buf_size = 1;

        for (int i = 0; i < buf_shape.size(); i++) {
            buf_size *= buf_shape[i];
        }

        for (int i = 0; i < buf_size; i += 16) {
            BurstType burst((buffer[i]), (buffer[i + 1]), (buffer[i + 2]), (buffer[i + 3]), (buffer[i + 4]),
                            (buffer[i + 5]), (buffer[i + 6]), (buffer[i + 7]), (buffer[i + 8]), (buffer[i + 9]),
                            (buffer[i + 10]), (buffer[i + 11]), (buffer[i + 12]), (buffer[i + 13]), (buffer[i + 14]),
                            (buffer[i + 15]));
            bdata.push_back(burst);
        }

        for (int i = 0; i < buf_shape.size(); i++) {
            if (i == buf_shape.size() - 1)
                bshape.push_back(ceil(buf_shape[i] / (double)16));
            else
                bshape.push_back(buf_shape[i]);
        }
    }

    void load_fp16_from_txt(string filename, vector<unsigned long> file_shape)
    {
        ifstream file(filename.c_str());

        if (file.is_open()) {
            int i = 0;
            while (!file.eof()) {
                uint16_t tmp_data;
                file >> tmp_data;
                if (file.eof()) {
                    break;
                }
                u16_data.push_back(tmp_data);
            }
        }

        for (int i = 0; i < u16_data.size(); i += 16) {
            BurstType burst((u16_data[i]), (u16_data[i + 1]), (u16_data[i + 2]), (u16_data[i + 3]), (u16_data[i + 4]),
                            (u16_data[i + 5]), (u16_data[i + 6]), (u16_data[i + 7]), (u16_data[i + 8]),
                            (u16_data[i + 9]), (u16_data[i + 10]), (u16_data[i + 11]), (u16_data[i + 12]),
                            (u16_data[i + 13]), (u16_data[i + 14]), (u16_data[i + 15]));
            bdata.push_back(burst);
        }

        for (int i = 0; i < file_shape.size(); i++) {
            if (i == file_shape.size() - 1)
                bshape.push_back(ceil(file_shape[i] / (double)16));
            else
                bshape.push_back(file_shape[i]);
        }

        file.close();
    }

    void dump_fp16(string filename)
    {
        ofstream file(filename.c_str());
        if (file.is_open()) {
            for (int i = 0; i < bdata.size(); i++) {
                for (int j = 0; j < 16; j++) {
                    file << bdata[i].u16_data_[j] << " ";
                }
            }
        }

        file.close();
    }

    void dump_int8(string filename)
    {
        ofstream file(filename.c_str());
        if (file.is_open()) {
            for (int i = 0; i < bdata.size(); i++) {
                //  for (int j = 0; j < 16; j++) {
                //       char* temp = (char*)&(bdata[i].u16_data_[j]);
                //       file << temp[0] << temp[1];
                //  }
                for (int j = 0; j < 32; j++) {
                    file << bdata[i].u8_data_[j];
                }
            }
        }
        file.close();
    }

    void copy_burst(BurstType* b, unsigned long size)
    {
        bshape.push_back(size);

        for (int i = 0; i < size; i++) {
            BurstType burst;
            burst.set(b[i]);
            bdata.push_back(burst);
        }
    }

    unsigned long get_total_dim()
    {
        unsigned long dim = 1;
        for (int i = 0; i < bshape.size(); i++) {
            dim *= bshape[i];
        }
        return dim;
    }
};

}  // namespace DRAMSim
#endif
