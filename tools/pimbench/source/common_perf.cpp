#include <common_perf.h>
#include <random>
#include "pim_runtime_api.h"

PerformanceAnalyser::PerformanceAnalyser() { parser = new Parser(); }
PerformanceAnalyser::~PerformanceAnalyser() { delete parser; }
void PerformanceAnalyser::SetArgs()
{
    platform = (parser->get_platform() == "opencl") ? RT_TYPE_OPENCL : RT_TYPE_HIP;
    precision = (parser->get_precision() == "fp16") ? PIM_FP16 : PIM_INT8;
    order = (parser->get_order() == "i_x_w") ? I_X_W : W_X_I;
    operation = parser->get_operation();
    num_batch = parser->get_num_batch();
    num_channels = parser->get_num_chan();
    input_height = parser->get_inp_height();
    input_width = parser->get_inp_width();
    output_height = parser->get_out_height();
    output_width = parser->get_out_width();
    device_id = parser->get_device_id();
    num_iter = parser->get_num_iter();
}

int PerformanceAnalyser::SetUp(int argc, char* argv[])
{
    int ret = parser->parse_args(argc, argv);
    if (ret == -1) return -1;
    SetArgs();
    Tick();
    ret = PimInitialize(platform, precision);
    Tock();
    start_up_time = calculate_elapsed_time();
    ret = set_device();
    return ret;
}

void PerformanceAnalyser::TearDown(void) { PimDeinitialize(); }
void PerformanceAnalyser::Tick() { start = std::chrono::high_resolution_clock::now(); }
void PerformanceAnalyser::Tock() { end = std::chrono::high_resolution_clock::now(); }
std::chrono::duration<double> PerformanceAnalyser::calculate_elapsed_time()
{
    time_duration = end - start;
    return time_duration;
}

void PerformanceAnalyser::calculate_avg_time(){ kernel_execution_time = avg_kernel_time / (double)(num_iter); }
void PerformanceAnalyser::calculate_gflops(double flt_ops) { gflops = flt_ops / (double)kernel_execution_time.count(); }
void PerformanceAnalyser::print_analytical_data()
{
    std::cout << "Time analytics: \nPlatform: " << parser->get_platform() << std::endl;
    std::cout << "Time taken to initialize PIM : " << start_up_time.count() << " seconds\n";
    std::cout << "Time taken to execute operation : " << kernel_execution_time.count() << " seconds\n";
    std::cout << "GFlops : " << gflops << " gflops\n";
}

int PerformanceAnalyser::set_device()
{
    int ret = 0;
    if (platform == RT_TYPE_HIP) {
        ret = PimSetDevice(device_id);
    } else {
        DLOG(WARNING) << "Device cant be set for desired platform\n";
    }
    return ret;
}

int compare_data(char* data_a, char* data_b, size_t size)
{
    int ret = 0;

    for (int i = 0; i < size; i++) {
        // std::cout << (int)data_a[i] << " : " << (int)data_b[i] << std::endl;
        if (data_a[i] != data_b[i]) {
            ret = -1;
            break;
        }
    }
    return ret;
}

void set_half_data(half_float::half* buffer, half_float::half value, size_t size)
{
    for (int i = 0; i < size; i++) {
        buffer[i] = value;
    }
}

void set_rand_half_data(half_float::half* buffer, half_float::half variation, size_t size)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-variation, variation);

    for (size_t i = 0; i < size; i++) {
        buffer[i] = half_float::half(dis(gen));
    }
}

int compare_half_Ulps_and_absoulte(half_float::half data_a, half_float::half data_b, int allow_bit_cnt,
                                   float absTolerance)
{
    uint16_t Ai = *((uint16_t*)&data_a);
    uint16_t Bi = *((uint16_t*)&data_b);

    float diff = fabs((float)data_a - (float)data_b);
    int maxUlpsDiff = 1 << allow_bit_cnt;

    if (diff <= absTolerance) {
        return true;
    }

    if ((Ai & (1 << 15)) != (Bi & (1 << 15))) {
        if (Ai == Bi) return true;
        return false;
    }

    // Find the difference in ULPs.
    int ulpsDiff = abs(Ai - Bi);

    if (ulpsDiff <= maxUlpsDiff) return true;

    return false;
}

int compare_half_relative(half_float::half* data_a, half_float::half* data_b, int size, float absTolerance)
{
    int pass_cnt = 0;
    int warning_cnt = 0;
    int fail_cnt = 0;
    int ret = 0;
    std::vector<int> fail_idx;
    std::vector<float> fail_data_pim;
    std::vector<float> fail_data_goldeny;

    float max_diff = 0.0;
    int pass_bit_cnt = 4;
    int warn_bit_cnt = 8;

    for (int i = 0; i < size; i++) {
        // std::cout<<"data_a : "<<data_a[i]<<" , data_b : "<<data_b[i]<<std::endl;
        if (compare_half_Ulps_and_absoulte(data_a[i], data_b[i], pass_bit_cnt)) {
            // std::cout << "c data_a : " << (float)data_a[i] << " data_b : " << (float)data_b[i] << std::endl;
            pass_cnt++;
        } else if (compare_half_Ulps_and_absoulte(data_a[i], data_b[i], warn_bit_cnt, absTolerance)) {
            // std::cout << "w data_a : " << (float)data_a[i] << " data_b : " << (float)data_b[i] << std::endl;
            warning_cnt++;
        } else {
            if (abs(float(data_a[i]) - float(data_b[i])) > max_diff) {
                max_diff = abs(float(data_a[i]) - float(data_b[i]));
            }
            //  std::cout << "f data_a : " << (float)data_a[i] << " data_b : "  <<(float)data_b[i]  << std::endl;

            fail_idx.push_back(pass_cnt + warning_cnt + fail_cnt);
            fail_data_pim.push_back((float)data_a[i]);
            fail_data_goldeny.push_back((float)data_b[i]);

            fail_cnt++;
            ret = 1;
        }
    }

    int quasi_cnt = pass_cnt + warning_cnt;
    if (ret) {
        printf("relative - pass_cnt : %d, warning_cnt : %d, fail_cnt : %d, pass ratio : %f, max diff : %f\n", pass_cnt,
               warning_cnt, fail_cnt,
               ((float)quasi_cnt / ((float)fail_cnt + (float)warning_cnt + (float)pass_cnt) * 100), max_diff);
#ifdef DEBUG_PIM
        for (int i = 0; i < fail_idx.size(); i++) {
            std::cout << fail_idx[i] << " pim : " << fail_data_pim[i] << " golden :" << fail_data_goldeny[i]
                      << std::endl;
        }
#endif
    }
    return ret;
}

void addCPU(half_float::half* inp1, half_float::half* inp2, half_float::half* output, int length)
{
    for (int i = 0; i < length; i++) {
        output[i] = inp1[i] + inp2[i];
    }
}

void matmulCPU(half_float::half* input, half_float::half* weight, half_float::half* output, int m, int n, int k,
               half_float::half alpha, half_float::half beta)
{
    for (int mi = 0; mi < m; mi++) {
        for (int ni = 0; ni < n; ni++) {
            float temp = 0;
            for (int ki = 0; ki < k; ki++) {
                temp += (input[mi * k + ki] * weight[ki * n + ni]);
            }
            int out_idx = mi * n + ni;
            output[out_idx] = alpha * temp + beta * output[out_idx];
        }
    }
}

void addBiasCPU(half_float::half* output, half_float::half* bias, int size)
{
    for (int i = 0; i < size; i++) {
        output[i] += bias[i];
    }
}

void reluCPU(half_float::half* data, int size)
{
    for (int i = 0; i < size; i++) {
        if (data[i] < 0) data[i] = 0;
    }
}
