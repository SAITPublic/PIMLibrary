#include <stdio.h>
#include <random>
#include "half.hpp"

using half_float::half;
using namespace std;
#define LENGTH (16384 * 1024)

int main()
{
    int num_input = LENGTH;
    int input_size = LENGTH / 1024 * sizeof(half);

    std::random_device rd;   // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd());  // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    half* input0 = (half*)malloc(num_input * sizeof(half));
    half* output = (half*)malloc(num_input * sizeof(half));

    for (int i = 0; i < num_input; i++) {
        input0[i] = half(dis(gen));
        output[i] = ((float)input0[i]) > 0.0 ? input0[i] : 0.0;
    }

    char fileName0[50];
    char fileName1[50];
    sprintf(fileName0, "input0_%dKB.dat", input_size);
    sprintf(fileName1, "output_%dKB.dat", input_size);

    FILE *fp0, *fp1;

    if ((fp0 = fopen(fileName0, "wb")) != NULL && (fp1 = fopen(fileName1, "wb")) != NULL) {
        fwrite(input0, sizeof(half), num_input, fp0);
        fwrite(output, sizeof(half), num_input, fp1);
    }

    fclose(fp0);
    fclose(fp1);
    free(input0);
    free(output);
}
