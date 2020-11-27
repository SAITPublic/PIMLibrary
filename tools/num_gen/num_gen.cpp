#include <stdio.h>
#include <random>
#include "half.hpp"

using half_float::half;

#define LENGTH (16384 * 1024)

#if 1
int main()
{
    int num_input = LENGTH;
    int input_size = LENGTH / 1024 * sizeof(half);

    std::random_device rd;   // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd());  // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    half* input0 = (half*)malloc(num_input * sizeof(half));
    half* input1 = (half*)malloc(num_input * sizeof(half));
    half* output = (half*)malloc(num_input * sizeof(half));

    for (int i = 0; i < num_input; i++) {
        input0[i] = half(dis(gen));
        input1[i] = half(dis(gen));
        output[i] = input0[i] + input1[i];
    }

    char fileName0[50];
    char fileName1[50];
    char fileName2[50];
    sprintf(fileName0, "input0_%dKB.dat", input_size);
    sprintf(fileName1, "input1_%dKB.dat", input_size);
    sprintf(fileName2, "output_%dKB.dat", input_size);

    FILE *fp0, *fp1, *fp2;

    if ((fp0 = fopen(fileName0, "wb")) != NULL && (fp1 = fopen(fileName1, "wb")) != NULL &&
        (fp2 = fopen(fileName2, "wb")) != NULL) {
        fwrite(input0, sizeof(half), num_input, fp0);
        fwrite(input1, sizeof(half), num_input, fp1);
        fwrite(output, sizeof(half), num_input, fp2);
    }

    fclose(fp0);
    fclose(fp1);
    fclose(fp2);
    free(input0);
    free(input1);
    free(output);
}
#endif

#if 0
int main()
{
    int num_input = LENGTH;
    int input_size = LENGTH / 1024 * sizeof(float);

    std::random_device rd;   // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd());  // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    float* input0 = (float*)malloc(num_input * sizeof(float));
    float* input1 = (float*)malloc(num_input * sizeof(float));
    float* output = (float*)malloc(num_input * sizeof(float));

    for (int i = 0; i < num_input; i++) {
        input0[i] = dis(gen);
        input1[i] = dis(gen);
        output[i] = input0[i] + input1[i];
    }

    char fileName0[50];
    char fileName1[50];
    char fileName2[50];
    sprintf(fileName0, "input0_%dKB.dat", input_size);
    sprintf(fileName1, "input1_%dKB.dat", input_size);
    sprintf(fileName2, "output_%dKB.dat", input_size);

    FILE *fp0, *fp1, *fp2;

    if ((fp0 = fopen(fileName0, "wb")) != NULL && (fp1 = fopen(fileName1, "wb")) != NULL && (fp2 = fopen(fileName2, "wb")) != NULL) {
        fwrite(input0, sizeof(float), num_input, fp0);
        fwrite(input1, sizeof(float), num_input, fp1);
        fwrite(output, sizeof(float), num_input, fp2);
    }

    fclose(fp0);
    fclose(fp1);
    fclose(fp2);
    free(input0);
    free(input1);
    free(output);
}
#endif
