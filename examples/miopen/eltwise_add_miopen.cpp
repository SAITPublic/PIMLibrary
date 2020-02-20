#include <miopen/miopen.h>
#include <array>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <thread>
#include <vector>

#include <utility>

void ElementWiseAdd()
{
    miopenTensorDescriptor_t a_desc, b_desc, c_desc;

    void *a_data, *b_data, *c_data;

    miopenCreateTensorDescriptor(&a_desc);
    miopenCreateTensorDescriptor(&b_desc);
    miopenCreateTensorDescriptor(&c_desc);

    std::vector<int> a_len = {2, 3, 4, 1};
    std::vector<int> b_len = {2, 3, 4, 1};
    std::vector<int> c_len = {2, 3, 4, 1};

    miopenSet4dTensorDescriptor(a_desc, miopenFloat, a_len[0], a_len[1], a_len[2], a_len[3]);
    miopenSet4dTensorDescriptor(b_desc, miopenFloat, b_len[0], b_len[1], b_len[2], b_len[3]);
    miopenSet4dTensorDescriptor(c_desc, miopenFloat, c_len[0], c_len[1], c_len[2], c_len[3]);

    float alpha_0 = 1;
    float alpha_1 = 1;
    float beta = 1;

    size_t a_sz = a_len[0] * a_len[1] * a_len[2] * a_len[3];
    size_t b_sz = b_len[0] * b_len[1] * b_len[2] * b_len[3];
    size_t c_sz = c_len[0] * c_len[1] * c_len[2] * c_len[3];

    hipMalloc(&a_data, sizeof(float) * a_sz);
    hipMalloc(&b_data, sizeof(float) * b_sz);
    hipMalloc(&c_data, sizeof(float) * c_sz);

    miopenHandle_t handle;
    hipStream_t s;
    hipStreamCreate(&s);
    miopenCreateWithStream(&handle, s);

    miopenOpTensor(handle, miopenTensorOpAdd, &alpha_0, a_desc, a_data, &alpha_1, b_desc, b_data, &beta, c_desc,
                   c_data);

    miopenDestroy(handle);
}

int main() { ElementWiseAdd(); }
