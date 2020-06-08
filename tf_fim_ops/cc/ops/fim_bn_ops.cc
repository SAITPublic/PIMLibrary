#include <iostream>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;  // NOLINT(build/namespaces)

REGISTER_OP("FimBn")
    .Input("first: float16")
    .Input("mean: float16")
    .Input("var: float16")
    .Input("beta: float16")
    .Input("gamma: float16")
    .Input("epsilon: double")
    .Output("ans: float16")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status::OK();
    });
