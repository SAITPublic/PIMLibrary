#include <iostream>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;  // NOLINT(build/namespaces)

// Todo: How to create new ShapeHandle and pass to c->set_output()
REGISTER_OP("FimEltwise")
    .Input("first: float16")
    .Input("second: float16")
    .Input("third: int32")
    .Output("ans: float16")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status::OK();
    });
