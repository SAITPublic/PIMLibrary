#include <iostream>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;  // NOLINT(build/namespaces)

// Todo: How to create new ShapeHandle and pass to c->set_output()
REGISTER_OP("MiopenEltwise")
    .Input("first: float16")
    .Input("second: float16")
    .Input("third: int32")
    .Output("ans: float16")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        // shape_inference::ShapeHandle out_shape;
        // out_shape = c->MakeShape({1});
        // TF_RETURN_IF_ERROR(c->Subshape(c->input(0),0 ,0 , &out_shape));
        // std::cout << "out shape " << out_shape(0) << std::endl;
        // c->set_output(0,c->UnknownShape());
        // c->set_output(0,out_shape);
        // c->set_output(0, c->input(0));
        return Status::OK();
    });
