#include "iostream"
#include "pim_runtime_api.h"
#include "pim_compiler.h"

#define MAT_ROW (256)
#define MAT_COL (1024)

using namespace std;
using namespace pimc;
using namespace pimc::frontend;

int eltwise_add(){
    //Declare variables
    IndexVar i(0, MAT_ROW, "i");
    IndexVar j(0, MAT_COL, "j");
    Buffer B(MAT_ROW, MAT_COL, "B");
    Buffer C(MAT_ROW, MAT_COL, "C");
    Var D("D");

    //Define Computation
    D[i][j] = C[i][j] + B[i][j];

    //Run
    PimTarget target = PimCreateTarget();
    PimInitialize();
    PimCompileObj* obj = PimCompileCustom(D, {B, C}, target);
    PimBo* output = PimExecuteCustom(obj, target);
    PimDeinitialize();
    return 0;
}
