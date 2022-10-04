from __future__ import annotations
import numpy as np
import tvm
from tvm.ir.module import IRModule
from tvm import topi, relax, te
from tvm.script import tir as T
from tvm.script import relax as R


def nn_gradrelu_(x):
    return (x > 0)

print("Build mlp")

from utils import LowerToTensorIRPass

print("TVM version: ", tvm.__version__)

"""
    model
"""

@tvm.script.ir_module
class TestPlus:
    @R.function
    def main(x: Tensor((3, 3), "float32"), y: Tensor((3, 3), "float32")) -> Tensor(None, "float32", ndim=2):
        
        # block 0
        with R.dataflow():
            # linear0
            out = relax.add(x, y)
            R.output(out)
        return out

def run_binary(module):
    TIRModule = LowerToTensorIRPass()(module)
    TIRModule.show()

    # build and run
    ex = relax.vm.build(TIRModule, target="llvm")
    vm = relax.VirtualMachine(ex, tvm.cpu())

    input0 = np.random.randn(3, 3).astype(np.float32)
    input1 = np.random.randn(3, 3).astype(np.float32)

    print("input0: ", input0)
    print("intpu1: ", input1)
    output = vm["main"](tvm.nd.array(input0), tvm.nd.array(input1))
    for i in range(len(output)):
        print("output{}: ".format(i), output[i])

run_binary(relax.transform.SimpleAD()(TestPlus))