from __future__ import annotations
import numpy as np
import tvm
from tvm.ir.module import IRModule
from tvm import topi, relax, te
from tvm.script import tir as T
from tvm.script import relax as R


def nn_gradrelu(x):
    return (x > 0)

print("Build mlp")

from utils import LowerToTensorIRPass

print("TVM version: ", tvm.__version__)

"""
    model
"""

@tvm.script.ir_module
class TestGradRelu:
    @R.function
    def main(x: Tensor((3, 3), "float32")) -> Tensor(None, "float32", ndim=2):
        
        # block 0
        with R.dataflow():
            # linear0
            out = relax.nn.gradrelu(x)
            R.output(out)
        return out

@tvm.script.ir_module
class TestMultiply:
    @R.function
    def main(x: Tensor((3, 3), "float32"), y: Tensor((3, 3), "float32")) -> Tensor(None, "float32", ndim=2):
        
        # block 0
        with R.dataflow():
            # linear0
            out = relax.multiply(x, y)
            R.output(out)
        return out


TIRModule = LowerToTensorIRPass()(TestMultiply)
TIRModule.show()

# build and run
ex = relax.vm.build(TIRModule, target="llvm")
vm = relax.VirtualMachine(ex, tvm.cpu())

input0 = np.random.randn(3, 3).astype(np.float32)
input1 = np.random.randn(3, 3).astype(np.float32)

print("input: ", input0, input1)
output = vm["main"](tvm.nd.array(input0), tvm.nd.array(input1))
print("output: ", output)
output_numpy = np.multiply(input0, input1)
print("output_numpy: ", output_numpy)