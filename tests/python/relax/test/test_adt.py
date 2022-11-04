from __future__ import annotations
import numpy as np
import tvm
from tvm.ir.module import IRModule
from tvm import topi, relax, te
from tvm.script import tir as T
from tvm.script import relax as R
from tvm.relax.testing import dump_ast
import tvm.script
from tvm.runtime.container import ADT, tuple_object
import _gradient

from tvm.ir.base import assert_structural_equal

from utils import LowerToTensorIRPass

"""
    model
"""

@tvm.script.ir_module
class Module:
    @R.function
    def main(x: Tuple(Tuple(Tensor((2, 2), "float32"), Tensor((2, 2), "float32")),
                      Tensor((2, 2), "float32")), y: Tensor((2,2), "float32")):
        # block 0
        with R.dataflow():
            # linear0
            lv0 = x[0]
            R.output(lv0)
        return lv0
Module.show()
var = Module["main"].params[0]
ex = relax.vm.build(Module, target="llvm")
vm = relax.VirtualMachine(ex, tvm.cpu())

input = tuple_object(( tuple_object(( tvm.nd.array(np.zeros((2,2))), tvm.nd.array(np.zeros((2,2))) )),
                      tvm.nd.array(np.zeros((2, 2))) ))
res = vm["main"](input)
x, y = res
print(x.numpy(), y.numpy(), sep='\n')
