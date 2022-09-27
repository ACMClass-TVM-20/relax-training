from __future__ import annotations
import numpy as np
import tvm
from tvm.ir.module import IRModule
from tvm import topi, relax, te
from tvm.script import tir as T
from tvm.script import relax as R

@tvm.script.ir_module
class Product:
    @R.function
    def main(A: Tensor((1, 128), "float32"),
             B: Tensor((128, 128), "float32")) -> Tensor(None, "float32", ndim=2):
        with R.dataflow():
            out = R.call_tir(, (A, B), (1, 128), dtype="float32")
            R.output(out)
        return out

Product.show()

# build and run
ex = relax.vm.build(Product, target="llvm")
vm = relax.VirtualMachine(ex, tvm.cpu())

A = np.random.uniform(-1.0, 1.0, (1, 128)).astype(np.float32)
B = np.random.uniform(-1.0, 1.0, (128, 128)).astype(np.float32)

TA = tvm.nd.array(A)
TB = tvm.nd.array(B)

res = vm["main"](TA, TB).numpy()
print(res)
