from __future__ import annotations

import numpy as np
import tvm
from tvm import relax, te


def te_matmul(A: te.Tensor, B: te.Tensor) -> te.Tensor:
    assert A.shape[1] == B.shape[0]
    n = A.shape[0]
    m = B.shape[1]
    k = te.reduce_axis((0, A.shape[1]), name="k")
    return te.compute((n, m), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="matmul")

A = relax.Var("A", (3, 3), relax.DynTensorType(2, "float32"))

bb = relax.BlockBuilder()

with bb.function("main", [A]):
    with bb.dataflow():
        C = bb.emit_te(te_matmul, A, A)
        D = bb.emit_te(te_matmul, C, A)
        R = bb.emit_output(D)
    L = bb.emit_te(te_matmul, C, A)
    bb.emit_func_output(L)

MyModule = bb.get()
MyModule.show()

ex = relax.vm.build(MyModule, target="llvm")
vm = relax.VirtualMachine(ex, tvm.cpu())

A_ndarray = tvm.nd.array(np.ones((3,3)).astype(np.float32))
res = vm["main"](A_ndarray)
print(res)
