from __future__ import annotations  # must import to defer parsing of annotations
import pytest
import sys
import tvm
import tvm.testing
from tvm import relax
from tvm.ir.base import assert_structural_equal
import numpy as np

import tvm.script
from tvm.script import tir as T, relax as R


@tvm.script.ir_module
class Module:
    @T.prim_func
    def addone(A: T.Buffer[(16, 16), "float32"], B: T.Buffer[(16, 16), "float32"]) -> None:
        for i, j in T.grid(16, 16):
            with T.block("addone"):
                vi, vj = T.axis.remap("SS", [i, j])
                B[vi, vj] = A[vi, vj] + T.float32(1)

    @R.function
    def before(c0: Tensor((16, 16), "float32")):
        lv0 = relax.call_tir(addone, (c0,), (16, 16), dtype="float32")
        return lv0

old_mod = Module
new_mod = relax.transform.FoldConstant()(old_mod)

old_mod.show()
new_mod.show()