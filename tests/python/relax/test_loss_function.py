# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import pytest
import tvm
from tvm import relax
from tvm.ir.base import assert_structural_equal
from tvm.script.parser import ir as I, relax as R, tir as T
from tvm.relax.training.loss_function import *


def test_softmax_cross_entropy_loss():
    @R.function
    def expected(
        inputs: R.Tensor((1, 10), dtype="float32"), targets: R.Tensor((1, 10), dtype="float32")
    ) -> R.Tensor(None, dtype="float32", ndim=0):
        R.func_attr({"global_symbol": "softmax_cross_entropy"})
        with R.dataflow():
            gv: R.Tensor((), dtype="float32") = R.nn.softmax_cross_entropy(inputs, targets)
            R.output(gv)
        return gv

    loss_func = softmax_cross_entropy_loss(
        shape_annotation=(1, 10), type_annotation=relax.DynTensorType(2, "float32")
    )
    assert_structural_equal(expected, loss_func)

    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor((3, 10), "float32"),
            w0: R.Tensor((10, 5), "float32"),
            b0: R.Tensor((5,), "float32"),
        ):
            with R.dataflow():
                lv0 = R.nn.matmul(x, w0)
                out = R.add(lv0, b0)
                R.output(out)
            return out

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((3, 10), dtype="float32"),
            w0: R.Tensor((10, 5), dtype="float32"),
            b0: R.Tensor((5,), dtype="float32"),
            label: R.Tensor((3, 5), dtype="float32"),
        ) -> R.Tensor(None, dtype="float32", ndim=0):
            # block 0
            with R.dataflow():
                lv0: R.Tensor((3, 5), dtype="float32") = R.nn.matmul(x, w0, out_dtype="")
                out: R.Tensor((3, 5), dtype="float32") = R.add(lv0, b0)
                loss: R.Tensor((), dtype="float32") = R.nn.softmax_cross_entropy(out, label)
                R.output(loss)
            return loss

    loss_func1 = softmax_cross_entropy_loss(
        shape_annotation=(3, 5), type_annotation=relax.DynTensorType(2, "float32")
    )
    After = tvm.relax.transform.ExtendFunc(Before.get_global_var("main"), loss_func1)(Before)
    assert_structural_equal(After["main_softmax_cross_entropy"], Expected["main"])


if __name__ == "__main__":
    pytest.main([__file__])
