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


from __future__ import annotations  # must import to defer parsing of annotations
import pytest
import tvm
import tvm.script
from tvm import topi, relax, te
from tvm.script import tir as T
from tvm.script import relax as R
import _gradient
from tvm.ir import structural_equal
from tvm.ir.base import assert_structural_equal

def execute_mod(mod, func_name, *args):
    ex = relax.vm.build(TIRModule, target="llvm")
    vm = relax.VirtualMachine(ex, tvm.cpu())


def test_mlp_script():
    @tvm.script.ir_module
    class Before:
        @R.function
        def main(x: Tensor((1, 784), "float32"), # x shall be 2d tensor due to restriction of matmul
                 w0: Tensor((784, 10), "float32"),
                 b0: Tensor((10,), "float32"),
                 label: Tensor((1, 10), "float32")):
            with R.dataflow():
                lv0 = relax.nn.matmul(x, w0)
                out = relax.add(lv0, b0)
                loss = relax.nn.softmax_cross_entropy(out, label)
                R.output(out, loss)
            return out, loss

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: Tensor((1, 784), "float32"),
                 w0: Tensor((784, 10), "float32"),
                 b0: Tensor((10,), "float32"),
                 label: Tensor((1, 10), "float32")):
            # block 0
            with R.dataflow():
                lv0 = relax.nn.matmul(x, w0)
                out = relax.add(lv0, b0)
                loss = relax.nn.softmax_cross_entropy(out, label)
                loss_adjoint = relax.ones_like(loss)
                lv = relax.nn.softmax(out)
                lv1 = relax.sub(lv, label)
                out_adjoint = relax.multiply(loss_adjoint, lv1)
                lv0_adjoint = relax.collapse_sum_like(out_adjoint, lv0)
                lv2 = relax.transpose(x)
                lv3 = relax.nn.matmul(lv2, lv0_adjoint)
                w0_adjoint = relax.collapse_sum_like(lv3, w0)
                b0_adjoint = relax.collapse_sum_like(out_adjoint, b0)
                R.output(out, loss, w0_adjoint, b0_adjoint)
            return ((out, loss), (w0_adjoint, b0_adjoint))

    After = relax.transform.SimpleAD(func_name="main", target_names="loss", require_grad_names=["w0", "b0"])(Before)
    After.show()
    print("---------------------")
    Expected.show()
    assert_structural_equal(After, Expected)


if __name__ == "__main__":
    # pytest.main([__file__])
    test_mlp_script()
