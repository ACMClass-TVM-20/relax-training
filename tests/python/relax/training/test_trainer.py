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

import numpy as np
import pytest
import tvm
import tvm.script
from tvm import relax
from tvm import relax as rx
from tvm.ir.base import assert_structural_equal
from tvm.relay.testing import rand
# from tvm.script import relax as R
from tvm.testing import assert_allclose
from tvm.testing.utils import check_numerical_grads
from tvm.script._parser import ir as I, relax as R, tir as T
from tvm._ffi.base import TVMError
from tvm.ir.op import Op


def test_append_call():
    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor((1, 5), "float32"),
                 y: R.Tensor((1, 5), "float32")):
            with R.dataflow():
                lv1 = R.add(x, y)
                R.output(lv1)
            return lv1

    @I.ir_module
    class Expected:
        @R.function
        def main1(x: R.Tensor((1, 5), "float32"),
                 y: R.Tensor((1, 5), "float32"),
                 z: R.Tensor((1, 5), "float32")):
            with R.dataflow():
                lv1 = R.add(x, y)
                lv2 = R.softmax_cross_entropy(lv1, z)
                R.output(lv1, lv2)
            return lv2

    z = relax.Var("z", (1, 5), relax.DynTensorType(ndim=2, dtype="float32"))
    new_out = relax.Var("new_out", (), relax.DynTensorType(dtype="float32"))
    After = relax.transform.AppendCall(
        func=Before.get_global_var("main"),
        op=Op.get("relax.nn.softmax_cross_entropy"),
        out=new_out,
        args=[Before["main"].body.body, z]
    )(Before)
    Before.show()
    Expected.show()
    After.show()
    assert_structural_equal(Expected["main1"], After["main1"])


if __name__ == "__main__":
    pytest.main([__file__])
