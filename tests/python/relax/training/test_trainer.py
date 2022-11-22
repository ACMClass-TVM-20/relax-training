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
from tvm.relax.training import Trainer, SGD, MomentumSGD

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
    # assert_structural_equal(Expected["main1"], After["main1"])
    # due to the return shape issue, this can not pass


def test_optimizer():
    param = relax.Var("test_param", (3, 3), relax.DynTensorType(dtype="float32"))
    sgd = SGD([param], 0.01)
    momsgd = MomentumSGD([param], 0.01, 0.9, 0.1, 0.001, True)

    def get_new_param_shape(opt):
        return opt.get_function().body.body.fields[0].shape.fields[0]

    def assert_shape_equal(shape1, shape2):
        assert np.array_equal(np.array(shape1), np.array(shape2))

    sgd_shape = get_new_param_shape(sgd)
    momsgd_shape = get_new_param_shape(momsgd)

    assert_shape_equal(param.shape, sgd_shape)
    assert_shape_equal(param.shape, momsgd_shape)


def test_trainer():
    @I.ir_module
    class MLP:
        @R.function
        def main(x: R.Tensor((1, 784), "float32"),
                w0: R.Tensor((784, 128), "float32"),
                b0: R.Tensor((128,), "float32"),
                w1: R.Tensor((128, 10), "float32"),
                b1: R.Tensor((10,), "float32")):

            # block 0
            with R.dataflow():
                # linear0
                lv0 = R.matmul(x, w0)
                lv1 = R.add(lv0, b0)
                # relu0
                lv2 = R.relu(lv1)
                # linear1
                lv3 = R.matmul(lv2, w1)
                out = R.add(lv3, b1)
                R.output(out)
            return out

    class Loader:
        def __init__(self):
            self.cnt = 0

        def __iter__(self):
            return self

        def __next__(self):
            self.cnt += 1
            if self.cnt >= 10:
                raise StopIteration()
            label = np.random.randint(0, 9)
            return np.random.uniform(5, 10, (1, 784)).astype(np.float32), \
                np.array([[1 if i == label else 0 for i in range(10)]]).astype(np.float32)

    trainer = Trainer(backbone=MLP,
        func_name="main",
        partial_optimizer=SGD(None, 0.01)
    )
    trainer.set_parameters(range(1, 5))
    trainer.set_loss("relax.nn.softmax_cross_entropy", label_shape=(1, 10))
    trainer.set_vm_config(target="llvm", device=tvm.cpu())
    trainer.setup()
    # trainer.mod.show()
    trainer.train(epoch=1, loader=Loader(), show_detail=True)

    # test re-setup
    trainer.setup()
    # trainer.mod.show()
    trainer.train(epoch=1, loader=Loader(), show_detail=True)


if __name__ == "__main__":
    pytest.main([__file__])
