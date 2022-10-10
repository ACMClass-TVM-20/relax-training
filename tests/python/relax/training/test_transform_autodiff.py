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
import numpy as np
import tvm
import tvm.script
from tvm import topi, relax, te
from tvm import relax as rx
from tvm.script import tir as T
from tvm.script import relax as R
import _gradient
from tvm.ir.base import assert_structural_equal
from tvm.testing.utils import check_numerical_grads
from tvm.testing import assert_allclose
from utils import LowerToTensorIRPass
from tvm.relay.testing import rand

def execute_mod(mod, func_name, *args):
    lowered_mod = LowerToTensorIRPass()(mod)
    ex = relax.vm.build(lowered_mod, target="llvm")
    vm = relax.VirtualMachine(ex, tvm.cpu())
    return vm[func_name](*args)

def check_mod_grad_equal(mod1, mod2, func_name):
    args = []
    for arg in mod1[func_name].params:
        shape = [int(l) for l in arg.shape]
        args.append(rand("float32", *shape))
    res1, grad1 = execute_mod(mod1, func_name, *args)
    res2, grad2 = execute_mod(mod2, func_name, *args)
    for (l, r) in zip(res1, res2):
        assert_allclose(l.numpy(), r.numpy())
    for (l, r) in zip(grad1, grad2):
        assert_allclose(l.numpy(), r.numpy())

def test_mlp_script():
    @tvm.script.ir_module
    class Before:
        @R.function
        # this input shall be:
        #     x: Tensor((20,), "float32"),
        #     w0: Tensor((20, 10), "float32"),
        #     b0: Tensor((10,), "float32"),
        #     label: Tensor((10,), "float32")
        # but we cannot do so due to the current restriction of matmul shape inference
        def main(x: Tensor((1, 20), "float32"),
                 w0: Tensor((20, 10), "float32"),
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
        def main(x: Tensor((1, 20), "float32"),
                 w0: Tensor((20, 10), "float32"),
                 b0: Tensor((10,), "float32"),
                 label: Tensor((1, 10), "float32")):
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

    After = relax.transform.SimpleAD(func_name="main", target="loss", require_grads=["w0", "b0"])(Before)
    assert_structural_equal(After, Expected)
    check_mod_grad_equal(Expected, After, "main")

def test_batch_mlp_script():
    @tvm.script.ir_module
    class Before:
        @R.function
        def main(x: Tensor((5, 20), "float32"), # x shall be 2d tensor due to restriction of matmul
                 w0: Tensor((20, 10), "float32"),
                 b0: Tensor((10,), "float32"),
                 label: Tensor((5, 10), "float32")):
            with R.dataflow():
                lv0 = relax.nn.matmul(x, w0)
                out = relax.add(lv0, b0)
                loss = relax.nn.softmax_cross_entropy(out, label)
                R.output(out, loss)
            return out, loss

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: Tensor((5, 20), "float32"),
                 w0: Tensor((20, 10), "float32"),
                 b0: Tensor((10,), "float32"),
                 label: Tensor((5, 10), "float32")):
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

    After = relax.transform.SimpleAD(func_name="main", target="loss", require_grads=["w0", "b0"])(Before)
    assert_structural_equal(After, Expected)
    check_mod_grad_equal(Expected, After, "main")

def test_mlp_blockbuilder():
    layers, in_size, out_size, hidden_size, batch_size = 3, 5, 5, 5, 5

    ty = rx.DynTensorType(dtype="float32")

    input_list = [rx.Var("x", [batch_size, in_size], ty)]
    w_list = [rx.Var("w_0", [in_size, hidden_size], ty)] + \
        [rx.Var("w_" + str(i + 1), [hidden_size, hidden_size], ty) for i in range(layers - 2)] + \
        [rx.Var("w_" + str(layers - 1), [hidden_size, out_size], ty)]
    b_list = [rx.Var("b_" + str(i), [hidden_size], ty) for i in range(layers - 1)] + \
        [rx.Var("b_" + str(layers - 1), [out_size], ty)]
    label_list = [rx.Var("y", [batch_size, out_size], ty)]
    args_list = input_list + w_list + b_list + label_list

    bb = rx.BlockBuilder()
    with bb.function("MLP", args_list):
        with bb.dataflow():
            current = input_list[0]
            for i in range(layers):
                lv0 = bb.emit(relax.op.matmul(current, w_list[i]))
                lv1 = bb.emit(relax.op.add(lv0, b_list[i]))
                current = bb.emit(relax.op.nn.relu(lv1) if i < layers - 1 else lv1)
            loss = bb.emit(relax.op.nn.softmax_cross_entropy(current, label_list[0]))
            gv0 = bb.emit_output(current)
            gv1 = bb.emit_output(loss)
        bb.emit_func_output((gv0, gv1))

    Before = bb.get()
    After = relax.transform.SimpleAD("MLP", gv1)(Before)

    # Check numerical gradients equal
    args = []
    for arg in After["MLP"].params[:-1]:
        shape = [int(l) for l in arg.shape]
        args.append(rand("float32", *shape))
    label = np.random.rand(batch_size, out_size).astype(np.float32)
    label /= label.sum(axis=1, keepdims=True)
    args.append(tvm.nd.array(label))

    _, grad = execute_mod(After, "MLP", *args)

    def func(*inputs):
        _, loss = execute_mod(Before, "MLP", *[tvm.nd.array(i) for i in inputs])
        return loss.numpy()
    check_numerical_grads(func, [i.numpy() for i in args], [i.numpy() for i in grad])

def test_binding_uses():
    # This case tests:
    # - Different type of bindings: assign binding & call binding;
    # - One def and multiple uses.
    # - Unused variable in module
    @tvm.script.ir_module
    class Before:
        @R.function
        def main(x: Tensor((5, 5), "float32"),
                 y: Tensor((5,), "float32"),
                 z: Tensor((5,), "float32"),
                 u: Tensor((5,), "float32")):
            with R.dataflow():
                lv1 = x
                lv2 = relax.add(x, y)
                lv3 = relax.add(lv2, y)
                lv4 = relax.add(x, lv3)
                lv5 = lv3
                lv6 = relax.add(x, lv5)
                lv7 = relax.sum(lv4)
                lv8 = relax.add(lv6, z)
                R.output(lv7)
            return lv7
    After = relax.transform.SimpleAD(func_name="main", target="lv7")(Before)
    # After.show()

    args = [rand("float32", (5, 5)), rand("float32", (5,)), rand("float32", (5,)), rand("float32", (5,))]
    output, grads = execute_mod(After, "main", *args)
    assert_allclose(output[0].numpy(), 2 * args[0].numpy() + 2 * args[1].numpy())
    expected_grads_nd = [2 * np.ones_like(args[0].numpy()),
                         10 * np.ones_like(args[1].numpy()),
                         np.zeros_like(args[2].numpy()),
                         np.zeros_like(args[3].numpy())]

    for i, j in zip(grads, expected_grads_nd):
        assert_allclose(i.numpy(), j)



# TODO:
# - [x] The transformed function should replicate all bindings from the original function and return the same values;
# - [] User can determine which inputs to differentiate or (by default) all inputs will be differentiated. Tests can cover these two cases;
# - [x] The return type & value should be correct;
# - [x] And we can also test two models: hand written one layer perceptron and hand written batch one layer perceptron.
# - [x] Besides, we have found a test function named check_numerical_grads  in python/tvm/testing/utils.py . Using that maybe we can test the MLP numerically.

if __name__ == "__main__":
    # pytest.main([__file__])
    # test_mlp_script()
    # test_batch_mlp_script()
    # test_mlp_blockbuilder()
    test_binding_uses()
