from __future__ import annotations

import time
from typing import Optional, Union

import numpy as np
import tvm
from tvm import relax
from tvm import relax as rx
from tvm import relay, te, tir, topi
from tvm.ir.base import assert_structural_equal
from tvm.ir.module import IRModule
from tvm.relax import BlockBuilder, ExternFunc, ShapeExpr, Tuple
from tvm.relax.testing import dump_ast, nn
from tvm.script import relax as R
from tvm.script import tir as T
from tvm.tir.function import PrimFunc

from utils import LowerToTensorIRPass

class Optimizer:
    def __init__(self, param_list: list[relax.Var]) -> None:
        self.args = None
        self.param_list = param_list
        pass


    def add_argument(*args: list[relax.Var]) -> None:
        """
        ??
        """
        pass
    def get_relax_function() -> relax.Function:
        BlockBuilder
        """Use blockbuilder to build a new function.

        Returns
        -------
        func: relax.Function

        # Deleted #
        # For optimizers without any internal attributes:
        # def func(params: Tuple(Tensor, ...), gradients: Tuple(Tensor, ...))
        # 	return new_params


        For optimizers with internal attributes:
        def main_adjoint():
            return ...
        def func(params: Tuple(Tensor, ...), gradients: Tuple(Tensor, ...), optimizer_args)
            # optimizer_args:
            # Tuple(lr, num_steps) for SGD
            # Tuple(lr, num_steps, v_list: Tuple, s_list: Tuple, ) adam
            return new_params, new_optimizer_args
        """

    def get_ir_module() -> IRModule:
        """
        Wrap function into a irmodule
        """
        pass

    def step():
        """
        Two problems to consider:
        - Shall we modify tvm.nd.array?
        - How to build and run?
        """
        pass

# usage
optimizer = Optimizer(None)
params, gradient = None, None
func = None
params, optimizer.args = func(params, gradient, optimizer.args)
class AdamOptimizer(Optimizer):
    def __init__(self, param_list, init_lr, beta1, beta2, eps, normalize: bool):
        pass


# ad pass:
# old func -> new func

# under package relax.transform
def gradient(func: Union[relax.Function, relax.GlobalVar],
             require_grads: Union[list[relax.Var], list[int]] = [],
             module: Optional[IRModule] = None) -> relax.Function:
    """High level reverse-mode auto-differentiation.

    It is implemented in C++ and has a python wrapper.

    Implementation details:
    There is a pass extending ExprMutator to transform the old function into a new AD one.

    Parameters
    ----------
    func: Union[relax.Function, relax.GlobalVar]
        The function or global var to differentiate.
        func should only return one value.
    require_grads: Union[list[relax.Var], list[int]]
        The relax variables which need adjoints. Must be arguments of func.
        If the list is empty, it will emit an adjoint for each input.
    module: Optional[IRModule]
        If the module is given, the returned function will be added to the module
        with name: [name of func] + "_adjoint".

    Returns
    -------
    ret: relax.Function
        The result function.


    Example
    -------

    @tvm.script.ir_module
    class Before:
        @R.function
        def main(x: Tensor((1, 20), "float32"),
                 w0: Tensor((20, 10), "float32"),
                 b0: Tensor((10,), "float32"),
                 label: Tensor((1, 10), "float32")):
            with R.dataflow():
                lv0 = relax.nn.matmul(x, w0)
                out = relax.add(lv0, b0)
                loss = relax.nn.softmax_cross_entropy(out, label)
                R.output(out, loss)
            return out

    relax.transform.gradient(Before["main"], require_grads=[1, 2], module=Before)

    # Before should be isomorphic with Expected
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
                R.output(out, loss)
            return out
        @R.function
        def main_adjoint(x: Tensor((1, 20), "float32"),
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
            return (out, (w0_adjoint, b0_adjoint))
    """
    pass


def SimpleAD(func: relax.GlobalVar, require_grads: Union[list[relax.Var], list[int]] = []) \
        -> tvm.ir.transform.Pass:
    """Simple high level reverse-mode auto-differentiation.

    Parameters
    ----------
    func: Union[relax.Function, relax.GlobalVar]
        The function to be passed.
        The function should return only one value.
    require_grads: Union[list[relax.Var], list[int]]
        The relax variables which need adjoints. Must be inputs.
        If the list is empty, it will emit an adjoint for each input.

    Returns
    -------
    ret: tvm.ir.transform.Pass
    """
    pass
