from __future__ import annotations

import time

import math
import numpy as np
import tvm
from tvm import relax
from tvm import relay, te, tir, topi
from tvm.ir.base import assert_structural_equal
from tvm.ir.module import IRModule
from tvm.relax import ExternFunc, ShapeExpr, Tuple
from tvm.relax.testing import dump_ast, nn
from tvm.script import relax as R
from tvm.script import tir as T
from tvm.tir.function import PrimFunc

from utils import LowerToTensorIRPass

np.random.seed(1)

def build_lstm_mod(steps_num, in_size, hidden_size, out_size, batch_size=1):
    """
        inputs: x_t, y (label), C_init, H_init, Wh_{}, Wx_{}, B_{}, Wh_q, B_q
    """
    dtype = relax.DynTensorType(dtype="float32")

    inputs_list = []
    x_list = [relax.Var("x_" + str(i), [batch_size, in_size], dtype) for i in range(steps_num)]
    inputs_list += x_list
    y = relax.Var("y", [batch_size, out_size], dtype)
    inputs_list.append(y)

    C = relax.Var("C", [batch_size, hidden_size], dtype)
    H = relax.Var("H", [batch_size, hidden_size], dtype)
    inputs_list.append(C)
    inputs_list.append(H)

    params = {}

    for suffix in ["f", "i", "c", "o"]:
        params["Wh_" + suffix] = relax.Var("Wh_" + suffix, [hidden_size, hidden_size], dtype)
        params["Wx_" + suffix] = relax.Var("Wx_" + suffix, [in_size, hidden_size], dtype)
        params["B_" + suffix]  = relax.Var("B_" + suffix, [hidden_size], dtype)
        inputs_list += [params["Wh_" + suffix], params["Wx_" + suffix], params["B_" + suffix]]
    params["Wh_q"] = relax.Var("Wh_q", [hidden_size, out_size], dtype)
    params["B_q"] = relax.Var("B_q", [out_size], dtype)
    inputs_list += [params["Wh_q"], params["B_q"]]

    assert x_list[0] == inputs_list[0]

    bb = relax.BlockBuilder()
    with bb.function("LSTM", inputs_list):
        with bb.dataflow():
            for i in range(steps_num):
                F = bb.emit(relax.op.nn.sigmoid(
                    relax.op.add(
                        relax.op.add(
                            relax.op.nn.matmul(H, params["Wh_f"]),
                            relax.op.nn.matmul(x_list[i], params["Wx_f"])
                        ),
                        params["B_f"]
                    )
                ))

                I = bb.emit(relax.op.nn.sigmoid(
                    relax.op.add(
                        relax.op.add(
                            relax.op.nn.matmul(H, params["Wh_i"]),
                            relax.op.nn.matmul(x_list[i], params["Wx_i"])
                        ),
                        params["B_i"]
                    )
                ))

                C_tilde = bb.emit(relax.op.nn.tanh(
                    relax.op.add(
                        relax.op.add(
                            relax.op.nn.matmul(H, params["Wh_c"]),
                            relax.op.nn.matmul(x_list[i], params["Wx_c"])
                        ),
                        params["B_c"]
                    )
                ))

                O = bb.emit(relax.op.nn.sigmoid(
                    relax.op.add(
                        relax.op.add(
                            relax.op.nn.matmul(H, params["Wh_o"]),
                            relax.op.nn.matmul(x_list[i], params["Wx_o"])
                        ),
                        params["B_o"]
                    )
                ))

                C = bb.emit(relax.op.add(relax.op.multiply(F, C), relax.op.multiply(I, C_tilde)))
                H = bb.emit(relax.op.multiply(O, relax.op.nn.tanh(C)))

            out = bb.emit(relax.op.add(relax.op.nn.matmul(H, params["Wh_q"]), params["B_q"]))
            loss = bb.emit_output(relax.op.nn.softmax_cross_entropy(out, y))
        bb.emit_func_output(loss)
    mod = bb.get()
    return relax.transform.SimpleAD(mod.get_global_var("LSTM"), require_grads=list(params.values()))(mod)


mod = build_lstm_mod(3, 10, 10, 20, 1)
mod.show()
