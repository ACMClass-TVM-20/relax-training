from __future__ import annotations
import numpy as np
import tvm
from tvm import relax
from tvm import relax as rx
from tvm import relay, te, tir, topi
from tvm.ir.base import assert_structural_equal
from tvm.ir.module import IRModule
from tvm.relax import ExternFunc, ShapeExpr, Tuple
from tvm.relax.testing import nn
from tvm.script import relax as R
from tvm.script import tir as T
from tvm.tir.function import PrimFunc
from utils import LowerToTensorIRPass

from utils import LowerToTensorIRPass

def n_layer_perceptron(layers, in_size, out_size, hidden_size):
	# m = tir.Var("m", "int64")
	# n = tir.Var("n", "int64")
	# l = tir.Var("l", "int64")
	# vec_type = rx.DynTensorType(ndim=1, dtype="float32")
	# mat_type = rx.DynTensorType(ndim=2, dtype="float32")
	vec_type = rx.DynTensorType(dtype="float32")
	mat_type = rx.DynTensorType(dtype="float32")

	input_list = [rx.Var("x", [1, in_size], mat_type)]
	w_list = [rx.Var("w_0", [in_size, hidden_size], mat_type)] + \
		[rx.Var("w_" + str(i + 1), [hidden_size, hidden_size], mat_type) for i in range(layers - 2)] + \
		[rx.Var("w_" + str(layers - 1), [hidden_size, out_size], mat_type)]
	b_list = [rx.Var("b_" + str(i), [hidden_size], vec_type) for i in range(layers - 1)] + \
		[rx.Var("b_" + str(layers - 1), [out_size], vec_type)]
	
	bb = rx.BlockBuilder()
	with bb.function("MLP", input_list + w_list + b_list):
		with bb.dataflow():
			current = input_list[0]
			for i in range(layers):
				lv0 = bb.emit(relax.op.matmul(current, w_list[i]))
				lv1 = bb.emit(relax.op.add(lv0, b_list[i]))
				current = bb.emit(relax.op.nn.relu(lv1))
			gv0 = bb.emit_output(current)
		bb.emit_func_output(gv0)
	return bb.get(), relax.transform.SimpleAD("MLP", gv0, w_list + b_list)(bb.get())

mod1, mod2 = n_layer_perceptron(3, 2, 2, 3)
mod1.show()
mod2.show()