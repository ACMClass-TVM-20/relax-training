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

import numpy as np

import tvm
from tvm.ir.module import IRModule
from tvm.relax.block_builder import BlockBuilder
from tvm.relax.expr import Call, Expr, Function, Tuple, TupleGetItem
from tvm.ir import Attrs
from typing import List
from tvm.relax.transform.op_legalizer import op_legalization_map
from tvm import ir, te, topi

def map_gradrelu_(bb: BlockBuilder, args: List[Expr], attrs: Attrs, output_shape: Expr):
    def _gradrelu_(x):
        return te.compute(shape=x.shape, fcompute=lambda *indices: te.if_then_else(x(*indices)>0, 1.0, 0.0), name="gradrelu_")
    return bb.call_te(_gradrelu_, args[0])

def te_cross_entropy(x, y):
    return -topi.sum(topi.log(x) * y)

def map_cross_entropy(bb: BlockBuilder, args: List[Expr], attrs: Attrs, output_shape: Expr):
    return bb.call_te(te_cross_entropy, args[0], args[1])

def map_softmax_cross_entropy(bb: BlockBuilder, args: List[Expr], attrs: Attrs, output_shape: Expr):
    func = lambda x, y: te_cross_entropy(topi.nn.softmax(x), y)
    return bb.call_te(func, args[0], args[1])

def map_sigmoid(bb: BlockBuilder, args: List[Expr], attrs: Attrs, output_shape: Expr):
    return bb.call_te(topi.sigmoid, args[0])

extra_map = {
  ir.Op.get("relax.nn.gradrelu_"): map_gradrelu_,
  ir.Op.get("relax.nn.cross_entropy"): map_cross_entropy,
  ir.Op.get("relax.nn.softmax_cross_entropy"): map_softmax_cross_entropy,
  ir.Op.get("relax.nn.sigmoid"): map_sigmoid,
}

op_legalization_map.update(extra_map)
