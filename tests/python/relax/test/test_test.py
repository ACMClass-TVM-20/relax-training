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


from __future__ import annotations
import numpy as np
import tvm
from tvm.ir.module import IRModule
from tvm import topi, relax, te
from tvm.script import tir as T
from tvm.script import relax as R
from tvm.relax.testing import dump_ast
import tvm.script
import _gradient

@tvm.script.ir_module
class Before:
    @R.function
    def main(x: Tuple(Tensor((10, 5), "float32"), Tensor((10, 5), "float32")),
                y: Tensor((10, 5), "float32")):
        with R.dataflow():
            z0 = (x, (x, x))
            z1 = z0[1]
            z2 = z1[0]
            z3 = z2[1]
            z4 = relax.add(z3, y)
            z10 = relax.Tuple((z3, y))
            z5 = relax.TupleGetItem(z10, 1)
            z6 = relax.add(z5, z4)
            z7 = relax.TupleGetItem(x, 0)
            z8 = relax.add(z7, z6)
            z9 = relax.sum(z8)
            R.output(z9)
        return z9

Before.show()
