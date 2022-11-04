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

c = relax.Constant(relax.Constant(tvm.nd.array(np.array(1.0).astype(np.float32))))

@tvm.script.ir_module
class Before:
    @R.function
    def main(x: Tensor((), "float32")):
        with R.dataflow():
            out = relax.add(x, c)
            R.output(out)
        return out
