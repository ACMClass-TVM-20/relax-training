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
"""Some loss functions in the form of relax.Function."""

from typing import Any, List, Optional, Union
import typing
from ..expr import Expr, Type
from tvm.script.parser import relax as R
from tvm import relax


def softmax_cross_entropy_loss(
    shape_annotation: Optional[Union[List[Any], typing.Tuple[Any, ...]]],
    type_annotation: Optional[Type],
):
    """Binary softmax cross entropy loss."""

    bb = relax.BlockBuilder()

    inputs = relax.Var("inputs", shape_annotation, type_annotation)
    targets = relax.Var("targets", shape_annotation, type_annotation)

    func_name = "softmax_cross_entropy"

    with bb.function(func_name, [inputs, targets]):
        with bb.dataflow():
            loss = bb.emit_output(R.nn.softmax_cross_entropy(inputs, targets))
        bb.emit_func_output(loss)

    return bb.get()[func_name].with_attr("global_symbol", func_name)


def mse_loss(reduction: str):
    """MSE Loss.

    Parameters
    ----------
    reduction: str
        Specifies the reduction to apply to the output.
        sum: the output will be summed.
        mean: the sum of the output will be divided by the number of elements in the output.
    """
    ...
