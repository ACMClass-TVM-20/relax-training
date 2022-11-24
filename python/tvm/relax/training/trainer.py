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

import tvm
from tvm import relax
from tvm.ir.module import IRModule
from tvm.ir.op import Op
from typing import Union

import numpy as np
import math

from tvm.runtime.container import tuple_object
from tvm.relax.transform.op_legalizer import OperatorLegalizer

from .gradient import *

class Trainer:
    """simple wrapper for relax training."""

    def __init__(
        self,
        backbone: IRModule,
        func_name: str,
        partial_optimizer,
        dtype="float32") -> None:
        # should be config after
        self._parameters_indices = None
        self._loss_op = None
        self._loss_arg = None
        self._vm_config = {}

        self._optimizer = partial_optimizer
        self._backbone = backbone

        self._parameters_buffer = []
        self._parameters_name_to_pos = {}

        self.func_name = func_name
        self.train_func_name = ""
        self._vm = None
        self.mod = None
        self.dtype = dtype

    def add_parameters(self, indices):
        if self._parameters_indices is None:
            self._parameters_indices = []
        if isinstance(indices, int):
            indices = [indices]
        self._parameters_indices += list(indices)

    def set_parameters(self, indices):
        """Specify parameters by their indices in the train_func
        """
        self._parameters_indices = list(indices)

    def set_loss(self, loss_op: Union[str, Op], label_shape):
        """Specify an op and label shape. Label type will be dtype.
        """
        if isinstance(loss_op, str):
            loss_op = Op.get(loss_op)
        self._loss_op = loss_op
        self._loss_arg = relax.Var("label", label_shape, relax.DynTensorType(dtype=self.dtype))

    def set_vm_config(self, target, device = tvm.cpu(), memory_cfg = None):
        """Specify the following vm config: target, device, memory_cfg"""
        self._vm_config = locals()

    def setup(self):
        """Setup the trainer.
               * Perfrom the follwing pass by order: AppendCall, SimpleAD, Lower.
               * Allocate buffers for parameters.
        """
        loss = relax.Var("loss", [], relax.DynTensorType(dtype=self.dtype))

        # Pass 1.
        try:
            assert self._loss_op is not None
            assert self._loss_arg is not None
        except AssertionError:
            raise Exception("Trainer Error: Please set loss first before you setup")
        append_call_mod = relax.transform.AppendCall(
            func=self._backbone.get_global_var(self.func_name),
            op=self._loss_op,
            out=loss,
            args=[self._backbone[self.func_name].body.body, self._loss_arg]
        )(self._backbone)
        loss_func_name = self.func_name + "1"

        # Pass 2.
        try:
            assert self._parameters_indices is not None
        except AssertionError:
            raise Exception("Trainer Error: Please set parameters first before you setup")

        require_grads = [append_call_mod[loss_func_name].params[index] for index in self._parameters_indices]
        self.mod = relax.transform.SimpleAD(
            func=append_call_mod.get_global_var(loss_func_name),
            require_grads=require_grads
        )(append_call_mod)
        self.train_func_name = loss_func_name + "_adjoint"

        # Allocate Buffer for Parameters.
        def _convert_from_tvm_shape(tvm_shape):
            return [int(dim) for dim in tvm_shape]

        param_list = []
        self._parameters_buffer = []
        for i in range(len(self.mod[self.train_func_name].params)):
            if i in self._parameters_indices:
                param = self.mod[self.train_func_name].params[i]
                param_list.append(param)
                self._parameters_buffer.append(
                    tvm.nd.array(np.zeros(shape=_convert_from_tvm_shape(param.shape), dtype=np.dtype(self.dtype)))
                )
                self._parameters_name_to_pos[param.name_hint] = len(self._parameters_buffer) - 1

        # Build Optimizer
        self._optimizer.set_params(param_list)
        self._optimizer.state = None
        self.mod[self._optimizer.__class__.__name__] = self._optimizer.get_function()

        # Pass 3.
        legalizer = OperatorLegalizer(self.mod)
        lowered_mod = legalizer.transform()

        # Build VM.
        try:
            assert "target" in self._vm_config
            assert "device" in self._vm_config
            assert "memory_cfg" in self._vm_config
        except AssertionError:
            raise Exception("Trainer Error: Please set vm_config first before you setup")
        ex = relax.vm.build(lowered_mod, target=self._vm_config["target"])
        self._vm = relax.VirtualMachine(ex, device=self._vm_config["device"], memory_cfg = self._vm_config["memory_cfg"])

    def _check_setup(self):
        try:
            assert self._vm is not None
        except AssertionError:
            raise Exception("Trainer Error: Please setup first.")

    def _prepare_inputs(self, func_name, inputs):
        ptr_inputs = 0
        ptr_params = 0
        input_len = len(self.mod[func_name].params)
        assert len(inputs) + len(self._parameters_buffer) == input_len
        to_vm = []
        for i in range(input_len):
            if i in self._parameters_indices:
                to_vm.append(self._parameters_buffer[ptr_params])
                ptr_params += 1
            else:
                to_vm.append(tvm.nd.array(inputs[ptr_inputs]))
                ptr_inputs += 1
        return to_vm

    def forward(self, *inputs):
        """Forward. Return output.
        """
        self._check_setup()
        return self._vm[self.func_name](*self._prepare_inputs(self.func_name, inputs)).numpy()

    def backward(self, *inputs):
        """Backward. Return loss.
        """
        self._check_setup()
        loss, grads = self._vm[self.train_func_name](*self._prepare_inputs(self.train_func_name, inputs))
        assert len(grads) == len(self._parameters_buffer)
        tvm_params = []
        new_params, self._optimizer.state = self._vm[self._optimizer.__class__.__name__](
            tuple_object(self._parameters_buffer), grads, self._optimizer.state
        )
        assert len(new_params) == len(self._parameters_buffer)
        self._parameters_buffer = new_params
        return loss.numpy()

    def train(self, epoch, loader, data_hook = lambda x: x, show_detail = False):
        for i in range(epoch):
            loss_buffer = []
            for dataline in loader:
                loss = self.backward(*data_hook(dataline))
                loss_buffer.append(loss)
            if show_detail:
                print(f"Train Epoch #{i}, Loss = {np.mean(loss_buffer)}")

    def rand_init_params(self):
        self._parameters_buffer = [
            tvm.nd.array(math.sqrt(6.0 / np.sum(v.shape)) * np.random.uniform(-1.0, 1.0, v.shape).astype(np.float32)) \
                for v in self._parameters_buffer
        ]

    def load_params(self, extern_param_dict: dict):
        for key in extern_param_dict:
            self._parameters_buffer[self._parameters_name_to_pos[key]] = tvm.nd.array(extern_param_dict[key])
