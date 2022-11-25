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
    """Simple wrapper for relax training.

    Examples
    --------
    Initialize it first, do some settings and then train.

    .. code-block:: python
        trainer = Trainer(backbone=MLP, func_name="main", parameters_indices=range(1, 5))
        trainer.prepare("relax.nn.softmax_cross_entropy", SGD(None, 0.01))
        trainer.set_vm_config(target="llvm", device=tvm.cpu())
        trainer.setup()
        trainer.rand_init_params()
        trainer.train(epoch=10, loader=loader, data_hook=_hook, show_detail=True)
    """

    def __init__(self, backbone: IRModule, func_name: str, parameters_indices, dtype="float32") -> None:
        """Default initializer for relax.training.Optimizer.

        Parameters
        ----------
        backbone: IRModule
            Backbone of the training module. It should be a relax module with a function
            whose name is `func_name`.

        func_name: str
            The name of the target function. The function should return the output of the module.

        parameters_indices:
            The indices of parameters in the input list of the target function.

        dtype: str
            The dtype of all data here. For simplicity, we suppose all data uses the same dtype. It should be
            a string which can be used to initialize both numpy dtype and relax DynTensorType.
            By default it is float32.
        """

        # should be config after
        if isinstance(parameters_indices, int):
            parameters_indices = [parameters_indices]
        self._parameters_indices = list(parameters_indices)
        self._loss_op = None
        self._loss_arg = None
        self._vm_config = {}

        self._optimizer = None
        self._backbone = backbone

        self._parameters_buffer = []
        self._parameters_name_to_pos = {}

        self.func_name = func_name
        self.train_func_name = ""
        self._vm = None
        self.mod = None
        self.dtype = dtype

    def prepare(self, loss_op: Union[str, Op], optimizer):
        """Specify an op and label shape. Label type will be dtype.

        Parameters
        ----------
        loss_op:
            The op of loss. Loss will be calculated using this binary operator, with two arguments be
            out of the backbone and a label. They must be of the same shape.

        optimizer:
            Specify the optimizer. It should be a 'partial initialized' optimizer with the argument
            param_list be None. You do not need to specify param_list here because the trainer will automatically
            config it.
        """
        if isinstance(loss_op, str):
            loss_op = Op.get(loss_op)
        self._loss_op = loss_op
        self._loss_arg = relax.Var("label", self._backbone[self.func_name].body.body.shape, relax.DynTensorType(dtype=self.dtype))

        self._optimizer = optimizer

    def set_vm_config(self, target, device = tvm.cpu(), memory_cfg = None):
        """Specify the following vm config: target, device, memory_cfg"""
        self._vm_config = {
            "target": target,
            "device": device,
            "memory_cfg": memory_cfg
        }

    def setup(self):
        """Setup the trainer.
               * Perfrom the follwing pass by order: AppendCall, SimpleAD, Lower.
               * Add Optimizer.
               * Allocate buffers for parameters.
        """
        loss = relax.Var("loss", [], relax.DynTensorType(dtype=self.dtype))

        try:
            assert self._loss_op is not None
            assert self._loss_arg is not None
            assert self._optimizer is not None
        except AssertionError:
            raise Exception("Trainer Error: Please call 'prepare' first before you setup")

        append_call_mod = relax.transform.AppendCall(
            func=self._backbone.get_global_var(self.func_name),
            op=self._loss_op,
            out=loss,
            args=[self._backbone[self.func_name].body.body, self._loss_arg]
        )(self._backbone)
        loss_func_name = self.func_name + "1"

        # Pass 2.
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
        """Forward. Return output in numpy.
        """
        self._check_setup()
        return self._vm[self.func_name](*self._prepare_inputs(self.func_name, inputs)).numpy()

    def backward(self, *inputs):
        """Backward. Return loss in numpy.
        """
        self._check_setup()
        loss, grads = self._vm[self.train_func_name](*self._prepare_inputs(self.train_func_name, inputs))
        assert len(grads) == len(self._parameters_buffer)
        new_params, self._optimizer.state = self._vm[self._optimizer.__class__.__name__](
            tuple_object(self._parameters_buffer), grads, self._optimizer.state
        )
        assert len(new_params) == len(self._parameters_buffer)
        self._parameters_buffer = new_params
        return loss.numpy()

    def train(self, epoch: int, loader, data_hook = lambda x: x, show_detail = False):
        """A simple wrapper for the training loop.

        Parameters
        ----------
        epoch: int
            The number of the training epochs.

        loader:
            The data loader. It should be a iterable object with the input data returned every iteration.

        data_hook:
            A hook function which takes the return value of the loader iteration as input, and return things
            that you want to feed to the module.
            It is used to preprocess the input data. By default it is an identity function.

        show_detail: boolean
            Whether to show some information about training.
        """
        self._check_setup()
        for i in range(epoch):
            loss_buffer = []
            for dataline in loader:
                loss = self.backward(*data_hook(dataline))
                loss_buffer.append(loss)
            if show_detail:
                print(f"Train Epoch #{i}, Loss = {np.mean(loss_buffer)}")

    def rand_init_params(self):
        """Randomly initialize parameters using np.random.uniform.
        """
        self._check_setup()
        self._parameters_buffer = [
            tvm.nd.array(math.sqrt(6.0 / np.sum(v.shape)) * np.random.uniform(-1.0, 1.0, v.shape).astype(np.float32)) \
                for v in self._parameters_buffer
        ]

    def load_params(self, extern_param_dict: dict):
        """Load parameters from a dict.
        The key of the dict should be the same with the parameter name in backbone.
        """
        self._check_setup()
        for key in extern_param_dict:
            self._parameters_buffer[self._parameters_name_to_pos[key]] = tvm.nd.array(extern_param_dict[key])
