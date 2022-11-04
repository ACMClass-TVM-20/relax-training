from __future__ import annotations
import numpy as np
import tvm
from tvm.ir.module import IRModule
from tvm import topi, relax, te
from tvm.relax.training.optimizer import MomentumSGD
from tvm.runtime.container import tuple_object
from tvm.script import tir as T
from tvm.script import relax as R
from tvm.relax.testing import dump_ast
import tvm.script
import _gradient
from tvm.relax.training import Optimizer, SGD

from tvm.ir.base import assert_structural_equal
import torch
import torchvision
# import matplotlib.pyplot as plt
import pickle as pkl
np.random.seed(1)
test_data = torchvision.datasets.FashionMNIST(
    root="data",
    train=False,
    download=False,
    transform=torchvision.transforms.ToTensor()
)

loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print("Dataset loaded finish.")

"""
with open("fasionmnist_mlp_params.pkl", "rb") as fp:
    mlp_params = pkl.load(fp)
mlp_params["w0"] = mlp_params["w0"].T
mlp_params["w1"] = mlp_params["w1"].T
"""

import math
a0 = math.sqrt(6.0 / (784 + 128))
a1 = math.sqrt(6.0 / (128+10))
mlp_params = [
(a0 * np.random.uniform(-1.0, 1.0, (784, 128))).astype(np.float32),
np.random.uniform(-1.0, 1.0, (128,)).astype(np.float32),
(a1 * np.random.uniform(-1.0, 1.0, (128, 10))).astype(np.float32),
np.random.uniform(-1.0, 1.0, (10,)).astype(np.float32)
]


# print("Build mlp")

from utils import LowerToTensorIRPass

# print("TVM version: ", tvm.__version__)

"""
    model
"""

@tvm.script.ir_module
class MultiLayerPerceptron:
    @R.function
    def main(w0: Tensor((784, 128), "float32"),
             b0: Tensor((128,), "float32"),
             w1: Tensor((128, 10), "float32"),
             b1: Tensor((10,), "float32"),
             x: Tensor((1, 784), "float32"),
             label: Tensor((1,10), "float32")):
        # block 0
        with R.dataflow():
            # linear0
            lv0 = relax.nn.matmul(x, w0)
            lv1 = relax.add(lv0, b0)
            # relu0
            lv2 = relax.nn.relu(lv1)
            # linear1
            lv3 = relax.nn.matmul(lv2, w1)
            out = relax.add(lv3, b1)
            loss = relax.nn.softmax_cross_entropy(out, label)
            R.output(loss)
        return loss
    # @R.function
    # def SGD(params: Tuple(Tensor((784, 128), "float32"), Tensor((128,), "float32"), Tensor((128, 10), "float32"), Tensor((10,), "float32")),
    #         gradients: Tuple(Tensor((784, 128), "float32"), Tensor((128,), "float32"), Tensor((128, 10), "float32"), Tensor((10,), "float32")),
    #         optim_states: Tuple(Tensor((), "float32"))):
    #     with R.dataflow():
    #         w01: Tensor((784, 128), "float32") = params[0]
    #         b01: Tensor((128,), "float32") = params[1]
    #         w11: Tensor((128, 10), "float32") = params[2]
    #         b11: Tensor((10,), "float32") = params[3]
    #         w0_adjoint1: Tensor((784, 128), "float32") = gradients[0]
    #         b0_adjoint1: Tensor((128,), "float32") = gradients[1]
    #         w1_adjoint1: Tensor((128, 10), "float32") = gradients[2]
    #         b1_adjoint1: Tensor((10,), "float32") = gradients[3]
    #         num_steps: Tensor((), "float32") = optim_states[0]
    #         lv9: Tensor((), "float32") = relax.add(num_steps, relax.const(1.0))
    #         lv12: Tensor((784, 128), "float32") = relax.multiply(relax.const(0.03), w0_adjoint1)
    #         lv22: Tensor((784, 128), "float32") = relax.sub(w01, lv12)
    #         lv32: Tensor((128,), "float32") = relax.multiply(relax.const(0.03), b0_adjoint1)
    #         lv41: Tensor((128,), "float32") = relax.sub(b01, lv32)
    #         lv51: Tensor((128, 10), "float32") = relax.multiply(relax.const(0.03), w1_adjoint1)
    #         lv61: Tensor((128, 10), "float32") = relax.sub(w11, lv51)
    #         lv71: Tensor((10,), "float32") = relax.multiply(relax.const(0.03), b1_adjoint1)
    #         lv81: Tensor((10,), "float32") = relax.sub(b11, lv71)
    #         gv: Tuple(Tensor((784, 128), "float32"), Tensor((128,), "float32"), Tensor((128, 10), "float32"), Tensor((10,), "float32")) = (lv22, lv41, lv61, lv81)
    #         gv1: Tuple(Tensor((), "float32")) = (lv9,)
    #         R.output(gv, gv1)
    #     return (gv, gv1)



# MultiLayerPerceptron.show()

# print(dump_ast(MultiLayerPerceptron["main"]))


AutoDiffMLP = relax.transform.SimpleAD(MultiLayerPerceptron.get_global_var("main"), require_grads=[0, 1, 2, 3])(MultiLayerPerceptron)
# AutoDiffMLP.show()

param_list = AutoDiffMLP["main"].params[:-2]
lr = 0.001
opt = MomentumSGD(param_list, lr, 0.9, 0.1, 0.001, True)
AutoDiffMLP["MomentumSGD"] = opt.get_function()
AutoDiffMLP.show()

# # assert_structural_equal(AutoDiffMLP["main_adjoint"], Expected["main_adjoint"])
TIRModule = LowerToTensorIRPass()(AutoDiffMLP)
TIRModule.show()

# # build and run
ex = relax.vm.build(TIRModule, target="llvm")
vm = relax.VirtualMachine(ex, tvm.cpu())

"""
    train
"""

# success, total = 0, 0
batch_size = 64
total_loss = 0
epoch = 0
tvm_params = tuple_object([tvm.nd.array(v) for v in mlp_params])
for img, label in loader:
    data_nd = tvm.nd.array(img.reshape(1, 784))
    label_nd = tvm.nd.array(np.array([[1 if i == label[0] else 0 for i in range(10)]]).astype(np.float32))
    loss, grads = vm["main_adjoint"](*tvm_params, data_nd, label_nd)
    tvm_params, opt.state = vm["MomentumSGD"](tvm_params, grads, opt.state)

    epoch += 1
    total_loss += loss.numpy()

    if epoch % batch_size == 0:
        print("epoch={}, loss={}".format(epoch, total_loss))
        total_loss = 0
