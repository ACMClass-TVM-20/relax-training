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

mlp_params = {}
import math
a0 = math.sqrt(6.0 / (784 + 128))
a1 = math.sqrt(6.0 / (128+10))
mlp_params["w0"] = (a0 * np.random.uniform(-1.0, 1.0, (784, 128))).astype(np.float32)
mlp_params["w1"] = (a1 * np.random.uniform(-1.0, 1.0, (128, 10))).astype(np.float32)
mlp_params["b0"] = np.random.uniform(-1.0, 1.0, (128,)).astype(np.float32)
mlp_params["b1"] = np.random.uniform(-1.0, 1.0, (10,)).astype(np.float32)


# print("Build mlp")

from utils import LowerToTensorIRPass

# print("TVM version: ", tvm.__version__)

"""
    model
"""

@tvm.script.ir_module
class MultiLayerPerceptron:
    @R.function
    def main(x: Tensor((1, 784), "float32"),
             w0: Tensor((784, 128), "float32"),
             b0: Tensor((128,), "float32"),
             w1: Tensor((128, 10), "float32"),
             b1: Tensor((10,), "float32"),
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

# MultiLayerPerceptron.show()

# print(dump_ast(MultiLayerPerceptron["main"]))

# @tvm.script.ir_module
# class AutoDiffMLP:
#     @R.function
#     def main(x: Tensor((1, 784), "float32"),
#              w0: Tensor((784, 128), "float32"),
#              b0: Tensor((128,), "float32"),
#              w1: Tensor((128, 10), "float32"),
#              b1: Tensor((10,), "float32"),
#              label: Tensor((1,10), "float32")):

#         # block 0
#         with R.dataflow():
#             # linear0
#             lv0 = relax.nn.matmul(x, w0) # shape: (1, 128)
#             lv1 = relax.add(lv0, b0)
#             # relu0
#             lv2 = relax.nn.relu(lv1)
#             # linear1
#             lv3 = relax.nn.matmul(lv2, w1) # shape: (1, 10)
#             out = relax.add(lv3, b1)
#             # lv4 = relax.nn.softmax(out)
#             loss = relax.nn.softmax_cross_entropy(out, label)
#             # gradient

#             # crossEnt-softMax derive
#             out_adjoint = relax.sub(lv4, label) # shape: (1, 10)

#             b1_adjoint  = out_adjoint
#             lv3_adjoint = out_adjoint

#             lv2_trans = relax.transpose(lv2)  # shape: (128, 1)
#             w1_trans  = relax.transpose(w1)   # shape: (10, 128)
#             lv2_adjoint = relax.nn.matmul(lv3_adjoint, w1_trans) # shape: (1, 128)
#             w1_adjoint  = relax.nn.matmul(lv2_trans, lv3_adjoint) # shape: (128, 10)

#             lv1_gradrelu_ = relax.nn.gradrelu_(lv1) # shape: (1, 128)
#             lv1_adjoint  = relax.multiply(lv2_adjoint, lv1_gradrelu_) # shape: (1, 128)

#             b0_adjoint = lv1_adjoint
#             lv0_adjoint = lv1_adjoint

#             x_trans = relax.transpose(x) # shape: (784, 1)
#             w0_adjoint = relax.nn.matmul(x_trans, lv0_adjoint) # shape: (784, 128)

#             R.output(out, loss, w0_adjoint, b0_adjoint, w1_adjoint, b1_adjoint)
#         return out, loss, w0_adjoint, b0_adjoint, w1_adjoint, b1_adjoint
# AutoDiffMLP.show()

# print(dump_ast(AutoDiffMLP["main"]))

AutoDiffMLP = relax.transform.SimpleAD(MultiLayerPerceptron.get_global_var("main"), require_grads=[1, 2, 3, 4])(MultiLayerPerceptron)
AutoDiffMLP.show()
# assert_structural_equal(AutoDiffMLP["main_adjoint"], Expected["main_adjoint"])
TIRModule = LowerToTensorIRPass()(AutoDiffMLP)
TIRModule.show()
# print("-------------------------------")
# AutoDiffMLP.show()
# print("-------------------------------")
# MultiLayerPerceptron.show()

# # # build and run
ex = relax.vm.build(TIRModule, target="llvm")
vm = relax.VirtualMachine(ex, tvm.cpu())

"""
    train
"""

"""
    test
nd_params = {k: tvm.nd.array(v) for k, v in mlp_params.items()}
img, label = next(iter(loader))
data_nd = tvm.nd.array(img.reshape(1, 784))
output = vm["main"](data_nd, nd_params["w0"], nd_params["b0"], nd_params["w1"], nd_params["b1"])
pred_kind = np.argmax(output.numpy(), axis=1)
print("Test Predict: ", class_names[pred_kind[0]])
print("True: ", class_names[label[0]])
"""

# success, total = 0, 0
lr = 0.03
batch_size = 64
total_loss = 0
epoch = 0
gradient_dict = {}
arg_names = ["w0", "b0", "w1", "b1"]
for arg in arg_names:
    gradient_dict[arg] = 0
for img, label in loader:
    nd_params = {k: tvm.nd.array(v) for k, v in mlp_params.items()}
    data_nd = tvm.nd.array(img.reshape(1, 784))
    label_nd = tvm.nd.array(np.array([[1 if i == label[0] else 0 for i in range(10)]]).astype(np.float32))
    loss, res = vm["main_adjoint"](data_nd, nd_params["w0"], nd_params["b0"], nd_params["w1"], nd_params["b1"], label_nd)
    w0_grad, b0_grad, w1_grad, b1_grad = res
    # loss, (w0_grad, b0_grad, w1_grad, b1_grad) = vm["main_adjoint"](data_nd, nd_params["w0"], nd_params["b0"], nd_params["w1"], nd_params["b1"], label_nd)
    # pred_kind = np.argmax(output[0].numpy(), axis=1)
    # total += 1
    # if pred_kind[0] == label[0]:
    #     success += 1

    # print("label: ", label_nd)
    # print("output:", output[0])
    # print("loss:", output[1])
    # print("w0_grad", w0_grad)
    # print("b0_grad", b0_grad)
    # print("w1_grad", w1_grad)
    # print("b1_grad", b1_grad)
    # break

    epoch += 1
    total_loss += loss.numpy()
    gradient_dict["w0"] += w0_grad.numpy()
    gradient_dict["b0"] += b0_grad.numpy()
    gradient_dict["w1"] += w1_grad.numpy()
    gradient_dict["b1"] += b1_grad.numpy()

    if epoch % batch_size == 0:
        print("epoch={}, loss={}".format(epoch, total_loss))

        for arg in gradient_dict:
            mlp_params[arg] -= lr * (gradient_dict[arg] / batch_size)
            gradient_dict[arg] = 0

        total_loss = 0


# print("Prediction Rate: ", float(success)/float(total))
