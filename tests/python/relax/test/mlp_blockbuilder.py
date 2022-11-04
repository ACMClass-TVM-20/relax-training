from __future__ import annotations

import time

import numpy as np
import tvm
from tvm import relax
from tvm import relax as rx
from tvm import relay, te, tir, topi
from tvm.ir.base import assert_structural_equal
from tvm.ir.module import IRModule
from tvm.relax import ExternFunc, ShapeExpr, Tuple
from tvm.relax.testing import dump_ast, nn
from tvm.script import relax as R
from tvm.script import tir as T
from tvm.tir.function import PrimFunc

from utils import LowerToTensorIRPass

# Builder ......
np.random.seed(1)
def n_layer_perceptron(layers, in_size, out_size, hidden_size, batch_size=1):
    # m = tir.Var("m", "int64")
    # n = tir.Var("n", "int64")
    # l = tir.Var("l", "int64")
    # vec_type = rx.DynTensorType(ndim=1, dtype="float32")
    # mat_type = rx.DynTensorType(ndim=2, dtype="float32")
    vec_type = rx.DynTensorType(dtype="float32")
    mat_type = rx.DynTensorType(dtype="float32")

    input_list = [rx.Var("x", [batch_size, in_size], mat_type)]
    w_list = [rx.Var("w_0", [in_size, hidden_size], mat_type)] + \
        [rx.Var("w_" + str(i + 1), [hidden_size, hidden_size], mat_type) for i in range(layers - 2)] + \
        [rx.Var("w_" + str(layers - 1), [hidden_size, out_size], mat_type)]
    b_list = [rx.Var("b_" + str(i), [hidden_size], vec_type) for i in range(layers - 1)] + \
        [rx.Var("b_" + str(layers - 1), [out_size], vec_type)]
    label_list = [rx.Var("y", [batch_size, out_size], mat_type)]
    args_list = input_list + w_list + b_list + label_list

    bb = rx.BlockBuilder()
    with bb.function("MLP", args_list):
        with bb.dataflow():
            current = input_list[0]
            for i in range(layers):
                lv0 = bb.emit(relax.op.matmul(current, w_list[i]))
                lv1 = bb.emit(relax.op.add(lv0, b_list[i]))
                current = bb.emit(relax.op.nn.relu(lv1)) if i < layers - 1 else lv1
            loss = bb.emit(relax.op.nn.softmax_cross_entropy(current, label_list[0]))
            gv0 = bb.emit_output(current)
            gv1 = bb.emit_output(loss)
        bb.emit_func_output((gv0, gv1))
    return bb.get(), relax.transform.SimpleAD("MLP", gv1, w_list + b_list)(bb.get())

def build_vm(layers, in_size, out_size, hidden_size, batch_size=1):
    mod1, mod2 = n_layer_perceptron(layers, in_size, out_size, hidden_size, batch_size)
    # mod1.show()
    # mod2.show()
    # print(dump_ast(mod2["MLP"]))

    TIRModule = LowerToTensorIRPass()(mod2)
    # TIRModule.show()

    ex = relax.vm.build(TIRModule, target="llvm")
    vm = relax.VirtualMachine(ex, tvm.cpu())
    return vm

# Numeric Tests ......

# layers, in_size, out_size, hidden_size, batch_size = 2, 2, 2, 2, 3
# vm = build_vm(layers, in_size, out_size, hidden_size, batch_size)
# def rand_matrix_tvm(shape):
#     return tvm.nd.array(np.random.randint(5, size=shape).astype(np.float32))

# input_list = [rand_matrix_tvm((batch_size, in_size))]
# w_list = [rand_matrix_tvm((in_size, hidden_size))] + \
#     [rand_matrix_tvm((hidden_size, hidden_size)) for i in range(layers - 2)] + \
#     [rand_matrix_tvm((hidden_size, out_size))]
# b_list = [rand_matrix_tvm((hidden_size)) for i in range(layers - 1)] + \
#     [rand_matrix_tvm((out_size))]
# label_list = [rand_matrix_tvm((batch_size, out_size))]

# args = input_list + w_list + b_list + label_list
# print("args:")
# print(args)

# res = vm["MLP"](*args)
# for i in res[0]:
#     print("res:", i)
# for i in res[1:]:
#     print("grad:", i)


# Model Tests ......

# import matplotlib.pyplot as plt
import pickle as pkl

import torch
import torchvision

test_data = torchvision.datasets.FashionMNIST(
    root="data",
    train=False,
    download=False,
    transform=torchvision.transforms.ToTensor()
)


layers, in_size, out_size, hidden_size, batch_size = 2, 784, 10, 128, 64

loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

vm = build_vm(layers, in_size, out_size, hidden_size, batch_size)

mlp_params = {}
import math

a0 = math.sqrt(6.0 / (784 + 128))
a1 = math.sqrt(6.0 / (128 + 128))
a2 = math.sqrt(6.0 / (128 + 10))

def rand_matrix_np(shape, base=1.0):
    return base * np.random.uniform(-1.0, 1.0, size=shape).astype(np.float32)
def rand_matrix_tvm(shape, base=1.0):
    return tvm.nd.array(rand_matrix_np(shape, base))
def to_tvm(array):
    return [tvm.nd.array(i) for i in array]

w_list = [rand_matrix_np((in_size, hidden_size), a0)] + \
    [rand_matrix_np((hidden_size, hidden_size), a1) for i in range(layers - 2)] + \
    [rand_matrix_np((hidden_size, out_size), a2)]
b_list = [rand_matrix_np((hidden_size)) for i in range(layers - 1)] + \
    [rand_matrix_np((out_size))]

success, total = 0, 0
lr = 0.03
epoch = 0

time0 = time.perf_counter()

data_len = len(loader)
train_data_threshold = int(data_len * 0.9)

for img, label in loader:
    # print(img.shape)
    if img.shape[0] != batch_size:
        continue
    data_nd = img.reshape((batch_size, in_size))
    label_nd = np.array([[1 if i == label[j] else 0 for i in range(out_size)] for j in range(batch_size)]).astype(np.float32)
    args = to_tvm([data_nd] + w_list + b_list + [label_nd])
    output, *grads = vm["MLP"](*args)
    output, loss = output[0], output[1]
    pred_kind = np.argmax(output.numpy(), axis=1)

    """
    print("label: ", label_nd)
    print("output:", output)
    print("loss:", loss)
    print("w0_grad", w0_grad)
    print("b0_grad", b0_grad)
    print("w1_grad", w1_grad)
    print("b1_grad", b1_grad)
    """

    epoch += 1
    if epoch < train_data_threshold:
        print("epoch={}, loss={}".format(epoch, loss.numpy()))

        cnt = 0
        for i in range(len(w_list)):
            w_list[i] -= lr * grads[cnt].numpy() / batch_size
            cnt += 1
        for i in range(len(b_list)):
            b_list[i] -= lr * grads[cnt].numpy() / batch_size
            cnt += 1
    else:
        total += batch_size
        success += np.count_nonzero(pred_kind == label.numpy())


print("Prediction Rate On TestSet: ", success / total)
print("time: ", time.perf_counter() - time0)
