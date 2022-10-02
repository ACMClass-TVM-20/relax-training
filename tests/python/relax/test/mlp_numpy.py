from __future__ import annotations
import numpy as np

import torch
import torchvision
import matplotlib.pyplot as plt
import pickle as pkl

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


print("Build mlp")


def nn_relu(x):
    return np.maximum(x, 0.0)

def nn_gradrelu_(x):
    return (x > 0)

def nn_softmax(x):
    x = np.exp(x)
    return x / np.sum(x)

def nn_cross_entropy(x, y):
    ret = np.sum(-np.log(x) * y)
    return 0.0 if np.isnan(ret) else ret

class AutoDiffMLP:

    def main(self, x, w0, b0, w1, b1, label):
        # linear0
        lv0 = np.matmul(x, w0) # shape: (1, 128)
        lv1 = np.add(lv0, b0)
        # relu0
        lv2 = nn_relu(lv1)
        # linear1
        lv3 = np.matmul(lv2, w1) # shape: (1, 10)
        out = np.add(lv3, b1)
        lv4 = nn_softmax(out)
        loss = nn_cross_entropy(lv4, label)
        # gradient

        # crossEnt-softMax derive
        out_adjoint = np.subtract(lv4, label) # shape: (1, 10)

        # b1_adjoint  = out_adjoint
        # lv3_adjoint = out_adjoint
        lv2_trans = np.transpose(lv2)  # shape: (128, 1)
        w1_trans  = np.transpose(w1)   # shape: (10, 128)
        lv2_adjoint = np.matmul(out_adjoint, w1_trans) # shape: (1, 128)
        w1_adjoint  = np.matmul(lv2_trans, out_adjoint) # shape: (128, 10)

        lv1_gradrelu_ = nn_gradrelu_(lv1) # shape: (1, 128)
        lv1_adjoint  = np.multiply(lv2_adjoint, lv1_gradrelu_) # shape: (1, 128)

        # b0_adjoint = lv1_adjoint
        # lv0_adjoint = lv1_adjoint

        x_trans = np.transpose(x) # shape: (784, 1)
        w0_trans = np.transpose(w0)
        w0_adjoint = np.matmul(x_trans, lv1_adjoint) # shape: (784, 128)

        return out, loss, w0_adjoint, lv1_adjoint, w1_adjoint, out_adjoint


model = AutoDiffMLP()

"""
    train
"""

success, total = 0, 0
lr = 0.03

batch_size = 64
total_loss = 0
epoch = 0
gradient_dict = {}
arg_names = ["w0", "b0", "w1", "b1"]

for arg in arg_names:
    gradient_dict[arg] = 0

for img, label in loader:
    data_nd = np.array(img.reshape(1, 784))
    label_nd = np.array([[1 if i == label[0] else 0 for i in range(10)]]).astype(np.float32)
    output, loss, w0_grad, b0_grad, w1_grad, b1_grad = model.main(data_nd, mlp_params["w0"], mlp_params["b0"], mlp_params["w1"], mlp_params["b1"], label_nd)
    pred_kind = np.argmax(output, axis=1)
    total += 1
    if pred_kind[0] == label[0]:
        success += 1
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
    total_loss += loss
    b0_grad_numpy = b0_grad
    b1_grad_numpy = b1_grad
    gradient_dict["w0"] += w0_grad
    gradient_dict["b0"] += b0_grad_numpy.reshape(b0_grad_numpy.shape[1])
    gradient_dict["w1"]    += w1_grad
    gradient_dict["b1"]    += b1_grad_numpy.reshape(b1_grad_numpy.shape[1])

    if epoch % batch_size == 0:
        print("epoch={}, loss={}".format(epoch, total_loss))

        for arg in gradient_dict:
            mlp_params[arg] -= lr * (gradient_dict[arg] / batch_size)
            gradient_dict[arg] = 0

        total_loss = 0
    

print("Prediction Rate: ", float(success)/float(total))
