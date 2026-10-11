import torch
import torch.nn as nn
import numpy as np
import pytest
from myTensor import Tensor_float as tf
from myTensor import Tensor_int as ti
from myLayer import FcForward, FcBackward, ConvForward, ConvBackward, MaxPoolingForward, MaxPoolingBackward, SoftmaxForward, SoftmaxLoss, CrossEntropyLoss, CrossEntropyLossBackward

def test_Fc():
    inputs = tf([2, 3], "gpu")
    weights = tf([3, 4], "gpu")
    bias = tf([4], "gpu")
    output = tf([2, 4], "gpu")
    inputs.random(0, 10)
    weights.random(0, 10)
    bias.random(0, 10)
    FcForward(inputs, output, weights, bias)
    inputs_torch = torch.tensor(inputs.get_data()).reshape(2, 3)
    inputs_torch.requires_grad = True
    weights_torch = torch.tensor(weights.get_data()).reshape(3, 4).T
    bias_torch = torch.tensor(bias.get_data()).reshape(4)
    fc = nn.Linear(3, 4)
    fc.weight = nn.Parameter(weights_torch)
    fc.bias = nn.Parameter(bias_torch)
    fc.weight.requires_grad = True
    fc.bias.requires_grad = True
    output_torch = fc(inputs_torch)
    assert np.allclose(np.array(output.get_data()), output_torch.detach().numpy().reshape(8))
    inputs_grad = tf([2, 3], "gpu")
    weights_grad = tf([3, 4], "gpu")
    bias_grad = tf([4], "gpu")
    output_grad = tf([2, 4], "gpu")
    output_grad.random(0, 10)
    FcBackward(inputs, output, weights, bias, inputs_grad, output_grad, weights_grad, bias_grad)
    output_grad_torch = torch.tensor(output_grad.get_data()).reshape(2, 4)
    output_torch.backward(output_grad_torch)
    assert np.allclose(np.array(inputs_grad.get_data()), inputs_torch.grad.detach().numpy().reshape(6))
    assert np.allclose(np.array(weights_grad.get_data()), fc.weight.grad.T.detach().numpy().reshape(12))
    assert np.allclose(np.array(bias_grad.get_data()), fc.bias.grad.detach().numpy().reshape(4))

def test_Conv():
    inputs = tf([2, 3, 5, 5], "gpu")
    weights = tf([6, 3, 3, 3], "gpu")
    output = tf([2, 6, 3, 3], "gpu")
    inputs.random(0, 10)
    weights.random(0, 10)
    ConvForward(inputs, output, weights, 0, 0, 1, 1)
    inputs_torch = torch.tensor(inputs.get_data()).reshape(2, 3, 5, 5)
    inputs_torch.requires_grad = True
    weights_torch = torch.tensor(weights.get_data()).reshape(6, 3, 3, 3)
    weights_torch.requires_grad = True
    output_torch = torch.nn.functional.conv2d(inputs_torch, weights_torch, bias=None, stride=1, padding=0)
    assert np.allclose(np.array(output.get_data()), output_torch.detach().numpy().reshape(108))
    inputs_grad = tf([2, 3, 5, 5], "gpu")
    weights_grad = tf([6, 3, 3, 3], "gpu")
    output_grad = tf([2, 6, 3, 3], "gpu")
    output_grad.random(0, 10)
    ConvBackward(inputs, output, weights, inputs_grad, output_grad, weights_grad, 0, 0, 1, 1)
    output_grad_torch = torch.tensor(output_grad.get_data()).reshape(2, 6, 3, 3)
    output_torch.backward(output_grad_torch)
    assert np.allclose(np.array(inputs_grad.get_data()), inputs_torch.grad.detach().numpy().reshape(150))
    assert np.allclose(np.array(weights_grad.get_data()), weights_torch.grad.detach().numpy().reshape(162))

def test_MaxPooling():
    inputs = tf([2, 3, 5, 5], "gpu")
    output = tf([2, 3, 2, 2], "gpu")
    mask = tf([2, 3, 2, 2], "gpu")
    inputs.random(0, 10)
    MaxPoolingForward(inputs, output, mask, [2, 2], 0, 0, 2, 2)
    inputs_torch = torch.tensor(inputs.get_data()).reshape(2, 3, 5, 5)
    inputs_torch.requires_grad = True
    maxpool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
    output_torch, mask_torch = maxpool(inputs_torch)
    assert np.allclose(np.array(output.get_data()), output_torch.detach().numpy().reshape(24))
    assert np.allclose(np.array(mask.get_data()), mask_torch.detach().numpy().reshape(24))
    inputs_grad = tf([2, 3, 5, 5], "gpu")
    output_grad = tf([2, 3, 2, 2], "gpu")
    output_grad.random(0, 10)
    MaxPoolingBackward(output_grad, mask, inputs_grad, inputs.get_shape(), [2, 2], 0, 0, 2, 2)
    output_grad_torch = torch.tensor(output_grad.get_data()).reshape(2, 3, 2, 2)
    output_torch.backward(output_grad_torch)
    assert np.allclose(np.array(inputs_grad.get_data()), inputs_torch.grad.detach().numpy().reshape(150))

def test_Softmax_and_CrossEntropyLoss():
    inputs = tf([50, 10], "gpu")
    output = tf([50, 10], "gpu")
    inputs.random(0, 10)
    SoftmaxForward(inputs, output)
    inputs_torch = torch.tensor(inputs.get_data()).reshape(50, 10)
    inputs_torch.requires_grad = True
    softmax = nn.Softmax(dim=1)
    output_torch = softmax(inputs_torch)
    assert np.allclose(np.array(output.get_data()), output_torch.detach().numpy().reshape(500))
    labels = ti([50], "gpu")
    labels.random(0, 10)
    loss = SoftmaxLoss(output, labels)
    labels_torch = torch.tensor(labels.get_data()).reshape(50)
    loss_torch = 0
    for i in range(50):
        max_idx = 0
        max_val = output_torch[i][0]
        for j in range(1, 10):
            if output_torch[i][j] > max_val:
                max_val = output_torch[i][j]
                max_idx = j
        loss_torch += int(max_idx != labels_torch[i])
    loss_torch /= 50
    assert np.allclose(np.array([loss]), np.array([loss_torch]))
    loss = tf([1], "gpu")
    CrossEntropyLoss(output, labels, loss)
    criterion = nn.CrossEntropyLoss(reduction="sum")
    loss_torch = criterion(inputs_torch, labels_torch)
    assert np.allclose(np.array(loss.get_data()), loss_torch.detach().numpy())
    inputs_grad = tf([50, 10], "gpu")
    CrossEntropyLossBackward(output, labels, inputs_grad)
    loss_torch.backward()
    assert np.allclose(np.array(inputs_grad.get_data()), inputs_torch.grad.detach().numpy().reshape(500))

'''
if __name__ == "__main__":
    test_Fc()
    print("Fc test passed")
'''