import pytest
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy

def read_data(idx, lines):
    data_line = list(map(float, lines[idx].strip().split()))
    return data_line
    
def test_Fc():
    file1 = open("testdata.txt", "r")
    lines = file1.readlines()
    file2 = open("cppout.txt", "r")
    lines2 = file2.readlines()
    batch_size = 2
    feature_in = 3
    feature_out = 4
    input_data = read_data(0, lines)
    inputs = torch.tensor(input_data).reshape(batch_size, feature_in)
    inputs.requires_grad = True
    weight_data = read_data(1, lines)
    weight = torch.tensor(weight_data).reshape(feature_in, feature_out).T
    bias_data = read_data(2, lines)
    bias = torch.tensor(bias_data).reshape(feature_out)
    output_cpp_data = read_data(0, lines2)
    output_cpp = torch.tensor(output_cpp_data).reshape(batch_size, feature_out)

    fc = nn.Linear(feature_in, feature_out)
    fc.weight = nn.Parameter(weight)
    fc.bias = nn.Parameter(bias)
    fc.weight.requires_grad = True
    fc.bias.requires_grad = True
    output = fc(inputs)

    assert torch.allclose(output, output_cpp, atol=1e-5)

    output_grad_data = read_data(3, lines)
    output_grad = torch.tensor(output_grad_data).reshape(batch_size, feature_out)
    output.backward(output_grad)
    input_grad_cpp_data = read_data(1, lines2)
    input_grad_cpp = torch.tensor(input_grad_cpp_data).reshape(batch_size, feature_in)
    weight_grad_cpp_data = read_data(2, lines2)
    weight_grad_cpp = torch.tensor(weight_grad_cpp_data).reshape(feature_in, feature_out).T
    bias_grad_cpp_data = read_data(3, lines2)
    bias_grad_cpp = torch.tensor(bias_grad_cpp_data).reshape(feature_out)

    assert torch.allclose(inputs.grad, input_grad_cpp, atol=1e-5)
    assert torch.allclose(fc.weight.grad, weight_grad_cpp, atol=1e-5)
    assert torch.allclose(fc.bias.grad, bias_grad_cpp, atol=1e-5)

    file1.close()
    file2.close()

def test_Conv():
    file1 = open("testdata.txt", "r")
    lines = file1.readlines()
    file2 = open("cppout.txt", "r")
    lines2 = file2.readlines()
    batch_size = 2
    channel_in = 3
    map_size = 5
    channel_out = 6
    kernel_size = 3
    input_data = read_data(4, lines)
    inputs = torch.tensor(input_data).reshape(batch_size, channel_in, map_size, map_size)
    inputs.requires_grad = True
    weight_data = read_data(5, lines)
    weight = torch.tensor(weight_data).reshape(channel_out, channel_in, kernel_size, kernel_size)
    output_cpp_data = read_data(4, lines2)
    output_cpp = torch.tensor(output_cpp_data).reshape(batch_size, channel_out, map_size - kernel_size + 1, map_size - kernel_size + 1)

    conv = nn.Conv2d(channel_in, channel_out, kernel_size)
    conv.weight = nn.Parameter(weight)
    conv.weight.requires_grad = True
    output = conv(inputs)
    
    assert torch.allclose(output, output_cpp, atol=0.5)

    output_grad_data = read_data(6, lines)
    output_grad = torch.tensor(output_grad_data).reshape(batch_size, channel_out, map_size - kernel_size + 1, map_size - kernel_size + 1)
    output.backward(output_grad)
    input_grad_cpp_data = read_data(5, lines2)
    input_grad_cpp = torch.tensor(input_grad_cpp_data).reshape(batch_size, channel_in, map_size, map_size)
    weight_grad_cpp_data = read_data(6, lines2)
    weight_grad_cpp = torch.tensor(weight_grad_cpp_data).reshape(channel_out, channel_in, kernel_size, kernel_size)

    assert torch.allclose(inputs.grad, input_grad_cpp, atol=0.5)
    assert torch.allclose(conv.weight.grad, weight_grad_cpp, atol=0.5)

    file1.close()
    file2.close()

def test_MaxPool():
    file1 = open("testdata.txt", "r")
    lines = file1.readlines()
    file2 = open("cppout.txt", "r")
    lines2 = file2.readlines()
    batch_size = 2
    channel_in = 3
    map_size = 5
    kernel_size = 2

    input_data = read_data(7, lines)
    inputs = torch.tensor(input_data).reshape(batch_size, channel_in, map_size, map_size)
    inputs.requires_grad = True
    output_cpp_data = read_data(7, lines2)
    output_cpp = torch.tensor(output_cpp_data).reshape(batch_size, channel_in, map_size//2, map_size//2)
    mask_cpp_data = read_data(8, lines2)
    mask_cpp = torch.tensor(mask_cpp_data).reshape(batch_size, channel_in, map_size//2, map_size//2).long()

    pool = nn.MaxPool2d(kernel_size=kernel_size, stride=2, return_indices=True)
    output, indices = pool(inputs)

    assert torch.allclose(output, output_cpp, atol=1e-5)
    assert torch.allclose(indices, mask_cpp, atol=1e-5)

    output_grad_data = read_data(8, lines)
    output_grad = torch.tensor(output_grad_data).reshape(batch_size, channel_in, map_size//2, map_size//2)
    output.backward(output_grad)
    input_grad_cpp_data = read_data(9, lines2)
    input_grad_cpp = torch.tensor(input_grad_cpp_data).reshape(batch_size, channel_in, map_size, map_size)

    assert torch.allclose(inputs.grad, input_grad_cpp, atol=1e-5)

    file1.close()
    file2.close()

def test_Softmax_and_CrossEntropyLoss():
    file1 = open("testdata.txt", "r")
    lines = file1.readlines()
    file2 = open("cppout.txt", "r")
    lines2 = file2.readlines()
    batch_size = 50
    feature_in = 10
    input_data = read_data(9, lines)
    inputs = torch.tensor(input_data).reshape(batch_size, feature_in)
    inputs.requires_grad = True
    output_cpp_data = read_data(10, lines2)
    output_cpp = torch.tensor(output_cpp_data).reshape(batch_size, feature_in)

    softmax = nn.Softmax(dim=1)
    output = softmax(inputs)

    assert torch.allclose(output, output_cpp, atol=1e-5)

    target_data = list(map(int, read_data(10, lines)))
    target = torch.tensor(target_data).reshape(batch_size)
    loss_cpp_data = read_data(11, lines2)
    loss_cpp = torch.tensor(loss_cpp_data)

    loss1 = 0
    for i in range(batch_size):
        max_idx = 0
        max_val = output[i][0]
        for j in range(1, feature_in):
            if output[i][j] > max_val:
                max_val = output[i][j]
                max_idx = j
        loss1 += int(max_idx != target[i])
    loss1 /= batch_size
    loss1 = torch.tensor([loss1])

    assert torch.allclose(loss1, loss_cpp, atol=1e-5)

    criterion = nn.CrossEntropyLoss(reduction="sum")
    loss = criterion(inputs, target)
    loss_cpp_data = read_data(12, lines2)
    loss_cpp = torch.tensor(loss_cpp_data)

    assert torch.allclose(loss, loss_cpp, atol=1e-5)

    loss.backward()
    input_grad_cpp_data = read_data(13, lines2)
    input_grad_cpp = torch.tensor(input_grad_cpp_data).reshape(batch_size, feature_in)

    assert torch.allclose(inputs.grad, input_grad_cpp, atol=1e-5)

    file1.close()
    file2.close()

    with open("testdata.txt", "w") as file:
        pass
    with open("cppout.txt", "w") as file:
        pass
