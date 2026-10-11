import sys

import torch
from torchvision import datasets, transforms
import numpy as np
from myTensor import Tensor_float as tf
from myTensor import Tensor_int as ti

def load_mnist_data():
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])

    root = sys.argv[1] if len(sys.argv) > 1 else './data'
    train_dataset = datasets.MNIST(root=root, train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
    test_dataset = datasets.MNIST(root=root, train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    train_data, train_labels = next(iter(train_loader))
    train_data = train_data.numpy()
    train_labels = train_labels.numpy()

    test_data, test_labels = next(iter(test_loader))
    test_data = test_data.numpy()
    test_labels = test_labels.numpy()

    return (train_data, train_labels), (test_data, test_labels)

def main_test_Tensor():
    a = np.array([1, 2, 3, 4, 5 ,6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]).reshape(2, 2, 2, 2)
    mt = tf(a, "gpu")
    print("Tensor mt1 init from numpy:\n", mt)
    print("device:\n", mt.get_device())
    print()
    print("shape:\n", mt.get_shape())
    print()
    print("data:\n", mt.get_data())
    print()
    print("Test Tensor functions:")
    print()
    print("cpu():\n", mt.cpu())
    print("gpu():\n", mt.gpu())
    print("Size():\n", mt.Size())
    print()
    print("zeros():\n", mt.zeros())
    print("ones():\n", mt.ones())
    print("negative():\n", mt.negative())
    print("floatks():\n", mt.floatks(2))
    print("random():\n", mt.random(-1, 1))
    print("mults():\n", mt.mults(5))
    print("flatten():\n", mt.flatten())
    print("resize():\n", mt.resize([4, 4]))
    print("slice: get Tensor[2][1]\n", mt[2][1])
    mt2 = tf([4, 4], "gpu")
    mt2.random(-1, 1)
    print("another Tensor:\n", mt2)
    print("mt2+mt1:\n", mt2+mt)
    print("mt2-mt1:\n", mt2-mt)
    print("ReluForward() using mt1:\n", mt.ReluForward())
    print("ReluBackward() using mt2 as grad:\n", mt.ReluBackward(mt2))
    print("SigmoidForward() using mt1:\n", mt.SigmoidForward())
    print("SigmoidBackward() using mt2 as grad:\n", mt.SigmoidBackward(mt2))

if __name__ == "__main__":
    print("Test Tensor:")
    print()
    main_test_Tensor()
    print("Test Tensor done!")
    print()
    print("Test Build Tensor from MNIST data:")
    (train_data, train_labels), (test_data, test_labels) = load_mnist_data()
    train_data = tf(train_data, "gpu")
    train_labels = ti([60000], "gpu", train_labels.tolist())
    test_data = tf(test_data, "gpu")
    test_labels = ti([10000], "gpu", test_labels.tolist())
    print("Train data shape:", train_data.get_shape())
    print("Train labels shape:", train_labels.get_shape())
    print("Test data shape:", test_data.get_shape())
    print("Test labels shape:", test_labels.get_shape())
    print("Test Build Tensor from MNIST data done!")