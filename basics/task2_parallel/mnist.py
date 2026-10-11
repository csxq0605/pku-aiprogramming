"""
Task 2: the Task 1 CNN trained with and without nn.DataParallel, comparing wall time and accuracy.
Usage: python mnist.py [--epochs 10] [--data ./data]
Writes results/Loss-parallel-<bool>.png, results/Accuracy-parallel-<bool>.png, results/Time_Accuracy_Comparison.png
and results/result.json.
"""
import argparse
import json
import os
from time import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def parse_data(root):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307, ), (0.3081, ))])

    trainset = torchvision.datasets.MNIST(root = root, train = True, download = True, transform = transform)
    trainloader = DataLoader(trainset, batch_size = 16, shuffle = True, num_workers = 0)

    testset = torchvision.datasets.MNIST(root = root, train = False, download = True, transform = transform)
    testloader = DataLoader(testset, batch_size = 16, shuffle = False, num_workers = 0)

    classes = [str(x) for x in range(10)]
    return trainloader, testloader, classes


class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, padding = 1)
        self.conv2 = nn.Conv2d(6, 10, 3, padding = 1)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.linear1 = nn.Linear(10 * 7 * 7, 120)
        self.linear2 = nn.Linear(120, 84)
        self.linear3 = nn.Linear(84, 10)

    def forward(self, input):
        c1 = F.relu(self.conv1(input))
        s1 = self.maxpool(c1)
        c2 = F.relu(self.conv2(s1))
        s2 = self.maxpool(c2)
        s3 = s2.view(-1, 10 * 7 * 7)
        l1 = F.relu(self.linear1(s3))
        l2 = F.relu(self.linear2(l1))
        output = self.linear3(l2)
        return output


def run(parallel, trainloader, testloader, classes, epochs, seed, device):
    # same seed for both runs, so they start from the same weights and see the same batch order
    torch.manual_seed(seed)
    net = LeNet()
    if parallel:
        net = nn.DataParallel(net)
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr = 0.003, momentum = 0.6)

    losses = []
    start = time()
    print(f"Start Training (DataParallel={parallel})")

    for i in range(epochs):
        running_loss = 0.0
        for j, data in enumerate(trainloader):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if j % 500 == 499:
                losses.append(running_loss / 500)
                running_loss = 0.0
        print(f"[Epoch: {i + 1}/{epochs}, loss: {losses[-1]:.3f}]")
    if device == "cuda":
        torch.cuda.synchronize()
    train_time = time() - start
    print(f"Finished Training in {train_time:.2f} seconds")

    net.eval()
    correct = [0] * 10
    total = [0] * 10
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            predicted = net(images).argmax(dim = 1)
            for c in range(10):
                mask = labels == c
                total[c] += int(mask.sum())
                correct[c] += int((predicted[mask] == c).sum())

    accuracy = [100.0 * correct[i] / total[i] for i in range(10)]
    total_accuracy = 100.0 * sum(correct) / sum(total)
    print(f"Accuracy of the total: {total_accuracy:.2f}%")

    plt.figure(figsize=(10, 5))
    plt.plot(losses, label = "Training loss")
    plt.xlabel("x500 batches")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"results/Loss-parallel-{parallel}.png", dpi = 100)
    plt.close()

    plt.figure(figsize = (10, 5))
    plt.bar(["Total"] + classes, [total_accuracy] + accuracy)
    plt.ylim(90, 100)
    plt.xlabel("Classes")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy of the total and each class")
    plt.savefig(f"results/Accuracy-parallel-{parallel}.png", dpi = 100)
    plt.close()

    return train_time, total_accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--data', default='./data', help='MNIST root (downloaded if missing)')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")
    os.makedirs("results", exist_ok=True)

    trainloader, testloader, classes = parse_data(args.data)
    time1, accuracy1 = run(False, trainloader, testloader, classes, args.epochs, args.seed, device)
    time2, accuracy2 = run(True, trainloader, testloader, classes, args.epochs, args.seed, device)

    labels = ["Plain module", f"nn.DataParallel ({num_gpus} GPU)"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(labels, [time1, time2], color=['#2a78d6', '#eb6834'])
    ax1.set_ylabel("Training Time (s)")
    ax1.set_title("Training Time Comparison")

    ax2.bar(labels, [accuracy1, accuracy2], color=['#2a78d6', '#eb6834'])
    ax2.set_ylim(90, 100)
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy Comparison")

    plt.tight_layout()
    plt.savefig("results/Time_Accuracy_Comparison.png", dpi = 100)

    with open("results/result.json", "w") as f:
        json.dump(dict(num_gpus = num_gpus, epochs = args.epochs, seed = args.seed,
                       plain = dict(train_seconds = time1, test_accuracy = accuracy1),
                       data_parallel = dict(train_seconds = time2, test_accuracy = accuracy2)), f)
