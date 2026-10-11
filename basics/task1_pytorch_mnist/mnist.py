"""
Task 1: LeNet-style CNN on MNIST with PyTorch.
Usage: python mnist.py [--epochs 10] [--data ./data]
Writes results/Loss.png, results/Accuracy.png and results/result.json.
"""
import argparse
import json
import os
import time

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

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--data', default='./data', help='MNIST root (downloaded if missing)')
args = parser.parse_args()

torch.manual_seed(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs("results", exist_ok=True)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307, ), (0.3081, ))])

trainset = torchvision.datasets.MNIST(root = args.data, train = True, download = True, transform = transform)
trainloader = DataLoader(trainset, batch_size = 16, shuffle = True, num_workers = 0)

testset = torchvision.datasets.MNIST(root = args.data, train = False, download = True, transform = transform)
testloader = DataLoader(testset, batch_size = 16, shuffle = False, num_workers = 0)

classes = [str(x) for x in range(10)]

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

net = LeNet()
net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.003, momentum = 0.6)

losses = []
print(f"Start Training (device={device})")
start = time.time()

for i in range(args.epochs):
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
            # mean training loss of the last 500 batches
            losses.append(running_loss / 500)
            running_loss = 0.0
    print(f"[Epoch: {i + 1}/{args.epochs}, loss: {losses[-1]:.3f}]")
train_time = time.time() - start

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
for i in range(10):
    print(f"Accuracy of {classes[i]}: {accuracy[i]:.2f}%")

plt.figure(figsize=(10, 5))
plt.plot(losses, label = "Training loss")
plt.xlabel("x500 batches")
plt.ylabel("Loss")
plt.legend()
plt.savefig("results/Loss.png", dpi = 100)

plt.figure(figsize = (10, 5))
plt.bar(["Total"] + classes, [total_accuracy] + accuracy)
plt.ylim(90, 100)
plt.xlabel("Classes")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy of the total and each class")
plt.savefig("results/Accuracy.png", dpi = 100)

with open("results/result.json", "w") as f:
    json.dump(dict(epochs = args.epochs, seed = args.seed, losses = losses, test_accuracy = total_accuracy,
                   class_accuracy = accuracy, train_seconds = train_time), f)

print(f"Finished Training in {train_time:.1f} s")
