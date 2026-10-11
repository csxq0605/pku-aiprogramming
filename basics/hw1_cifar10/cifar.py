"""
HW1: LeNet on CIFAR-10 with PyTorch, comparing SGD momentum values.
Usage: python cifar.py [--momentum 0.9] [--lr 0.006] [--epochs 10] [--data ./cifar]
Writes results/Loss-momentum=<m>.png, results/Accuracy-momentum=<m>.png and results/momentum=<m>.json.
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

parser = argparse.ArgumentParser(description='Training parameters')
parser.add_argument('--lr', type=float, default=0.006, help='Learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--data', default='./cifar', help='CIFAR-10 root (downloaded if missing)')
args = parser.parse_args()

torch.manual_seed(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs("results", exist_ok=True)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root = args.data, train = True, download = True, transform = transform)
trainloader = DataLoader(trainset, batch_size = 64, shuffle = True, num_workers = 0)

testset = torchvision.datasets.CIFAR10(root = args.data, train = False, download = True, transform = transform)
testloader = DataLoader(testset, batch_size = 64, shuffle = False, num_workers = 0)

classes = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, 5)
        self.conv2 = nn.Conv2d(10, 16, 5)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.linear1 = nn.Linear(16 * 5 * 5, 120)
        self.linear2 = nn.Linear(120, 84)
        self.linear3 = nn.Linear(84, 10)

    def forward(self, input):
        c1 = F.relu(self.conv1(input))
        s1 = self.maxpool(c1)
        c2 = F.relu(self.conv2(s1))
        s2 = self.maxpool(c2)
        s3 = s2.view(-1, 16 * 5 * 5)
        l1 = F.relu(self.linear1(s3))
        l2 = F.relu(self.linear2(l1))
        output = self.linear3(l2)
        return output

net = LeNet()
net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = args.lr, momentum = args.momentum)

losses = []
print(f"Start Training (momentum={args.momentum}, lr={args.lr}, device={device})")
start = time.time()

for i in range(args.epochs):
    running_loss = 0.0
    for j, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if j % 250 == 249:
            # mean training loss of the last 250 batches
            losses.append(running_loss / 250)
            running_loss = 0.0
    print(f"[Epoch: {i + 1}, loss: {losses[-1]:.3f}]")
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
plt.xlabel("x250 batches")
plt.ylabel("Loss")
plt.legend()
plt.savefig(f"results/Loss-momentum={args.momentum}.png", dpi = 100)

plt.figure(figsize = (10, 5))
plt.bar(["Total"] + classes, [total_accuracy] + accuracy)
plt.xlabel("Classes")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy of the total and each class")
plt.savefig(f"results/Accuracy-momentum={args.momentum}.png", dpi = 100)

with open(f"results/momentum={args.momentum}.json", "w") as f:
    json.dump(dict(momentum = args.momentum, lr = args.lr, epochs = args.epochs, seed = args.seed, losses = losses,
                   test_accuracy = total_accuracy, class_accuracy = accuracy, train_seconds = train_time), f)

print(f"Finished Training in {train_time:.1f} s")
