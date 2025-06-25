import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import  DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307, ), (0.3081, ))])

trainset = torchvision.datasets.MNIST(root = './data', train = True, download = True, transform = transform)
trainloader = DataLoader(trainset, batch_size = 16, shuffle = True, num_workers = 0)

testset = torchvision.datasets.MNIST(root = './data', train = False, download = True, transform = transform)
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
net.to("cuda")
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.003, momentum = 0.6)

epoch = 10
losses = []
print("Start Training")

for i in range(epoch):
    running_loss = 0.0
    progress = tqdm(enumerate(trainloader), total = len(trainloader), desc = f"Epoch: {i + 1}/{epoch}", unit = "batch")
    for j, data in progress:
        inputs, labels = data[0].to("cuda"), data[1].to("cuda")
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if j % 500 == 499:
            progress.set_postfix(loss = f"{running_loss / 500:.3f}")
            losses.append(running_loss / 500)
            running_loss = 0.0
    progress.close()

torch.save(net.state_dict(), f"models/cnn.pth")

net.eval()
correct = [0] * 10
total = [0] * 10
with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to("cuda"), labels.to("cuda")
        outputs = net(images)
        predicted = torch.max(outputs, 1)[1]
        for i in range(len(predicted)):
            label = labels[i]
            total[label] += 1
            if label == predicted[i]:
                correct[label] += 1

accuracy = [int((correct[i] / total[i]) * 10000) // 100 for i in range(10)]
print(f"Accuracy of the total: {np.mean(accuracy):.2f}%" )
for i in range(10):
    print(f"Accuracy of {classes[i]}: {accuracy[i]:.2f}%")

plt.figure(figsize=(20, 10))
plt.plot(losses, label = "Training loss")
plt.xlabel, plt.ylabel = "Epoch No.", "Loss"
plt.legend()
plt.savefig(f"results/Loss.png")
plt.show()

plt.figure(figsize = (20, 10))
plt.bar(["Total"] + classes, [np.mean(accuracy)] + accuracy)
plt.xlabel, plt.ylabel = "Classes", "Accuracy"
plt.title("Accuracy of the total and each class")
plt.savefig(f"results/Accuracy.png")
plt.show()

print("Finished Training")