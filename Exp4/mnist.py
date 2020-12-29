import torch
import scipy
import random
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

import torchvision


class Model(nn.Module):
    def __init__(self) -> None:
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(8, 32, kernel_size=4, stride=1)
        self.hidden1 = nn.Linear(5 * 5 * 32, 150)
        self.dropout = nn.Dropout()
        self.hidden2 = nn.Linear(150, 40)
        self.hidden3 = nn.Linear(40, 10)

    def forward(self, x):  # 28 * 28
        x = self.conv1(x)  # 26 * 26 * 8
        x = F.relu(x)
        x = F.max_pool2d(x, 2)  # 13 * 13 * 8
        x = self.conv2(x)  # 10 * 10 * 32
        x = F.relu(x)
        x = F.max_pool2d(x, 2)  # 5 * 5 * 32

        x = x.view(-1, 5 * 5 * 32)

        x = self.hidden1(x)
        x = F.relu(x)
        x = self.hidden2(x)
        x = F.relu(x)
        x = self.hidden3(x)
        return x


class CNN():
    def __init__(self, lr=0.1, epochs=15):
        self.model = Model()
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)  # 随机梯度下降优化器
        self.criterion = nn.CrossEntropyLoss()  # 交叉熵
        self.epochs = epochs
        if torch.cuda.is_available():
            self.model.cuda()
            self.criterion = self.criterion.cuda()

    def fit(self, train: DataLoader, test: DataLoader):
        for epoch in range(self.epochs):
            loss = 0.0
            acc = 0.0
            for i, (data, target) in enumerate(train):
                X_train, y_train = data, target
                if torch.cuda.is_available():
                    X_train = X_train.cuda()
                    y_train = y_train.cuda()
                X_train, y_train = Variable(X_train), Variable(y_train)
                pred = self.model(X_train)
                loss_ = self.criterion(pred, y_train)
                loss += loss_.item()
                self.optimizer.zero_grad()
                loss_.backward()
                self.optimizer.step()

            for i, (data, target) in enumerate(test):
                X_test, y_test = data, target
                if torch.cuda.is_available():
                    X_test = X_test.cuda()
                    y_test = y_test.cuda()
                X_test, y_test = Variable(X_test), Variable(y_test)
                pred = self.model(X_test)
                pred = torch.max(pred.data, 1)[1]
                acc += torch.sum(pred == y_test)

            print('Epoch [{}/{}], Loss: {:.4f}, Acc: {:.4f}'.format(
                epoch + 1,
                self.epochs,
                loss,
                100 * acc / (len(data_loader_test) * 128),
            ))

    def pred(self, X: DataLoader):
        ans = []
        if (len(X) == 2):
            X = list(X)
            train = X[0]
            if torch.cuda.is_available:
                train = train.cuda()
            train = Variable(train)
            pred = self.model(train)
            pred = torch.max(pred.data, 1)[1]
            if torch.cuda.is_available:
                pred = pred.cpu()
            pred = pred.data.numpy()
            ans.append(pred[0])
            ans.append(X[1].data.numpy()[0])
        else:
            for i, (data, target) in enumerate(X):
                train = data
                if torch.cuda.is_available:
                    train = train.cuda()
                train = Variable(train)
                pred = self.model(train)
                pred = torch.max(pred.data, 1)[1]
                if torch.cuda.is_available:
                    pred = pred.cpu()
                pred = pred.data.numpy()
                ans.append(pred[0])
        return ans


if __name__ == "__main__":
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307, ), (0.3081, ))])
    data_train = datasets.MNIST(root="./data",
                                transform=transform,
                                train=True,
                                download=True)
    data_test = datasets.MNIST(root="./data", transform=transform, train=False)
    data_loader_train = torch.utils.data.DataLoader(dataset=data_train,
                                                    batch_size=128,
                                                    shuffle=True)
    data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
                                                   batch_size=128,
                                                   shuffle=True)

    cnn = CNN(lr=0.2, epochs=30)
    # cnn = torch.load('cnn.pt')
    cnn.fit(data_loader_train, data_loader_test)
    torch.save(cnn, 'cnn.pt')

    while True:
        idx = random.randint(0, len(data_test))
        image, _ = data_test[idx]
        image = torchvision.utils.make_grid(image)
        image = image.numpy().transpose(1, 2, 0)
        plt.imshow(image)
        test = torch.utils.data.DataLoader(dataset=data_test[idx], batch_size=1)
        ans = cnn.pred(test)
        plt.title("pred = " + str(ans[0]) + "  " + "label = " + str(ans[1]))
        plt.show()
