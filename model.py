# Lara Popov Zambiasi Bazzi Oberderfer
# Redes Neurais Convolucionais


import torch
import torch.nn as nn

# Rede Neural Convolucional
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=20, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=40, kernel_size=5)

        self.dropout1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(40 * 15 * 15, 1000)

        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1000, 1000)

        self.dropout3 = nn.Dropout(p=0.5)
        self.out = nn.Linear(1000, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, kernel_size=2, stride=2)

        x = self.conv2(x)
        x = torch.relu(x)
        # print(x.shape)
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        # print(x.shape)
        x = torch.flatten(x, 1)
        # print(x.shape)

        x = self.dropout1(x)
        x = self.fc1(x)
        x = torch.relu(x)

        x = self.dropout2(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.dropout3(x)
        x = self.out(x)

        return x


# Rede Neural Convolucional 02 sem dropout
class Net2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=20, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=40, kernel_size=5)
        self.fc1 = nn.Linear(40*15*15, 100)
        self.out = nn.Linear(100, 3)

    def forward(self, x):
        x = self.conv1(x)
        # print("Conv1") # torch.Size([150, 20, 146, 96])
        # print(x.shape)
        x = torch.relu(x)
        # print("relu") # torch.Size([150, 20, 146, 96])
        # print(x.shape)
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        # print("maxpool") # torch.Size([150, 20, 73, 48])
        # print(x.shape)
        x = self.conv2(x)
        # print("Conv2") # torch.Size([150, 40, 69, 44])
        # print(x.shape)
        x = torch.relu(x)
        # print("relu") # torch.Size([150, 40, 69, 44])
        # print(x.shape)
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        # print("maxpool") # torch.Size([125, 40, 15, 9])
        # print(x.shape)
        x = torch.flatten(x, 1)
        # print("flatten") # torch.Size([150, 29920])
        # print(x.shape)
        x = self.fc1(x)
        # print("fc1") # torch.Size([150, 29920])
        # print(x.shape)
        x = torch.relu(x)
        # print("relu") # torch.Size([150, 29920])
        # print(x.shape)
        x = self.out(x)

        return x
