import torch.nn as nn

class ResidualBlock32(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(32,32,3,padding=1)
        self.conv2 = nn.Conv2d(32,32,3,padding=1)
        self.relu = nn.ReLU()
    def forward(self,x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.relu(out)
        out = out + identity
        return out
class ResidualBlock64(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64,64,3,padding=1)
        self.conv2 = nn.Conv2d(64,64,3,padding=1)
        self.relu = nn.ReLU()
    def forward(self,x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.relu(out)
        out = out + identity
        return out
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            ResidualBlock32(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64,64,3,padding=1),
            nn.BatchNorm2d(64),
            ResidualBlock64(),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.identifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8*8*64,128),
            nn.ReLU(),
            nn.Linear(128,10)
        )
    def forward(self,x):
        x = self.features(x)
        x = self.identifier(x)
        return x