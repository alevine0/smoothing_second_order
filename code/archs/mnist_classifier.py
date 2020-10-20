
import torch.nn as nn
import torch.nn.functional as F



class MnistNet(nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 4, stride=2, padding=1)
        self.lin1 = nn.Linear(32*7*7,100)
        self.lin2 = nn.Linear(100, 10)
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = self.lin2(F.relu(self.lin1(out.view(out.size(0), -1))))
        return out
