
import torch.nn as nn
import torch.nn.functional as F



class SmallMnistNet(nn.Module):
    def __init__(self):
        super(SmallMnistNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 2, stride=1, padding=0)
        self.lin1 = nn.Linear(4*6*6,100)
        self.lin2 = nn.Linear(100, 10)
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.lin2(F.relu(self.lin1(out.view(out.size(0), -1))))
        return out
