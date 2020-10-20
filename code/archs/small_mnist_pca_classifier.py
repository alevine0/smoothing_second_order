
import torch.nn as nn
import torch.nn.functional as F



class SmallMnistPCANet(nn.Module):
    def __init__(self):
        super(SmallMnistPCANet, self).__init__()
        self.lin1 = nn.Linear(10,100)
        self.lin2 = nn.Linear(100,100)
        self.lin3 = nn.Linear(100, 10)
    def forward(self, x):
        out = F.relu(self.lin1(x.view(x.size(0), -1)))
        out = F.relu(self.lin2(out))
        out = self.lin3(out)
        return out
