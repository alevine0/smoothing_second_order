
import torch.nn as nn
import torch.nn.functional as F



class SwissRollNet(nn.Module):
    def __init__(self):
        super(SwissRollNet, self).__init__()
        self.lin1 = nn.Linear(2,10)
        self.lin2 = nn.Linear(10,10)
        self.lin3 = nn.Linear(10, 2)
    def forward(self, x):
        out = F.relu(self.lin1(x.view(x.size(0), -1)))
        out = F.relu(self.lin2(out))
        out = self.lin3(out)
        return out
