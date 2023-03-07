import torch.nn as nn
from hyperparameters import *

class GlobalAttention(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear0 = nn.Linear(
            in_features=HIDDEN_SIZE * 2,
            out_features=HIDDEN_SIZE * 2,
        )
        self.linear1 = nn.Linear(
            in_features=HIDDEN_SIZE * 4,
            out_features=HIDDEN_SIZE * 2,
        )

    def forward(self, input):
        output = self.linear0(input)
        output = self.linear1(output)

        return output