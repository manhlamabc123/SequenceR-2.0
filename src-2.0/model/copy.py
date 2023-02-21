import torch.nn as nn
from hyperparameters import *

class CopyGenerator(nn.Module):
    def __init__(self, output_size):
        super().__init__()

        self.linear = nn.Linear(
            in_features=HIDDEN_SIZE * 2,
            out_features=output_size,
            bias=True
        )
        self.linear_copy = nn.Linear(
            in_features=HIDDEN_SIZE * 2,
            out_features=1,
            bias=True
        )

    def forward(self, input):
        output_linear = self.linear(input)
        output_linear_copy = self.linear(input)

        return output_linear, output_linear_copy