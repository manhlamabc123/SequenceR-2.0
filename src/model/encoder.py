import torch.nn as nn
from hyperparameters import *
import torch

class Encoder(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=EMBEDDING_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=LSTM_LAYERS,
            dropout=DROPOUT_RATE,
            bidirectional=True
        )
        self.bridge = nn.Sequential(
            nn.Linear(
                in_features=HIDDEN_SIZE * 2,
                out_features=HIDDEN_SIZE * 2,
                bias=True
            ),
            nn.Linear(
                in_features=HIDDEN_SIZE * 2,
                out_features=HIDDEN_SIZE * 2,
                bias=True
            )
        )

    def forward(self, input):
        hidden_states, (hidden_state, cell_state) = self.lstm(input)
        last_hidden_state0 = torch.cat((hidden_state[0].unsqueeze(0), hidden_state[1].unsqueeze(0)), dim=2)
        last_hidden_state1 = torch.cat((hidden_state[2].unsqueeze(0), hidden_state[3].unsqueeze(0)), dim=2)
        last_hidden_state = torch.cat((last_hidden_state0, last_hidden_state1), dim=0)
        last_hidden_state = self.bridge(last_hidden_state)
        
        return hidden_states, last_hidden_state