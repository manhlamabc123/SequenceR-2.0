import torch.nn as nn
from hyperparameters import *
import torch

class RNNEncoder(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=input_size, 
            embedding_dim=EMBEDDING_SIZE, 
            padding_idx=EMBEDDING_PAD
        )
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
        # print('Input: ', input.size())
        output_embedding = self.embedding(input)
        # print('Embedding: ', output_embedding.size())
        hidden_states, (hidden_state, cell_state) = self.lstm(output_embedding)
        # print('Hidden state: ', hidden_state.size())
        last_hidden_state = torch.cat((hidden_state[2].unsqueeze(0), hidden_state[3].unsqueeze(0)), dim=2)
        # print('Last state: ', last_hidden_state.size())
        last_hidden_state = self.bridge(last_hidden_state)
        # print('Hidden state after bridge: ', last_hidden_state.size())
        
        return last_hidden_state