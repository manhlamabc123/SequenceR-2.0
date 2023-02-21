import torch.nn as nn
from hyperparameters import *

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
        self.linear = nn.Linear(
            in_features=HIDDEN_SIZE * 2,
            out_features=HIDDEN_SIZE * 2,
            bias=True
        )

    def forward(self, input):
        output_embedding = self.embedding(input)
        output_lstm, (hidden_state, cell_state) = self.lstm(output_embedding)
        output_linear = self.linear(self.linear(output_lstm))
        
        return output_linear, hidden_state