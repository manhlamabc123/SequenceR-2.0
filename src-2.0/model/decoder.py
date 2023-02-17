import torch.nn as nn
from hyperparameters import *

class InputFeedRNNDecoder(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=input_size,
            embedding_dim=EMBEDDING_SIZE,
            padding_idx=EMBEDDING_PAD
        )
        self.dropout = nn.Dropout(
            p=0.3,
            inplace=False
        )
        self.stacked_lstm = nn.Sequential(
            nn.Dropout(
                p=0.3,
                inplace=False
            ),
            nn.LSTMCell(
                input_size=HIDDEN_SIZE * 4,
                hidden_size=HIDDEN_SIZE * 2
            ),
            nn.LSTMCell(
                input_size=HIDDEN_SIZE * 2,
                hidden_size=HIDDEN_SIZE * 2
            )
        )
        self.global_attention = nn.Sequential(
            nn.Linear(
                in_features=HIDDEN_SIZE * 2,
                out_features=HIDDEN_SIZE * 2
            ),
            nn.Linear(
                in_features=HIDDEN_SIZE * 4,
                out_features=HIDDEN_SIZE * 2
            )
        )

    def forward(self, input):
        output_embedding = self.embedding(input)
        output_dropout = self.dropout(output_embedding)
        output_stacked_lstm = self.stacked_lstm(output_dropout)
        output_attention = self.global_attention(output_stacked_lstm)

        return output_attention