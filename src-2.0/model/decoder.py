import torch
import torch.nn as nn
from hyperparameters import *

class InputFeedRNNDecoder(nn.Module):
    def __init__(self, output_size):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=output_size,
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
                in_features=HIDDEN_SIZE * 4,
                out_features=HIDDEN_SIZE * 2
            ),
            nn.Linear(
                in_features=HIDDEN_SIZE * 2,
                out_features=HIDDEN_SIZE * 2
            )
        )
        self.softmax = nn.Softmax(dim=0)
        self.linear = nn.Linear(
            in_features=HIDDEN_SIZE * 2,
            out_features=output_size
        )
        self.log_softmax = nn.LogSoftmax(dim=2)

    def forward(self, input, decoder_hidden_state, cell_state, encoder_hidden_states):
        # Get input's sequence length
        sequence_length = encoder_hidden_states.shape[0]

        # Embedding & Dropout
        output_embedding = self.dropout(self.embedding(input))

        # Attention
        ## Expand decoder_hidden_state
        expanded_decoder_hidden_state = decoder_hidden_state.repeat(sequence_length, 1, 1)
        ## Concat expanded_decoder_hidden_state & encoder_hidden_states
        hidden_states = torch.cat((expanded_decoder_hidden_state, encoder_hidden_states), dim=2)
        ## Calculate attention_distribution
        attention_score = self.global_attention(hidden_states)
        attention_distribution = self.softmax(attention_score)
        ## Calculate context_vector
        context_vector = torch.bmm(attention_distribution.permute(1, 2, 0), encoder_hidden_states.permute(1, 0, 2)) 
        ## Concat context_vector & embedding
        lstm_input = torch.cat((context_vector.permute(1, 0, 2), output_embedding), dim=2)

        # LSTM
        (decoder_hidden_state, cell_state) = self.stacked_lstm(lstm_input, (decoder_hidden_state, cell_state))

        # Vocabulary Distribution
        vocabulary_distribution = self.log_softmax(self.linear(decoder_hidden_state))

        return decoder_hidden_state, cell_state, attention_distribution, vocabulary_distribution