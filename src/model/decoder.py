import torch
import torch.nn as nn
from hyperparameters import *

class InputFeedRNNDecoder(nn.Module):
    def __init__(self, output_size):
        super().__init__()

        self.dropout = nn.Dropout(
            p=0.3,
            inplace=False
        )
        self.lstm = nn.LSTM(
            input_size=HIDDEN_SIZE * 4,
            hidden_size=HIDDEN_SIZE * 2,
            num_layers=LSTM_LAYERS,
            dropout=DROPOUT_RATE,
            bidirectional=False
        )
        self.global_attention = nn.Sequential(
            nn.Linear(
                in_features=HIDDEN_SIZE * 4,
                out_features=HIDDEN_SIZE * 2
            ),
            nn.Linear(
                in_features=HIDDEN_SIZE * 2,
                out_features=1
            )
        )
        self.softmax = nn.Softmax(dim=0)
        self.linear = nn.Linear(
            in_features=HIDDEN_SIZE * 2,
            out_features=output_size
        )
        self.log_softmax = nn.LogSoftmax(dim=2)

        self.p_gen_copy = nn.Linear(
            in_features=HIDDEN_SIZE * 2,
            out_features=1,
            bias=True
        )
        self.p_gen = nn.Linear(
            in_features=HIDDEN_SIZE * 2,
            out_features=output_size,
            bias=True
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, decoder_hidden_state, cell_state, encoder_hidden_states):
        # Get input's sequence length
        sequence_length = encoder_hidden_states.shape[0]

        # Dropout
        output_dropout = self.dropout(input)

        # Attention
        ## Expand decoder_hidden_state
        expanded_decoder_hidden_state = decoder_hidden_state[1].unsqueeze(dim=0).repeat(sequence_length, 1, 1)
        ## Concat expanded_decoder_hidden_state & encoder_hidden_states
        hidden_states = torch.cat((expanded_decoder_hidden_state, encoder_hidden_states), dim=2)
        ## Calculate attention_distribution
        attention_score = self.global_attention(hidden_states)
        attention_distribution = self.softmax(attention_score)
        ## Calculate context_vector
        context_vector = torch.bmm(attention_distribution.permute(1, 2, 0), encoder_hidden_states.permute(1, 0, 2)) 
        ## Concat context_vector & dropout
        lstm_input = torch.cat((context_vector.permute(1, 0, 2), output_dropout), dim=2)

        # LSTM
        decoder_hidden_states, (decoder_hidden_state, cell_state) = self.lstm(lstm_input, (decoder_hidden_state, cell_state))

        # Vocabulary Distribution
        vocabulary_distribution = self.log_softmax(self.linear(decoder_hidden_state[1].unsqueeze(dim=0)))

        # Calculate p_gen
        p_gen = torch.add(torch.add(self.p_gen(context_vector.permute(1, 0, 2)), self.p_gen(decoder_hidden_state[1].unsqueeze(dim=0))), self.p_gen(input))
        p_gen = self.sigmoid(p_gen)

        # Calculate final_distribution
        final_distribution = torch.add(torch.mul(p_gen, vocabulary_distribution), torch.mul((1 - p_gen), attention_distribution.permute(2, 1, 0)))

        return final_distribution, decoder_hidden_state, cell_state