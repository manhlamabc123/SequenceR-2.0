import torch
import torch.nn as nn
from .encoder import RNNEncoder
from .decoder import InputFeedRNNDecoder
from .copy import CopyGenerator
from hyperparameters import *

class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.encoder = RNNEncoder(input_size)
        self.decoder = InputFeedRNNDecoder(input_size)
        self.generator = CopyGenerator(output_size)

    def forward(self, input, target=None):
        # Get target length
        if target == None:
            target_sequence_length = MAX_TGT_SEQ_LENGTH
        else:
            target_sequence_length = target.shape[0]

        # Encoder
        encoder_hidden_states, encoder_last_hidden_state = self.encoder(input)

        # Decoder
        decoder_hidden_state = encoder_last_hidden_state
        decoder_hidden_state = torch.zeros(decoder_hidden_state.shape)
        for i in range(target_sequence_length):
            decoder_hidden_state, decoder_cell_state = self.decoder(input, decoder_hidden_state, decoder_cell_state, encoder_hidden_states)

        output_generator = self.generator()
        pass