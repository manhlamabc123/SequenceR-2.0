import torch.nn as nn
from encoder import RNNEncoder
from decoder import InputFeedRNNDecoder
from copy import CopyGenerator
from attention import GlobalAttention

class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.encoder = RNNEncoder(input_size)
        self.decoder = InputFeedRNNDecoder(input_size)
        self.generator = CopyGenerator(output_size)
        self.attention = GlobalAttention()

    def forward(self, input, target):
        output_encoder, hidden_state, cell_state = self.encoder(input)
        output_decoder = self.decoder(input)
        output_generator = self.generator()
        pass