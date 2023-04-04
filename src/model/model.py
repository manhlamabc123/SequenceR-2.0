import torch
import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder
from hyperparameters import *
from constanst import *

class Model(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=EMBEDDING_SIZE,
            padding_idx=EMBEDDING_PAD
        )
        self.encoder = Encoder(vocab_size)
        self.decoder = Decoder(vocab_size)

    def forward(self, input, target=None):
        # Get target length
        if target is None:
            target_sequence_length = MAX_TGT_SEQ_LENGTH
        else:
            target_sequence_length = target.shape[0]

        # Embedding
        encoder_embedding = self.embedding(input)
        decoder_embedding = self.embedding(target)

        # Encoder
        encoder_hidden_states, encoder_last_hidden_state = self.encoder(encoder_embedding)

        # Decoder
        ## Init decoder_hidden_state as encoder_last_hidden_state
        decoder_hidden_state = encoder_last_hidden_state
        ## Init decoder_cell_state
        decoder_cell_state = torch.zeros(decoder_hidden_state.shape, device=DEVICE)
        ## Loop through target
        for i in range(target_sequence_length):
            ## Init decoder_input
            decoder_input = decoder_embedding[i].unsqueeze(dim=0)
            ## Forward through Decoder
            final_distribution, decoder_hidden_state, decoder_cell_state = self.decoder(decoder_input, decoder_hidden_state, decoder_cell_state, encoder_hidden_states)
            
        return final_distribution    