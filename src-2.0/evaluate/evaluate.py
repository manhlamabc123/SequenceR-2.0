import torch.nn as nn

from constanst import *

def evaluate(data_set, model: nn.Module, criterion: nn.NLLLoss):
    # Initialize some variables
    final_loss = 0
    current_loss = 0

    for i, data in enumerate(data_set):
        # Extract data from dataset_loader
        input, target = data # (length, BATCH_SIZE)
        input, target = input.to(DEVICE), target.to(DEVICE)
        
        # Foward
        output = model(input, target) # (length, BATCH_SIZE, target's dictionary_length)

        # Compute loss
        loss = criterion(output.view(-1, output.shape[2]), target.view(target.shape[0] * target.shape[1]))

        # Update loss
        current_loss += loss.item()

    final_loss = current_loss / len(data_set)
    return final_loss