import torch
import torch.nn as nn
import torch.optim as optim

from hyperparameters import *

def per_iter(optimizer: optim.Optimizer):
    # Arguments: model, train_set, optimizer, criterion
    # Variables
    # Loop through train_set
    for i in range(100):
    ## Source, Target
    ## Clear cache
        optimizer.zero_grad()
    ## Forward
    ## Compute loss
    ## Update loss
    ## Backward
    ## Update parameters
        optimizer.step()
    # Return final_loss
    pass

def train(model: nn.Module, learning_rate=LEARNING_RATE):
    # Arguments: model, train_set, val_set, learning_rate, epochs, print_every
    # Variables
    # Loss & Optimizer
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    # Training
    ## Start timer
    ## Make sure gradient tracking is on, and do a pass over the data
    ## Train per iteration
    ## Turn off gradient tracking cause it is not needed anymore
    ## Calculate Loss on val set
    ## Calculate Metric
    ## End timer
    ## Print timer, loss, metric for Train & Val
    ## Tracking best performance, and save the model's state
    # Return
    pass