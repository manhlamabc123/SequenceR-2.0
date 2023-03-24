import torch
import torch.nn as nn
import torch.optim as optim
from timeit import default_timer as timer

from hyperparameters import *
from constanst import *
from evaluate.evaluate import evaluate
from metrics.accuracy import accuracy
from metrics.perplexity import perplexity

def per_iter(train_set, model: nn.Module, optimizer: optim.Optimizer, criterion: nn.NLLLoss):
    # Arguments: train_set, model, optimizer, criterion
    
    # Variables
    loss_final = 0
    loss_current = 0

    # Loop through train_set
    for i, data in enumerate(train_set):
        ## Source, Target
        input, target = data

        ## Clear cache
        optimizer.zero_grad()

        ## Forward
        output = model(input, target)

        ## Compute loss
        loss = criterion(output, target)

        ## Update loss
        loss_current += loss.item()

        ## Backward
        loss.backward()

        ## Update parameters
        optimizer.step()

    # Return final_loss
    loss_final = loss_current / len(train_set)
    return loss_final

def train(model: nn.Module, train_set, val_set):
    # Arguments: model, train_set, val_set

    # Variables
    plot_loss_train = []
    plot_loss_val = []
    plot_accuracy_train = []
    plot_accuracy_val = []
    plot_perplexity_train = []
    plot_perplexity_val = []

    # Loss & Optimizer
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.NLLLoss()

    # Training
    for epoch in range(EPOCHS):
        ## Start timer
        start_time = timer()

        ## Make sure gradient tracking is on, and do a pass over the data
        model.train(True)

        ## Train per iteration
        train_loss = per_iter(train_set, model, optimizer, criterion)
        plot_loss_train.append(train_loss)

        ## Turn off gradient tracking cause it is not needed anymore
        model.train(False)

        ## Calculate Loss on val set
        val_loss = evaluate(val_set, model, criterion)
        plot_loss_val.append(val_loss)

        ## Calculate Metric
        accuracy_train = accuracy(train_set, model)
        accuracy_val = accuracy(val_set, model)
        perplexity_train = perplexity(train_set, model)
        perplexity_val = perplexity(val_set, model)
        plot_accuracy_train.append(accuracy_train)
        plot_accuracy_val.append(accuracy_val)
        plot_perplexity_train.append(perplexity_train)
        plot_perplexity_val.append(perplexity_val)

        ## End timer
        end_time = timer()

        ## Print timer, loss, metric for Train & Val
        if epoch % PRINT_EVERY == 0:
            print(f"- Loss       | Train: {train_loss:.4f} - Dev: {val_loss:.4f}")
            print(f"- Accuracy   | Train: {accuracy_train:.4f} - Dev: {accuracy_val:.4f}")
            print(f"- Perplexity | Train: {perplexity_train:.4f} - Dev: {perplexity_val:.4f}")
            print(f"- Epoch's time: {(end_time - start_time):.3f}s")

        ## Save
        if epoch % SAVE_EVERY == 0:
            torch.save(model.state_dict(), MODEL_PATH)