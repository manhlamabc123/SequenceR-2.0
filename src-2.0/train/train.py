def per_iter():
    # Arguments: model, train_set, optimizer, criterion
    # Variables
    # Loop through train_set
    ## Source, Target
    ## Clear cache
    ## Forward
    ## Compute loss
    ## Update loss
    ## Backward
    ## Update parameters
    # Return final_loss
    pass

def train():
    # Arguments: model, train_set, val_set, learning_rate, epochs, print_every
    # Variables
    # Loss & Optimizer
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