import argparse
import torch

from preprocess.preprocess import preprocess
from model.model import Model
from train.train import train
from constanst import *
from hyperparameters import *

parser = argparse.ArgumentParser(description="This is just a description")
parser.add_argument('-m', '--model', action='store', help="model's name", required=False)
group = parser.add_mutually_exclusive_group()
group.add_argument('-t', '--train', action='store_true', help='train model')
group.add_argument('-e', '--evaluate', action='store_true', help='evalute model')
args = parser.parse_args()

if args.train:
    print("> Processing data...\n")

    vocab, train_loader = preprocess(src_dir="src-train.txt", tgt_dir="tgt-train.txt")
    vocab, val_loader = preprocess(src_dir="src-val.txt", tgt_dir="tgt-val.txt", vocab=vocab)

    print("> Vocabulary size: ", vocab.__len__())
    print("> Size of train: ", len(train_loader) * BATCH_SIZE)
    print("> Size of validation: ", len(val_loader) * BATCH_SIZE)

    print("> Initialize model...")

    model = Model(vocab_size=vocab.__len__())
    model = torch.compile(model)

    print("> Training...")

    train(model, train_loader, val_loader)

    print("\n> Done!\n")

if args.evaluate:
    print("> Evaluating...\n")

    print("> Load dataset...\n")

    print("> Initialize model...\n")

    print("> Load pre-trained model...\n")
        
    # Result