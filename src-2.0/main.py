import argparse
import torch

from preprocess.preprocess import handling_corpus, preprocess

parser = argparse.ArgumentParser(description="This is just a description")
parser.add_argument('-m', '--model', action='store', help="model's name", required=False)
group = parser.add_mutually_exclusive_group()
group.add_argument('-d', '--data', action='store_true', help='data preprocessing')
group.add_argument('-t', '--train', action='store_true', help='train model')
group.add_argument('-e', '--evaluate', action='store_true', help='evalute model')
args = parser.parse_args()

if args.data:
    print("> Processing Data...\n")

    handling_corpus(
        src_dir="src-train.txt",
        tgt_dir="tgt-train.txt",
        output_file_name="train"
    )

    handling_corpus(
        src_dir="src-val.txt",
        tgt_dir="tgt-val.txt",
        output_file_name="val"
    )

    handling_corpus(
        src_dir="src-test.txt",
        tgt_dir="tgt-test.txt",
        output_file_name="test"
    )

    # preprocess(
    #     max_src_seq_length=1010,
    #     max_tgt_seq_length=100,
    #     src_vocab_threshold=1000,
    #     tgt_vocab_threshold=1000
    # )

    print("> Done!\n")

if args.train:
    print("> Training...\n")

    print("> Load dataset...\n")

    print("> Initialize model...\n")

    print("> Done!\n")

if args.evaluate:
    print("> Evaluating...\n")

    print("> Load dataset...\n")

    print("> Initialize model...\n")

    print("> Load pre-trained model...\n")
        
    # Result