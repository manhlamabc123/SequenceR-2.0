from constanst import *
from torchtext.data import Field, TabularDataset, BucketIterator
import json

def handling_corpus(src_dir: str, tgt_dir: str, output_file_name: str) -> None:
    ## Read dataset .txt to list of one-line code
    src = open(DATA_DIR + src_dir, 'r').read().splitlines()
    tgt = open(DATA_DIR + tgt_dir, 'r').read().splitlines()
    src_tgt = list(zip(src, tgt)) # Combines 2 lists into 1 lists of (source, target)
    ## From lists of (source, target) to lists of jsons
    raw_data = [
        {
            "src": example[0],
            "tgt": example[1]
        } for example in src_tgt
    ]
    ## From lists of jsons to .json
    with open(f'preprocess/preprocessed/{output_file_name}.json', 'w') as file:
        for item in raw_data:
            json.dump(item, file)
            file.write('\n')

def preprocess(
        max_src_seq_length: int = 192,
        max_tgt_seq_length: int = 192,
        src_vocab_threshold: int = 0,
        tgt_vocab_threshold: int = 0
):
    tokenize = lambda x: x.split()

    source = Field(
        sequential=True,
        use_vocab=True,
        tokenize=tokenize,
    )
    target = Field(
        sequential=True,
        use_vocab=True,
        tokenize=tokenize,
    )

    fields = {
        'src': ('src', source),
        'tgt': ('tgt', target),
    }

    # train_data, test_data = TabularDataset.splits(

    # )