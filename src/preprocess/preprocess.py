from constanst import *
from hyperparameters import *
from torchtext.vocab import build_vocab_from_iterator
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
        src_dir: str,
        tgt_dir: str,
        data_dir: str,
        max_src_seq_length: int = 192,
        max_tgt_seq_length: int = 192,
        max_vocab: int = 0
):
    ## Read dataset .txt to list of one-line code
    src = open(DATA_DIR + src_dir, 'r').read().splitlines()
    tgt = open(DATA_DIR + tgt_dir, 'r').read().splitlines()

    tokenize = lambda x: x.split()

    def yield_tokens():
        for line in src:
            tokens = tokenize(line)
            yield tokens

    token_generator = yield_tokens()

    vocab = build_vocab_from_iterator(
            token_generator, 
            max_tokens=max_vocab,
            specials=['<pad>', '<unk>', '<sos>', '<eos>']
        )
    
    # print(vocab.get_stoi())
    print(len(vocab.get_itos()))
    print(vocab.__len__())