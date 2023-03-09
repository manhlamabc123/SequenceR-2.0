from constanst import *
from hyperparameters import *
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
        data_dir: str,
        max_src_seq_length: int = 192,
        max_tgt_seq_length: int = 192,
        max_vocab: int = 0
):
    tokenize = lambda x: x.split()

    source = Field(
        sequential=True,
        use_vocab=True,
        init_token='<SOS>',
        eos_token='<EOS>',
        tokenize=tokenize,
    )
    target = Field(
        sequential=True,
        use_vocab=True,
        init_token='<SOS>',
        eos_token='<EOS>',
        tokenize=tokenize,
    )

    fields = {
        'src': ('src', source),
        'tgt': ('tgt', target),
    }

    train_data, val_data, test_data = TabularDataset.splits(
        path=data_dir,
        train='train.json',
        validation='val.json',
        test='test.json',
        format='json',
        fields=fields
    )

    print('> Train: ', len(train_data))
    print('> Validation: ', len(val_data))
    print('> Test: ', len(test_data))

    source.build_vocab(
        train_data,
        max_size=max_vocab
    )
    target.vocab = source.vocab

    train_iterator, val_iterator, test_iterator = BucketIterator.splits(
        (train_data, val_data, test_data),
        batch_size=BATCH_SIZE,
        device=DEVICE
    )

    return train_iterator, val_iterator, test_iterator, source