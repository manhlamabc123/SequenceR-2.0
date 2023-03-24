from constanst import *
from hyperparameters import *
from torchtext.vocab import build_vocab_from_iterator
import pandas as pd
from torch.utils.data import DataLoader
from preprocess.my_dataset import MyDataset

def text_preprocessor(dataframe):
    dataframe['src'] = dataframe['src'].apply(lambda x: x.split())
    dataframe['tgt'] = dataframe['tgt'].apply(lambda x: x.split())
    # dataframe = dataframe[dataframe['src'].str.len() <= MAX_SRC_SEQ_LENGTH]
    # dataframe = dataframe[dataframe['tgt'].str.len() <= MAX_TGT_SEQ_LENGTH]
    return dataframe

def preprocess(
        src_dir: str,
        tgt_dir: str,
        create_vocab = False
):
    # Read dataset .txt to list of one-line code
    src = open(DATA_DIR + src_dir, 'r').read().splitlines()
    tgt = open(DATA_DIR + tgt_dir, 'r').read().splitlines()

    # Create Vocab
    if create_vocab:
        # Tokenizer
        tokenize = lambda x: x.split()

        # Create Vocab. How to use build_vocab_from_iterator
        def yield_tokens():
            for line in src:
                tokens = tokenize(line)
                yield tokens

        token_generator = yield_tokens()

        vocab = build_vocab_from_iterator(
                token_generator, 
                max_tokens=SRC_VOCAB_THRESHOLD,
                specials=['<pad>', '<unk>', '<sos>', '<eos>']
            )
    
    # Build pandas DataFrame
    ## Create json
    raw_data = {
            "src": [line for line in src],
            "tgt": [line for line in tgt],
        }
    ## Create DataFrame
    df = pd.DataFrame(raw_data)
    ## Preprocess the data in DataFrame
    df = text_preprocessor(df)

    dataset = MyDataset(df)
    dataset_loader = DataLoader(dataset, batch_size=BATCH_SIZE)
    for data in dataset_loader:
        src, tgt = data
        print(src, tgt)
        break

    if create_vocab:
        return vocab
    else:
        return