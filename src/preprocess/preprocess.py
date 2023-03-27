from constanst import *
from hyperparameters import *
from torchtext.vocab import build_vocab_from_iterator
import pandas as pd
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

def text_preprocessor(dataframe, vocab):
    dataframe['src'] = dataframe['src'].apply(lambda x: x.split())
    dataframe['tgt'] = dataframe['tgt'].apply(lambda x: x.split())
    dataframe['src'] = dataframe['src'].apply(lambda x: ['<sos>'] + x)
    dataframe['tgt'] = dataframe['tgt'].apply(lambda x: ['<sos>'] + x)
    dataframe['src'] = dataframe['src'].apply(lambda x: x + ['<eos>'])
    dataframe['tgt'] = dataframe['tgt'].apply(lambda x: x + ['<eos>'])
    # dataframe = dataframe[dataframe['src'].str.len() <= MAX_SRC_SEQ_LENGTH]
    # dataframe = dataframe[dataframe['tgt'].str.len() <= MAX_TGT_SEQ_LENGTH]
    dataframe['src'] = dataframe['src'].apply(lambda x: vocab(x))
    dataframe['tgt'] = dataframe['tgt'].apply(lambda x: vocab(x))
    return dataframe

def preprocess(
        src_dir: str,
        tgt_dir: str,
        vocab = None,
):
    # Read dataset .txt to list of one-line code
    src = open(DATA_DIR + src_dir, 'r').read().splitlines()
    tgt = open(DATA_DIR + tgt_dir, 'r').read().splitlines()

    # Create Vocab
    if vocab == None:
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
        # Set '<unk>' token for Out-of-Vocab token
        vocab.set_default_index(1)
    
    # Build pandas DataFrame
    ## Create json
    raw_data = {
            "src": [line for line in src],
            "tgt": [line for line in tgt],
        }
    ## Create DataFrame
    df = pd.DataFrame(raw_data)
    ## Preprocess the data in DataFrame
    df = text_preprocessor(df, vocab)

    # DataFrame to List
    src = df['src'].values.tolist()
    tgt = df['tgt'].values.tolist()
    src_tgt = list(zip(src, tgt))

    # List to Tensor
    for i, pair in enumerate(src_tgt):
        src, tgt = pair
        src = torch.tensor(src, dtype=torch.long, device=DEVICE)
        tgt = torch.tensor(tgt, dtype=torch.long, device=DEVICE)
        src_tgt[i] = (src, tgt)

    # function to collate data samples into batch tesors
    def collate_fn(batch):
        src_batch, tgt_batch = [], []
        
        for src_sample, tgt_sample in batch:
            src_batch.append(src_sample)
            tgt_batch.append(tgt_sample)

        src_batch = pad_sequence(src_batch, padding_value=0)
        tgt_batch = pad_sequence(tgt_batch, padding_value=0)
        return src_batch, tgt_batch

    dataset_loader = DataLoader(src_tgt, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    return vocab, dataset_loader