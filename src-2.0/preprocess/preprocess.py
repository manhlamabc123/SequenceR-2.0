from constanst import *
import pandas as pd

def preprocess(
        src_dir: str,
        tgt_dir: str,
        max_src_seq_length: int = 192,
        max_tgt_seq_length: int = 192,
        src_vocab_threshold: int = 0,
        tgt_vocab_threshold: int = 0
):
    #----------Load raw dataset----------
    ## Read dataset (.txt to list of one-line code)
    src = open(DATA_DIR + src_dir, 'r').read().splitlines()
    tgt = open(DATA_DIR + tgt_dir, 'r').read().splitlines()
    ## Convert to pandas.DataFrame
    raw_data = {
        "src": [line for line in src],
        "tgt": [line for line in tgt],
    }
    data_frame = pd.DataFrame(raw_data, columns=["src", "tgt"]) # [33798 rows x 2 columns]
    #----------------------------------------
    
    #----------Tokenization----------
    ## Element: String -> List of String (Token)
    data_frame["src"] = data_frame["src"].apply(lambda element: element.split(" "))
    data_frame["tgt"] = data_frame["tgt"].apply(lambda element: element.split(" "))
    ## Filter all sequence with more than max_seq_length
    data_frame = data_frame[ data_frame["src"].str.len() <= max_src_seq_length ]
    data_frame = data_frame[ data_frame["tgt"].str.len() <= max_tgt_seq_length ] # [33469 rows x 2 columns]
    #----------------------------------------

    #----------Create vocabulary----------
    #----------------------------------------

    #----------Numericalize----------
    #----------------------------------------

    #----------Debug----------
    print(data_frame)
    #----------------------------------------