from constanst import *
import pandas as pd

def preprocess(src_dir: str, tgt_dir: str) -> None:
    # Read dataset (.txt to list of one-line code)
    src = open(DATA_DIR + src_dir, 'r').read().splitlines()
    tgt = open(DATA_DIR + tgt_dir, 'r').read().splitlines()

    # Convert to pandas
    raw_data = {
        "src": [line for line in src],
        "tgt": [line for line in tgt],
    }
    data_frame = pd.DataFrame(raw_data, columns=["src", "tgt"])
    print(data_frame.head(5))