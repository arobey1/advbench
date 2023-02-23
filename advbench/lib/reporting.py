import os
import tqdm
import json
import pandas as pd
from typing import List

def load_record(path: str) -> List[dict]:
    """Load the JSON stored in a given path to a list."""

    records = []
    with open(path, 'r') as f:
        for line in f:
            records.append(json.loads(line[:-1]))
    return records

def load_sweep_dataframes(path, depth=1):

    records = []

    def add_record(results_path):
        try:
            records.append(pd.read_pickle(results_path))

        # want to ignore existing files 
        # (e.g., results.txt, args.json, etc.)
        except IOError:
            pass

    if depth == 0:
        results_path = os.path.join(path, f'selection.pd')
        add_record(results_path)

    elif depth == 1:
        for i, subdir in list(enumerate(os.listdir(path))):
            results_path = os.path.join(path, subdir, f'selection.pd')
            add_record(results_path)

    else:
        raise ValueError(f'Depth {depth} is invalid.')

    return pd.concat(records, ignore_index=True)