import os
import tqdm
import pandas as pd

def load_records(path, depth=1):
    records = []

    def add_record(results_path):
        try:
            df = pd.read_pickle(results_path)
            records.append(df)

        # want to ignore existing results.txt file
        except IOError:
            pass

    if depth == 0:
        results_path = os.path.join(path, 'results.pkl')
        add_record(results_path)

    elif depth == 1:
        for i, subdir in tqdm.tqdm(list(enumerate(os.listdir(path))), ncols=80, leave=False):
            results_path = os.path.join(path, subdir, 'results.pkl')
            add_record(results_path)

    else:
        raise ValueError(f'Depth {depth} is invalid.')


    return pd.concat(records, ignore_index=True)