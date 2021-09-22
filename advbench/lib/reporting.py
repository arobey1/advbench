import os
import tqdm
import pandas as pd

def load_records(path):
    records = []
    for i, subdir in tqdm.tqdm(list(enumerate(os.listdir(path))), ncols=80, leave=False):
        results_path = os.path.join(path, subdir, 'results.pkl')

        # want to ignore existing results.txt file
        try:
            df = pd.read_pickle(results_path)
            records.append(df)
        except IOError:
            pass

    return pd.concat(records, ignore_index=True)