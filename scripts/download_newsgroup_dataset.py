"""This script downloads the 20-newsgroup dataset and saves it into
our expect corpus format in the data/ subfolder.

"""
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_20newsgroups


OUTPUT_FOLDER = Path('corpus_explorer/data')
OUTPUT_FILEPATH = OUTPUT_FOLDER / '20newsgroup_dataset.parquet'

dataset = fetch_20newsgroups(
    shuffle=True,
    random_state=1,
    remove=('headers', 'footers', 'quotes'),
    data_home=OUTPUT_FOLDER,
)

# Clean up the cache file that sklearn save
os.remove(OUTPUT_FOLDER / '20news-bydate_py3.pkz')

print(f'Saving dataset to {OUTPUT_FILEPATH}')
df = pd.DataFrame(
    {
        'text': dataset.data,
        'timestamp': datetime(2019, 1, 1),  # filler date
        'tags': [[dataset.target_names[i]] for i in dataset.target],
        'id': range(1, len(dataset.data) + 1),
    },
).set_index('id')
df.to_parquet(OUTPUT_FILEPATH)
