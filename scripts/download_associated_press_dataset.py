import os
import re
import requests
import tarfile
from datetime import datetime

import pandas as pd


url = 'http://www.cs.columbia.edu/~blei/lda-c/ap.tgz'
TEMP_FILE = 'temp_file.tgz'

with open(TEMP_FILE, 'wb') as f:
    f.write(requests.get(url).content)

raw_text = tarfile.open(TEMP_FILE).extractfile('ap/ap.txt').read().decode('utf-8')

# group 1 is YYMMDD date, group 2 is the text itself
pattern = r'<DOCNO> AP(.+?)-\d+ </DOCNO>\s+<TEXT>(.+?)</TEXT>'
dates, texts = zip(*re.findall(pattern, raw_text, flags=re.DOTALL))
dates = list(dates)

for i, date in enumerate(dates):
    year, month, day = 1900 + int(date[:2]), int(date[2:4]), int(date[4:6])
    dates[i] = datetime(year, month, day)

texts = [x.strip() for x in texts]

df = pd.DataFrame(
    {
        'text': texts,
        'timestamp': dates,
        'tags': None,
    },
)

df.to_parquet('corpus_explorer/data/ap_dataset.parquet')

os.remove(TEMP_FILE)
