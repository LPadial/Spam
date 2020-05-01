"""A dataset that contains spam messages from emails at HP Labs
https://archive.ics.uci.edu/ml/datasets/spambase
"""

import pandas as pd
from urllib.request import urlopen
import urllib.parse
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import BernoulliNB
import urllib
from http import HTTPStatus
import http
URL_SPAMBASE = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/"


def fetch_spambase(data_home='data'):
    """Load the Spambase dataset from the UCI ML repository.
    
    Parameters  
    ----------
    data_home: Path to download the files.

    Returns
    -------
    df : DataFrame with the attributes as described in the dataset docs.
    """
    """h1 = http.client.HTTPSConnection('www.python.org')
    print(h1)"""
    columns = []
    with urlopen(URL_SPAMBASE + 'spambase.names') as f:
        content = f.readlines()
    for line in content:
        if str(line,'utf-8').startswith(('word_freq', 'char_freq', 'capital_run')):
            columns.append(str(line,'utf-8').split(':')[0]) 
    columns.append('spam?')
    df = pd.read_csv(URL_SPAMBASE + 'spambase.data', header=None)
    df.columns = columns
    return df

def create_pipeline_spambase():
    """Creates a sample pipeline for spambase
    """
    clf = ('cls', BernoulliNB()) # Has binary and frequencies. 
    return Pipeline([clf])