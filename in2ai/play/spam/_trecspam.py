"""https://trec.nist.gov/data/spam.html 
https://trec.nist.gov/pubs/trec16/papers/SPAM.OVERVIEW16.pdf
https://plg.uwaterloo.ca/cgi-bin/cgiwrap/gvcormac/foo07

Los subdirectorios dalay full y partial indican que mensajes son spam y cuales no (ham)
En el subdirectorio data se encuentran los emails todos en archivos separados con formato de nombre {inmail.[d[3,4,5]]}
"""
import tarfile
from urllib.request import urlretrieve
from ._core import StopWordRemovalTransformer
from ._core import LemmatizeTransformer
from ._core import DocEmbeddingVectorizer


import pandas as pd
import os
import os.path
from dotenv import load_dotenv
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_extraction.text import CountVectorizer
import re

load_dotenv()

folders = ['full', 'partial', 'delay']
PATH_DATA = os.getenv("PATH_DATA")


def fetch_trecspam(data_home = PATH_DATA):
    """
    ----------
    data_home: Path to download the files.

    Returns
    -------
    df : DataFrame with the following attributes:
        - text: The text of the message.
        - spam?: Wheter the message is spam or not. 
    """
    category_email = []
    df = pd.DataFrame(columns=['text', 'spam?'])
    with tarfile.open(mode="r:gz", name=data_home+'/trec07p.tgz') as f:
        m = f.extractall(data_home)
        for folder in folders:
            with open(data_home + '/trec07p/' + folder + '/index', 'r') as ifile:
                content = ifile.readlines()
            for line in content:
                match = re.search(r'^((?:sp|h)am) .*?inmail\.(\d{1,5})$', line.lower())
                if match:
                    category = match.group(1)
                    email_num = match.group(2)
                    filename = data_home + '/trec07p/data/inmail.'+email_num
                    filecontent = open(filename, 'r', encoding='ISO-8859-1')
                    df = df.append({'text':filecontent.read(), 
                            'spam?':1 if category=='spam' else 0}, 
                            ignore_index=True)
    return df

def create_pipelines_trecspam():
    stop = ('stop', StopWordRemovalTransformer())
    lemma = ('lemma', LemmatizeTransformer())
    binz = ('binarizer', CountVectorizer())
    we = ('document embedding', DocEmbeddingVectorizer())
    sel = ('fsel', SelectKBest(score_func=mutual_info_classif, k=100))
    clf = ('cls', BernoulliNB()) # Binary features in the original paper. 
    return Pipeline([binz, sel, clf]),   \
           Pipeline([stop, binz, sel, clf]),  \
           Pipeline([lemma, binz, sel, clf]),     \
           Pipeline([stop, lemma, binz, sel, clf]), \
           Pipeline([stop, lemma, we, sel, clf])