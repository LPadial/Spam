"""
https://spamassassin.apache.org/old/publiccorpus/

An analysis of Spamassassin and BogoFilter.
Chiarella, J. (2003). An Analysis of Spam Filters (Doctoral dissertation, WORCESTER POLYTECHNIC INSTITUTE).

Datasets de correos electrónicos todos los comprimidos con nombre spam se trata de spam y los demás no (ham)
Los nombres están en formato {4 dígitos secuenciales}.{digitos aleatorios}
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

load_dotenv()

URL_SPAMASSASSIN = 'https://spamassassin.apache.org/old/publiccorpus/'
COMPRESS_FILES = ['20021010_easy_ham.tar.bz2', '20021010_hard_ham.tar.bz2', '20021010_spam.tar.bz2',
                    '20030228_easy_ham.tar.bz2', '20030228_easy_ham_2.tar.bz2', '20030228_hard_ham.tar.bz2',
                    '20030228_spam.tar.bz2', '20030228_spam_2.tar.bz2', '20050311_spam_2.tar.bz2']
PATH_DATA = os.getenv("PATH_DATA")


def fetch_spamassassin(data_home= PATH_DATA + '/spam_assassin/'):
    """
    ----------
    data_home: Path to download the files.

    Returns
    -------
    df : DataFrame with the following attributes:
        - text: The text of the message.
        - spam?: Wheter the message is spam or not. 
    """
    df = pd.DataFrame(columns=['text', 'spam?'])
    for i in COMPRESS_FILES:
        if not os.path.exists(data_home + COMPRESS_FILES[0]):
                urlretrieve(URL_SPAMASSASSIN + i, data_home + i) 
        with tarfile.open(mode="r:bz2", name=data_home+i) as f:
            folder = f.getnames()[0]
            emails = f.getnames()[1:]
            files = [name for name in emails if name.startswith(folder)]
            for name in files:
                m = f.extractfile(name)
                df = df.append({'text':str(m.read(), 'latin-1'), 
                    'spam?':1 if i.find('spam') != -1 else 0}, 
                    ignore_index=True)
    return df

def create_pipelines_spamassassin():
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