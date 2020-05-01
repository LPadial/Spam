"""A dataset that contains spam messages and messages from the Linguist list. 
The dataset is available from the pages of the NLP group of the AUEB:
http://nlp.cs.aueb.gr/software.html
"""
import tarfile
from urllib.request import urlretrieve
from ._core import StopWordRemovalTransformer
from ._core import LemmatizeTransformer
from ._core import DocEmbeddingVectorizer


import pandas as pd
import os.path
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_extraction.text import CountVectorizer

URL_LINGSPAM = 'http://nlp.cs.aueb.gr/software_and_datasets/lingspam_public.tar.gz'


def fetch_lingspam(data_home='C:/Users/laura/Desktop/Universidad/TMA/data'):
    """Load the Ling-Spam data-set from AUEB (classification).
    Download it if necessary.
    
    Read more in the :ref:`User Guide <lingspam_dataset>`.
    Parameters  
    ----------
    data_home: Path to download the files.

    Returns
    -------
    df : DataFrame with the following attributes:
        - text: The text of the message.
        - spam?: Wheter the message is spam or not. 
    """
    if not os.path.exists(data_home + '/lingspam_public.tar.gz'):
        urlretrieve(URL_LINGSPAM, data_home + '/lingspam_public.tar.gz')
    df = pd.DataFrame(columns=['text', 'spam?'])
    with tarfile.open(mode="r:gz", name=data_home+'/lingspam_public.tar.gz') as f:
        # We load only the raw texts. 
        folder = 'lingspam_public/bare/'
        #cada subcarpeta de lingspam_public/bare/ tiene muchos txt con mensajes clasificados por el nombre del archivos
        files = [name for name in f.getnames() if name.startswith(folder) and name.endswith('.txt')]
        for name in files:
            m = f.extractfile(name)
            df = df.append({'text':str(m.read(), 'utf-8'), 
                            'spam?':1 if 'spmsg' in name else 0}, 
                            ignore_index=True)
    return df   

def create_pipelines_lingspam():
    """Reproduces the pipelines evaluated in the LingSpam paper.
    I. Androutsopoulos, J. Koutsias, K.V. Chandrinos, George Paliouras, 
    and C.D. Spyropoulos, "An Evaluation of Naive Bayesian Anti-Spam 
    Filtering". In Potamias, G., Moustakis, V. and van Someren, M. (Eds.), 
    Proceedings of the Workshop on Machine Learning in the New Information 
    Age, 11th European Conference on Machine Learning (ECML 2000), 
    Barcelona, Spain, pp. 9-17, 2000.

    Diferences: use of lemmatization instead of stemming. 
    """
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
               