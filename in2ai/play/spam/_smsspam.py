import pandas as pd
from urllib.request import urlopen
import urllib.parse
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import BernoulliNB
import urllib
from http import HTTPStatus
import http
import zipfile
from urllib.request import urlretrieve
from nltk.tokenize import word_tokenize
import en_core_web_sm
from nltk.corpus import stopwords
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from keras.models import Sequential
from keras import layers
from sklearn.model_selection import StratifiedKFold
import os.path
import os
from dotenv import load_dotenv
load_dotenv()
from ._core import StopWordRemovalTransformer
from ._core import LemmatizeTransformer
from ._core import DocEmbeddingVectorizer



URL_SMSSPAM = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/"
DATA_HOME = os.getenv("PATH_DATA")
PATH_UNZIP = DATA_HOME + '/smsspam'


def fetch_smsspam(data_home=DATA_HOME):
    """Load the Spambase dataset from the UCI ML repository.
    
    Parameters  
    ----------
    data_home: Path to download the files.

    Returns
    -------
    df : DataFrame with the attributes as described in the dataset docs.
    """
    if not os.path.exists(data_home + '/smsspamcollection.zip'):
        urlretrieve(URL_SMSSPAM, data_home + '/smsspamcollection.zip')
    df = pd.DataFrame(columns=['text', 'spam?'])
    with zipfile.ZipFile(data_home+'/smsspamcollection.zip', 'r') as zip_ref:
        zip_ref.extractall(PATH_UNZIP)
    file = PATH_UNZIP + '/SMSSpamCollection'
    colnames=['label', 'sms']
    data = pd.read_csv(file, sep="\t", header=None, names=colnames)
    data.head()
    return data

def create_pipeline_smsspam():
    """Creates a sample pipeline for smsspam
    """
    vecz = ('vect', CountVectorizer())
    logr = ('classifier', LogisticRegression())
    stop = ('stop', StopWordRemovalTransformer())
    lemma = ('lemma', LemmatizeTransformer())
    binz = ('binarizer', CountVectorizer())
    we = ('document embedding', DocEmbeddingVectorizer())
    sel = ('fsel', SelectKBest(score_func=mutual_info_classif, k=100))
    clf = ('cls', BernoulliNB()) # Binary features in the original paper. 
    return Pipeline([vecz,logr]),    \
        Pipeline([binz, sel, clf]),   \
            Pipeline([stop, binz, sel, clf]),  \
                Pipeline([lemma, binz, sel, clf]),     \
                    Pipeline([stop, lemma, binz, sel, clf]), \
                        Pipeline([stop, lemma, we, sel, clf])