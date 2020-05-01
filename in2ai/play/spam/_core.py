"""Common functions and transformers.
"""

from sklearn.base import TransformerMixin
from functools import partial
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
import spacy
import string
from sklearn.metrics.classification import accuracy_score

STOPLIST = set(stopwords.words('english') + list(ENGLISH_STOP_WORDS))
SYMBOLS = " ".join(string.punctuation).split(" ") + ["-", "...", "”", "”"]
from spacy.lang.en import English

def cleanText(text):
    text = text.strip().replace("\n", " ").replace("\r", " ")
    text = text.lower()
    return text

class StopWordRemovalTransformer(TransformerMixin):
    """Removes stop words and punctuation from text (English only).
    """ 
    def __init__(self):
        super().__init__()
        spacy.load('en_core_web_sm')
        self._nlp = English()

    def tokenizeText(self, sample):
        tokens = self._nlp(sample)
        tokens = [tok.text for tok in tokens if tok not in STOPLIST]
        tokens = [tok for tok in tokens if tok not in SYMBOLS]
        return ' '.join(tokens)

    def transform(self, X, **transform_params):
        X = [cleanText(text) for text in X] 
        return [self.tokenizeText(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

class LemmatizeTransformer(TransformerMixin):  
    """Transforms words to lemmatized form (English only).
    """ 
    def __init__(self):
        super().__init__()
        self._nlp = spacy.load('en_core_web_sm')


    def lemmatizeText(self, sample):
        tokens = self._nlp(sample)
        tokens = [tok.lemma_.lower().strip() for tok in tokens]
        return ' '.join(tokens)

    def transform(self, X, **transform_params):
        X = [cleanText(text) for text in X] 
        return [self.lemmatizeText(text) for text in X]
 

    def fit(self, X, y=None, **fit_params):
        return self


class DocEmbeddingVectorizer(TransformerMixin):  
    """Convert a collection of text documents to a matrix containing a document embedding.
    """ 
    def __init__(self):
       super().__init__()
       self._nlp = spacy.load("en_vectors_web_lg")

    def transform(self, X, **transform_params):
        X = [cleanText(text) for text in X]
        return [self._nlp(text).vector for text in X]
 

    def fit(self, X, y=None, **fit_params):
        return self



def weighted_accuracy(lam):
    """
    To make accuracy and error rate sensitive to this cost, 
    we treat each legitimate message as if it were TIMES messages: 
    when a legitimate message is misclassified, this counts as TIMES errors.
    weighted_accuracy(df.spam.apply(lambda x: TIMES if x.spam else 1))
    TODO: Check effects. 
    """
    return partial(accuracy_score, weights=lam)

def tcr():
    """TODO: Total cost ratio (TCR)
    """
    pass
