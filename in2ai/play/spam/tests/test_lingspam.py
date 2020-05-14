from .._lingspam import fetch_lingspam
from .._lingspam import create_pipelines_lingspam
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.metrics import  make_scorer
from sklearn.metrics import accuracy_score
from .utils import tn
from .utils import tp
from .utils import fn
from .utils import fp

def test_fetch():
    """Test fetching the LingSpam dataset.
    """
    try:
        df = fetch_lingspam()
    except Exception as e:
        print(e)
    assert((2893, 2) == df.values.shape)

def test_basemodel():
    """Test the sample pipelines for LingSpam.
    """ 
    df = fetch_lingspam().sample(50) # add somthng like ".sample(100)" for testing with less data.
    pipelines = create_pipelines_lingspam()
    X = df['text'].values
    y = df['spam?'].values.astype('int')
    scoring = {
        'accuracy': make_scorer(accuracy_score), 
        'prec': 'precision',
        'tp': make_scorer(tp), 
        'tn': make_scorer(tn),
        'fp': make_scorer(fp),
        'fn': make_scorer(fn)
    }
    for p in pipelines:  
        trained = cross_validate(p, X, y, cv=3, scoring=scoring) # Reduce cv folds for quicker testing.   
        print(trained)

test_fetch()
test_basemodel()
   
    