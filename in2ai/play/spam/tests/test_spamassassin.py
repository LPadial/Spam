from .._spamassassin import fetch_spamassassin
from .._spamassassin import create_pipelines_spamassassin
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
import numpy as np

def test_fetch():
    try:
        df = fetch_spamassassin()
    except Exception as e:
        print(e)
    assert((10751, 2) == df.values.shape)

def test_basemodel():
    """Test the sample pipelines for LingSpam.
    """ 
    df = fetch_spamassassin().sample(50) # add somthng like ".sample(100)" for testing with less data.
    print(df.head())
    pipelines = create_pipelines_spamassassin()
    X = df['text'].values
    y = df['spam?'].values.astype('int')
    for p in pipelines:  
        scores = cross_val_score(p, X, y, cv=3) # Reduce cv folds for quicker testing.   
        print(scores, np.mean(scores))


test_fetch()
test_basemodel()