from .._lingspam import fetch_lingspam
from .._lingspam import create_pipelines_lingspam
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
import numpy as np

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
    print(pipelines)
    for p in pipelines:  
        scores = cross_val_score(p, X, y, cv=3) # Reduce cv folds for quicker testing.   
        print(scores, np.mean(scores))

test_fetch()
test_basemodel()
   
    