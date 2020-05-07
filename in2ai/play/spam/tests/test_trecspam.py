from .._trecspam import fetch_trecspam
from .._trecspam import create_pipelines_trecspam
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
import numpy as np
import re

def test_fetch():
    #try:
    df = fetch_trecspam()
    print(df.values.shape)
    
    '''except Exception as e:
        print(e)
    assert((10751, 2) == df.values.shape)'''

def test_basemodel():
    """Test the sample pipelines for TrecSpam.
    """ 
    df = fetch_trecspam().sample(50) # add somthng like ".sample(100)" for testing with less data.
    print(df.head())
    pipelines = create_pipelines_trecspam()
    X = df['text'].values
    y = df['spam?'].values.astype('int')
    for p in pipelines:  
        scores = cross_val_score(p, X, y, cv=3) # Reduce cv folds for quicker testing.   
        print(scores, np.mean(scores))


test_fetch()
test_basemodel()