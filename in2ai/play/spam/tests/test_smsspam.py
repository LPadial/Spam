from .._smsspam import fetch_smsspam
from .._smsspam import create_pipeline_smsspam
from sklearn.preprocessing import LabelEncoder 
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.metrics import  make_scorer
from sklearn.metrics import accuracy_score
from .utils import tn
from .utils import tp
from .utils import fn
from .utils import fp

def test_fetch():
    df = fetch_smsspam()
    print(df.head())
    assert((5572, 2) == df.values.shape)


def test_smsmodel():
    df = fetch_smsspam() #Caracteristicas como la frecuencia de palabras y demás es lo que convierten en spam o no spam al mensaje
    pipelines = create_pipeline_smsspam()
    le = LabelEncoder()
    df['target']= le.fit_transform(df['label']) 
    #Matriz de características --> Selecciona toda la matriz excepto la ultima columna que es el objetivo es spam o no
    X = df['sms'].values 
    #Vector objetivo es spam o no --> Selecciona todas las filas de última columna
    y = df['target'].values
    #y es el objetivo que intenta predecir en caso de aprendizaje supervisados
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
test_smsmodel()