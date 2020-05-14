from .._spambase import fetch_spambase
from .._spambase import create_pipeline_spambase
from sklearn.model_selection import cross_validate
from sklearn.metrics import  make_scorer
from sklearn.metrics import accuracy_score
from .utils import tn
from .utils import tp
from .utils import fn
from .utils import fp

def test_fetch():
    df = fetch_spambase()
    assert((4601, 58) == df.values.shape)

def test_basemodel():
    df = fetch_spambase() #Caracteristicas como la frecuencia de palabras y demás es lo que convierten en spam o no spam al mensaje
    p = create_pipeline_spambase()
    #Matriz de características --> Selecciona toda la matriz excepto la ultima columna que es el objetivo es spam o no
    X = df.iloc[:, :-1].values 
    #Vector objetivo es spam o no --> Selecciona todas las filas de última columna
    y = df.iloc[:, -1:].values.astype('int').ravel()
    #y es el objetivo que intenta predecir en caso de aprendizaje supervisados
    
    scoring = {
        'accuracy': make_scorer(accuracy_score), 
        'prec': 'precision',
        'tp': make_scorer(tp), 
        'tn': make_scorer(tn),
        'fp': make_scorer(fp),
        'fn': make_scorer(fn)
    }
    trained = cross_validate(p, X, y, cv=10, scoring=scoring)   #https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html
    print(trained)

test_fetch()
test_basemodel()