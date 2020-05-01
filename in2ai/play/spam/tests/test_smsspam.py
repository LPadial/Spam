from .._smsspam import fetch_smsspam
from .._smsspam import create_pipeline_smsspam
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder 

def test_fetch():
    df = fetch_smsspam()
    print(df.head())
    assert((5572, 2) == df.values.shape)


def test_smsmodel():
    df = fetch_smsspam() #Caracteristicas como la frecuencia de palabras y demás es lo que convierten en spam o no spam al mensaje
    p = create_pipeline_smsspam()
    le = LabelEncoder()
    df['target']= le.fit_transform(df['label']) 
    #Matriz de características --> Selecciona toda la matriz excepto la ultima columna que es el objetivo es spam o no
    X = df['sms'].values 
    #Vector objetivo es spam o no --> Selecciona todas las filas de última columna
    y = df['target'].values
    #y es el objetivo que intenta predecir en caso de aprendizaje supervisados
    scores = cross_val_score(p, X, y, cv=6, scoring='accuracy')   #https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html
    print(scores)

test_fetch()
test_smsmodel()