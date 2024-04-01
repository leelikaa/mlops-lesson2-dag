import sklearn
from sklearn.linear_model import LogisticRegression
import joblib
import pandas as pd

def train(train_csv: str) -> str:

    """Обучение модели логистической регрессии на тренировочной выборке и сохранение модели."""
    df_iris_train = (
        pd.read_csv(train_csv, index_col = 0))
    
    X = df_iris_train.drop('target', axis=1)
    Y = df_iris_train['target']
    
    modelLR = LogisticRegression()
    modelLR.fit(X,Y)
    
    return joblib.dump(modelLR, '/app/model.pkl')
