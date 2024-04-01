import pandas as pd
import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from typing import List

def load_data() -> str:

    """Загрузка датасета Iris."""
    iris = load_iris()
    iris_data = iris.data
    iris_feature_names = iris.feature_names
    iris_target = iris.target
    #iris_target_names = iris.target_names
    
    data = pd.DataFrame(iris_data, columns=iris_feature_names)
    data['target'] = iris_target
    
    #data['target_names'] = data['target'].map({i: name for i, name in enumerate(iris_target_names)})
    
    path = '/app/dataset/iris.csv'
    
    return data.to_csv(path, sep=',',index=False)


def prepare_data(csv_path: str) -> List[str]:

    """Чтение загруженного датасета и разделение на train и test выборки."""
    df_iris = (pd.read_csv(csv_path))
    
    X_train, X_test, y_train, y_test = train_test_split(df_iris.drop('target', axis=1), 
                                                    df_iris['target'], 
                                                    test_size=0.2)
    X_train['target'] = y_train
    X_test['target'] = y_test
    
    path_train = '/app/dataset/iris_train.csv'
    path_test = '/app/dataset/iris_test.csv'
    
    X_train.to_csv(path_train, sep=',',index=True)
    X_test.to_csv(path_test, sep=',',index=True)
    
    return