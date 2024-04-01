from sklearn.metrics import classification_report
import json
from sklearn.metrics import accuracy_score
import joblib
import pandas as pd
import sklearn

def test(model_path: str, test_csv: str) -> str:

    """Тестирование модели на тестовой выборке и сохранение результатов."""
    model = joblib.load(model_path)
    
    df_iris_test = (pd.read_csv(test_csv, index_col = 0))
    
    x = df_iris_test.drop('target', axis=1)
    y = df_iris_test['target']

    report = classification_report(y, model.predict(x), output_dict=True)
    accuracy = accuracy_score(y, model.predict(x))
    
    metrics = {'accuracy': accuracy, 'report': report}
    
    with open('/app/model_metrics.json', 'w') as f:
        json.dump(metrics, f)
    
    return