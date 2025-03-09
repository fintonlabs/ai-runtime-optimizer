import pandas as pd
import numpy as np
from performance_analyzer import PerformanceAnalyzer

def test_preprocess_data():
    data = pd.DataFrame({
        'feature1': [1, 2, np.nan],
        'feature2': [4, np.nan, 6],
        'target': [7, 8, 9]
    })
    analyzer = PerformanceAnalyzer(data)
    analyzer.preprocess_data()
    assert not analyzer.data.isnull().values.any()

def test_train_model():
    data = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6],
        'target': [7, 8, 9]
    })
    analyzer = PerformanceAnalyzer(data)
    analyzer.train_model()
    assert analyzer.model is not None

def test_predict():
    data = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6],
        'target': [7, 8, 9]
    })
    analyzer = PerformanceAnalyzer(data)
    analyzer.train_model()
    predictions = analyzer.predict(pd.DataFrame([[1, 2]], columns=['feature1', 'feature2']))
    assert len(predictions) == 1