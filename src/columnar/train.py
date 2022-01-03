from typing import Callable, List, Optional

import pandas as pd
from sklearn.model_selection import KFold
from sklearn.compose import ColumnTransformer

from . import mte

from sklearn.pipeline import Pipeline

class Config(object):
    def __init__(self, categoricals: List[str], numericals: List[str],
                 target: str, model=None, encoder=None):
        self.model = model
        self.categoricals = categoricals
        self.numericals = numericals
        self.target = target
        self.encoder = encoder
        
    def set_model(self, model) -> None:
        self.model = model
        
    def set_encoder(self, encoder) -> None:
        self.encoder = encoder

    def __repr__(self):
        return f"Config({self.__dict__})"


def fit_and_score(config: Config, kf: KFold, data: pd.DataFrame, reporting: Callable, 
                  **kwargs):
    """performs mean target encoding on each fold, then fit a sklearn model"""
    SUFFIX = '_'
    categoricals_encoded = [col + SUFFIX for col in config.categoricals]
    report = []
    all_cols = config.categoricals + config.numericals
    for train_idx, test_idx in kf.split(data):
        cv_train, cv_test = data.iloc[train_idx][all_cols], data.iloc[test_idx][all_cols]
        y_train, y_test = data.iloc[train_idx][config.target], data.iloc[test_idx][config.target]

        
        # 2. passthrough the numerical columns
        trans = ColumnTransformer([('categories', config.encoder, config.categoricals)],
                                  remainder='passthrough')
        pipe = Pipeline(steps=[
            ('transform', trans),
            ('clf', config.model)
        ])

        pipe.fit(cv_train, y_train)

        # 4. score on validation set
        preds = pipe.predict(cv_test)

        report.append(reporting(y_test, preds))

    results = pd.DataFrame(report)

    return pipe, results

