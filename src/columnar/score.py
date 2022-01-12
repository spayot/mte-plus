from typing import List, Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.base import clone

from . import model, report

class Scorer:
    """defines set of scoring functions to apply."""
    def __init__(self, **kwargs):
        self.scoring_fcts = kwargs
            
    def score(self, y_test: np.ndarray, y_preds: np.ndarray) -> dict[str, float]:
        """Calculates each scores given 2 arrays of results."""
        return {label: fct(y_test, y_preds) for label, fct in self.scoring_fcts.items()}
    
    def __repr__(self) -> str:
        return f"Scorer(scoring_fcts=[{','.join(self.scoring_fcts.keys())}])"
    
    
    
def cv_score(pipeline: model.CategoricalPipeline,
             data: pd.DataFrame, 
             kf: Optional[KFold] = None,
             scorer: Optional[Scorer] = None, 
             reporter: Optional[report.Report] = None,
             show: bool = True) -> pd.DataFrame:
    """
    evaluates cross-validation scores for a given pipeline on a given dataset.
    Note: reporter scorer supersedes the scorer parameter if present."""
    
    pipe = pipeline.clone()
    if kf is None:
        kf = KFold(n_splits=5)
    
    report = []

    if reporter is not None:
        scorer = reporter.scorer

    for train_idx, test_idx in kf.split(data):
        # reinstantiate pipeline
        pipeline_cv = pipeline.clone()
        # split train and test data using the CV fold
        cv_train, cv_test = data.iloc[train_idx], data.iloc[test_idx]

        # train a pipeline on cross validation fold
        pipeline_cv.fit(cv_train)

        # score on validation set
        preds = pipeline_cv.predict(cv_test)
        
        # get groundtruth values
        _, y_test = pipeline_cv.features.select_features(cv_test)
        
        # calculate scores
        report.append(scorer.score(y_test, preds))


    results = pd.DataFrame(report)

    if reporter is not None:
        reporter.add_to_report(pipeline_cv.config, results.mean())

        if show:
            display(reporter.show())

    return results
    
    
    
def fit_and_score(self, 
                  kf: KFold, 
                  data: pd.DataFrame, 
                  scorer: Optional[Scorer] = None, 
                  reporter: Optional[report.Report] = None,
                  show: bool = True) -> pd.DataFrame:
    """

    Note: reporter scorer supersedes the scorer parameter if present."""

    report = []

    if reporter:
        scorer = reporter.scorer

    for train_idx, test_idx in kf.split(data):
        # split train and test data using the CV fold
        cv_train, cv_test = data.iloc[train_idx], data.iloc[test_idx]

        # train a pipeline on cross validation fold
        pipe = self.train_model(cv_train)

        # score on validation set
        X_test, y_test = self.features.select_features(cv_test)
        preds = pipe.predict(X_test)

        report.append(scorer.score(y_test, preds))


    results = pd.DataFrame(report)

    if reporter is not None:
        reporter.add_to_report(self.config, results.mean())

        if show:
            display(reporter.show())

    return results