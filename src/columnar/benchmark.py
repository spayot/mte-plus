"""allows to benchmark the performance of various transformer / classifier pairs
on a given train and test set."""

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator, clone
from sklearn.preprocessing import MaxAbsScaler

from .score import Scorer
from .report import Report
from .feature_selection import FeatureSelection
from .utils import convert_time
from .embeddings.wrapper import TFEmbeddingWrapper
from .embeddings.data import df_to_dataset


def _get_key(transformer: TransformerMixin, 
             classifier: BaseEstimator) -> str:
    return str(transformer) + ':' + str(classifier)


class Logger:
    """logs individual results from a CV test"""
    def __init__(self):
        self.reports = dict()
    
    def log_results(self, transformer: TransformerMixin, 
                    classifier: BaseEstimator, 
                    results: dict[str, float]) -> None:
        
        report_key = _get_key(transformer, classifier)
                
        if report_key not in self.reports:
            self.reports[report_key] = []
        
        self.reports[report_key].append(results)
        
    def get_report(self, transformer: TransformerMixin, 
                    classifier: BaseEstimator) -> pd.DataFrame:
        """"""
        report_key = _get_key(transformer, classifier)
        return pd.DataFrame(self.reports[report_key])

    
class BenchmarkRunner:
    def __init__(self, 
                 features: FeatureSelection,
                 transformers: list[TransformerMixin],
                 classifiers: list[BaseEstimator], 
                 scorer: Scorer,
                 logger: Logger = None):
        
        self.features = features
        self.transformers = transformers
        self.scaler = MaxAbsScaler()
        self.classifiers = classifiers
        self.logger = logger if logger is not None else Logger()
        self.scorer = scorer
        
    def run(self, 
            X_train: np.ndarray, y_train: np.ndarray, 
            X_test: np.ndarray, y_test: np.ndarray) -> None:
        """run transformer / scaler / classifier pipeline for each 
        combination of transformer / classifier defined when instantiating 
        the benchmark. to increase efficiency, transformations are only done once, 
        and classifiers are fitted on the same transformed dataset. 
        Results are logged into the object's logger.
        """
        
        for t in self.transformers:
            # initialize transformer
            transformer = clone(t)
            
            # fit on training data and transform
            transformer.fit(X_train, y_train)
            
            X_train_ = transformer.transform(X_train)
            X_test_ = transformer.transform(X_test)
            
            # rescale data
            X_train_ = self.scaler.fit_transform(X_train_)
            X_test_ = self.scaler.transform(X_test_)
            
            for clf in self.classifiers:
                classifier = clone(clf)
                
                # fit on transformed features
                classifier.fit(X_train_, y_train)
                
                # get predictions from pipeline on test set
                preds = classifier.predict_proba(X_test_)[:,1]
                
                # log results
                self.logger.log_results(transformer, classifier, 
                                        results=self.scorer.score(y_test, preds))
                
            # add an extra evaluation with original DNN for TFEmbedding Wrappers
            if transformer.__class__ == TFEmbeddingWrapper:
                # get predictions
                preds = transformer.model.predict(df_to_dataset(X_test, shuffle=False))
                
                # log results
                self.logger.log_results(transformer, 'DNN', 
                                        results=self.scorer.score(y_test, preds))
                
    def create_reporter(self) -> Report:
        """returns a reporter object, using the logger data"""
        reporter = Report(scorer=self.scorer)
        reporter.set_columns_to_show(['classifier', 'transformer'] + list(self.scorer.scoring_fcts.keys()))

        for report_key, report in self.logger.reports.items():
            transformer_str, clf_str = report_key.split(':')
            config = {'feature_selection': str(self.features), 
                      'transformer': transformer_str, 
                      'scaler': str(self.scaler),
                      'classifier': clf_str}

            reporter.add_to_report(config, pd.DataFrame(report), show=False)

        return reporter