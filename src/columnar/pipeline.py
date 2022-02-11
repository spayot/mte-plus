"""
Defines the CategoricalPipeline Class, which allows to define a 4-steps pipeline for structured columnar datasets:
- feature selection: defining the target column as well as categorical and numerical columns to use as features for prediction
- categorical encoder: transforms categorical data into numerical ones
- scaler (optional): rescales the data
- model : a classifier to apply to the dataset.
"""
from typing import Callable, List, Optional, Tuple, Any


import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin, clone

from . import feature_selection

Config = dict[str, str]



class CategoricalPipeline:
    """a pipeline including feature selection, categorical encoding,
    scaling (optional) and a classifier"""
    
    def __init__(self, 
                 features: feature_selection.FeatureSelection,
                 transformer: TransformerMixin,
                 scaler: Optional[TransformerMixin] = None,
                 model: Optional[BaseEstimator] = None, 
                ):
        self.features = features
        self.transformer = transformer # the encoding strategy, defines how each column should be transformed
        self.scaler = scaler # a scaling layer
        self.model = model
        
        
        steps = [
            ('transformer', self.transformer),
            ('clf', self.model),
        ]
        
        if scaler is not None:
            steps.insert(1, ('scale', self.scaler))
        # chain the transformation and prediction steps
        self.pipe = Pipeline(steps=steps)
        
    def fit(self, df: pd.DataFrame, *args, **kwargs) -> None:
        # fit to the training data
        X, y = self.features.select_features(df)
        
        # fit the pipeline (note: all transformers need a FeatureSelection \
        # object as parameter at fitting time)
        self.pipe.fit(X, y, transformer__features=self.features, *args, **kwargs)
        
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """returns predicted probability for `target` to be true for each observation"""
        X, _ = self.features.select_features(df)
        
        return self.pipe.predict_proba(X)[:,1]
    
    def clone(self):
        """allows to create a clone version of the same pipeline (ie reinitialize with same steps)"""
        cls = self.__class__
        return cls(
            features = self.features,
            transformer = clone(self.transformer),
            scaler = None if self.scaler is None else clone(self.scaler),
            model = clone(self.model),
        )
        
    
    @property
    def config(self) -> Config:
        """returns a dictionary with the Pipeline's attributes turned into
        strings"""
        return {k: str(v) for k,v in self.__dict__.items() if k != 'pipe'}
    
    def __repr__(self) -> str:
        return f"""CategoricalPipeline(
        features={self.features},
        transformer={self.transformer},
        scaler={self.scaler},
        model={self.model}),   
        """