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

from . import encoder, feature_selection

Config = dict[str, str]

class CategoricalPipeline:
    """a pipeline including feature selection, categorical encoding,
    scaling (optional) and a classifier"""
    
    def __init__(self, 
                 features: feature_selection.FeatureSelection,
                 encoder: Optional[encoder.CategoricalEncoder] = None,
                 scaler: Optional[TransformerMixin] = None,
                 model: Optional[BaseEstimator] = None, ):
        self.features = features
        self.encoder = encoder
        self.scaler = scaler
        self.model = model
        
        trans = ColumnTransformer(
            [('categories', self.encoder, self.features.categoricals)], 
            remainder='passthrough')
        
        steps = [
            ('transform', trans),
            ('clf', self.model),
        ]
        
        if scaler is not None:
            steps.insert(1, ('scale', self.scaler))
        # chain the transformation and prediction steps
        self.pipe = Pipeline(steps=steps)
        
    def fit(self, df: pd.DataFrame, *args, **kwargs) -> None:
        # fit to the training data
        X, y = self.features.select_features(df)
        
        self.pipe.fit(X, y, *args, **kwargs)
        
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """"""
        X, _ = self.features.select_features(df)
        
        return self.pipe.predict(X)
    
    def clone(self):
        cls = self.__class__
        encoder = None if self.encoder is None else clone(self.encoder)
        scaler = None if self.scaler is None else clone(self.scaler)
        model = None if self.model is None else clone(self.model)
        return cls(
            features = self.features,
            encoder = encoder,
            scaler = scaler,
            model = model,
        )
        
    
    @property
    def config(self) -> Config:
        """returns a dictionary with the Pipeline's attributes turned into
        strings"""
        return {k: str(v) for k,v in self.__dict__.items()}
    
    def __repr__(self) -> str:
        return f"""CategoricalPipeline(
        features={self.features},
        encoder={self.encoder},
        scaler={self.scaler},
        model={self.model}),   
        """


