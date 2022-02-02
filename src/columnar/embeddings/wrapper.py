"""Wrapper Class to fit an embedding layer based on a 
simple Deep Learning model.

"""
from typing import Optional

import pandas as pd 
from sklearn.base import TransformerMixin
import tensorflow as tf

from ..encode import CategoricalEncoder
from ..feature_selection import FeatureSelection

from .data import df_to_dataset
from .layers import TFEmbeddingLayer, TFNormalizationLayer
from .models import TFCatEmbsClassifier


class TFEmbeddingWrapper(TransformerMixin):
    """Allows to transform categorical features into embeddings, 
    after fitting a simple neural network on the dataset.
    Note: numerical features are normalized for the sake of fitting 
    the DNN and generating embeddings."""
    def __init__(self, 
                 features: FeatureSelection,
                ):
        self.features = features
        
        
                
        # initialize model (concatenated embeddings + dense layers) based on dataset and feature selection
        self.model = TFCatEmbsModel(dataset, features)
        
        # compile
        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                           metrics=["accuracy"])
        
        # 
        self.model.fit(dataset)
    
    def get_params(self, deep: bool):
        return {'task': self.task, 'features': self.features, 'root_path': self.root_path}
    
    
    def fit(self, df: pd.DataFrame, 
            y: Optional[pd.Series] = None, 
            epochs: int = 3,
            verbose: str = 2,
            **kwargs,
           ) -> None:
        if y is None:
            y = df[self.features.target]
        
        # transform dataframe into dataset
        dataset = df_to_dataset(df.drop(columns=[self.features.target]), target=y)
        
        # initialize model (concatenated embeddings + dense layers) based on dataset and feature selection
        self.model = TFCatEmbsClassifier(dataset, features)
        
        # compile
        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                           metrics=["accuracy"])
        
        
        
        self.model.fit(dataset, epochs=epochs, verbose=verbose, **kwargs)
        
        return self
        
    
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass
    
    def _transform(self, dataset: tf.data.Dataset) -> None:
        pass
    
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass
    