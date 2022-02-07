"""Wrapper Class to fit an embedding layer based on a 
simple Deep Learning model.

"""
from typing import Optional

import numpy as np
import pandas as pd 
from sklearn.base import TransformerMixin
import tensorflow as tf

from ..encode import CategoricalEncoder
from ..feature_selection import FeatureSelection
from ..utils import set_repr

from .data import df_to_dataset
from .layers import EmbSizeStrategyName
from .models import TFCatEmbsClassifier, TFCatEmbsEncoder, BaseTransformStrategy


class TFEmbeddingWrapper(TransformerMixin):
    """Allows to transform categorical features into embeddings, 
    after fitting a simple neural network on the dataset.
    Note: at fitting time, numerical features are normalized for the sake 
    of generating quality embeddings, but at transform time, numerical
    features are passed through. """
    def __init__(self, 
                 features: FeatureSelection,
                 emb_size_strategy: EmbSizeStrategyName,
                ):
        self.features = features
        self.emb_size_strategy = emb_size_strategy
        
        
                
        # define encoding_strategy for each 
        self.transform_strategy = BaseTransformStrategy(features, emb_size_strategy)
        
        # define encoder based on that strategy
        encoder = TFCatEmbsEncoder(self.transform_strategy)

        # instantiate classifier
        self.model = TFCatEmbsClassifier(encoder)
    
    
    def get_params(self, deep: bool):
        return {'features': self.features, 'emb_size_strategy': self.emb_size_strategy}
    
    
    def fit(self, df: pd.DataFrame, 
            y: Optional[pd.Series] = None, 
            epochs: int = 3,
            verbose: str = 1,
            **kwargs,
           ) -> None:
        if y is None:
            y = df[self.features.target]
        
        # transform dataframe into dataset
        dataset = df_to_dataset(df, y)
        
        
        # initialize model (concatenated embeddings + dense layers) based on dataset and feature selection
        self.model.adapt(dataset)
        
        # compile
        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                           metrics=["accuracy"])
        
        
        
        self.model.fit(dataset, epochs=epochs, verbose=verbose, **kwargs)
        
        return self
        
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        num_features = df[self.features.numericals].values
        
        cat_data = df_to_dataset(df[self.features.categoricals], shuffle=False)
        
        # get concatenated embeddings for categorical data
        cat_features = self.model.encoder.predict(cat_data)
        
        return np.concatenate([num_features, cat_features], axis=1)
    
    def __repr__(self) -> str:
        return f"TFEmbeddingWrapper_{self.emb_size_strategy.capitalize()}Strategy"
        
        
    