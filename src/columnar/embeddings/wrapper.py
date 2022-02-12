"""Wrapper Class to fit an embedding layer based on a 
simple Deep Learning model.

"""
from typing import Optional

import numpy as np
import pandas as pd 
import tensorflow as tf

from ..encode import CategoricalTransformer
from ..feature_selection import FeatureSelection
from ..utils import set_repr

from .data import df_to_dataset
from .layers import EmbSizeStrategyName
from .models import TFCatEmbsClassifier, TFCatEmbsEncoder, BaseTransformStrategy


class TFEmbeddingWrapper(CategoricalTransformer):
    """Allows to transform categorical features into embeddings, 
    after fitting a simple neural network on the dataset.
    Note: at fitting time, numerical features are normalized for the sake 
    of generating quality embeddings, but at transform time, numerical
    features are passed through. """
    def __init__(self, 
                 emb_size_strategy: EmbSizeStrategyName,
                ):
        self.features: FeatureSelection = None
        self.emb_size_strategy = emb_size_strategy
        self.transform_strategy = None
        self.model: TFCatEmbsClassifier = None
    
    
    def get_params(self, deep: bool):
        return {'emb_size_strategy': self.emb_size_strategy}
    
    
    def fit(self, X: pd.DataFrame, y: pd.Series, 
            features: FeatureSelection,
            epochs: int = 3,
            verbose: str = 1,
            **kwargs,
           ) -> None:
        
        self.features = features
        
        # define encoding_strategy for each features
        self.transform_strategy = BaseTransformStrategy(features, self.emb_size_strategy)
        
        # create encoder model based on that strategy
        encoder = TFCatEmbsEncoder(self.transform_strategy)

        # instantiate classifier with encoder + classifier head
        self.model = TFCatEmbsClassifier(encoder)
        
        # transform dataframe into dataset
        dataset = df_to_dataset(X, y)
        
        
        # initialize model (concatenated embeddings + dense layers) based on dataset and feature selection
        self.model.adapt(dataset)
        
        # compile
        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                           metrics=["accuracy"])
        
        
        
        self.model.fit(dataset, epochs=epochs, verbose=verbose, **kwargs)
        
        self.fitted = True
        
        return self
        
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """transforms input X so that categorical features are represented as 
        embeddings while numericals are passed through"""
        num_features = X[self.features.numericals].values
        
        cat_data = df_to_dataset(X[self.features.categoricals], shuffle=False)
        
        # get concatenated embeddings for categorical data
        cat_features = self.model.encoder.predict(cat_data)
        
        return np.concatenate([num_features, cat_features], axis=1)
    
    
    def predict_class_from_df(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(df_to_dataset(X, shuffle=False))
    
    def __repr__(self) -> str:
        return f"TFEmbeddingWrapper_{self.emb_size_strategy.capitalize()}Strategy()"
        
        
    