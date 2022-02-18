"""Wrapper Class to fit an embedding layer based on a 
simple Deep Learning model.

"""
from typing import Optional

import numpy as np
import pandas as pd 
import tensorflow as tf

from ..transform import mono, composite
from .. import utils, feature_selection 

from . import data, emb_size, tf_strategy, models

class MonoEmbeddings(mono.MonoTransformer):
    """Allows to transform categorical features into embeddings, 
    after fitting a simple neural network on the dataset.
    Note: at fitting time, numerical features are normalized for the sake 
    of generating quality embeddings, but at transform time, numerical
    features are passed through. """
    def __init__(self, 
                 emb_size_strategy: emb_size.EmbSizeStrategyName,
                ):
        
        self.emb_size_strategy = emb_size_strategy
        self.transform_strategy = None
        self.model: models.TFCatEmbsClassifier = None
    
    
    def get_params(self, deep: bool):
        return {'emb_size_strategy': self.emb_size_strategy}
    
    
    def fit(self, X: pd.DataFrame, y: pd.Series, 
            columns: list[feature_selection.DatasetColumn],
            epochs: int = 3,
            verbose: str = 1,
            **kwargs,
           ) -> None:
        
        self.columns = columns
        
        # assume that all columns that are not to be transformed into embeddings are numericals
        self.numericals = [column for column in X.columns if column not in columns]
        
        # define encoding_strategy for each features
        self.transform_strategy = tf_strategy.BaseTFTransformStrategy(categoricals=columns, 
                                                                      numericals=self.numericals,
                                                                      emb_size_strategy=self.emb_size_strategy)
        
        # create encoder model based on that strategy
        encoder = models.TFCatEmbsEncoder(self.transform_strategy)

        # instantiate classifier with encoder + classifier head
        self.model = models.TFCatEmbsClassifier(encoder)
        
        # transform dataframe into dataset
        dataset = data.df_to_dataset(X, y)
        
        
        # initialize model (concatenated embeddings + dense layers) based on dataset and feature selection
        self.model.adapt(dataset)
        
        # compile
        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                           metrics=["accuracy"])
        
        
        
        self.model.fit(dataset, epochs=epochs, verbose=verbose, **kwargs)
        
        # set _fitted and columns class attributes
        super().fit(X, y, columns)
        
        return self
        
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """transforms input X so that categorical features are represented as 
        embeddings while numericals are passed through"""
        # validate transformer has been fitted first
        super().transform(X)
        
        cat_data = data.df_to_dataset(X[self.columns], shuffle=False)
        
        # get concatenated embeddings for categorical data
        cat_features = self.model.encoder.predict(cat_data)
        
        return cat_features
    
    
    def predict_class_from_df(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(data.df_to_dataset(X, shuffle=False))
    
    def __repr__(self) -> str:
        return f"MonoEmbeddings_{self.emb_size_strategy.capitalize()}Strategy()"
        
        
    