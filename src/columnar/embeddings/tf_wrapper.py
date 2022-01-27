"""Wrapper Class to fit an embedding layer based on a 
simple Deep Learning model."""
from typing import Optional

import pandas as pd 
import tensorflow as tf

from ..encoder import CategoricalEncoder
from ..feature_selection import FeatureSelection

from .tf_dataload import df_to_dataset
from .tf_preprocessing import TFEmbeddingLayer, TFNormalizationLayer
from .tf_models import TFCatEmbsModel


class TFEmbeddingWrapper(CategoricalEncoder):
    def __init__(self, 
                 dataframe: pd.DataFrame,
                 features: FeatureSelection,
                 **kwargs,
                ):
        self.features = features
        dataset = df_to_dataset(dataframe, 
                                target=features.target,
                                **kwargs,
                                )
        self.model = TFCatEmbsModel(dataset, features)
        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                           metrics=["accuracy"])
    
    
    def get_params(self, deep: bool):
        pass
    
    
    def fit(self, df: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        if y is None:
            y = df[self.features.target]
            
        dataset = df_to_dataset(df.drop(columns=[self.features.target]), target=y)
        self.model.fit(dataset)
    
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass
    
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass
    