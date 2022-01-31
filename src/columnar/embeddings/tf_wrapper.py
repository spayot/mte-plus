"""Wrapper Class to fit an embedding layer based on a 
simple Deep Learning model.

how to do proper get_params

"""
from typing import Optional

import pandas as pd 
import tensorflow as tf

from ..encoder import CategoricalEncoder
from ..feature_selection import FeatureSelection

from .tf_dataload import df_to_dataset
from .tf_preprocessing import TFEmbeddingLayer, TFNormalizationLayer
from .tf_models import TFCatEmbsModel


class TFEmbeddingWrapper(CategoricalEncoder):
    """Allows to transform categorical features into embeddings, 
    after fitting a simple neural network on the dataset.
    Note: numerical features are normalized for the sake of fitting 
    the DNN and generating embeddings."""
    def __init__(self, 
                 task: str,
                 features: FeatureSelection,
                 root_path: str = './',
                ):
        self.ROOT_PATH = root_path
        self.features = features
        self.task = task
        
        # load the dataset for the task
        loader = col.DataLoader(root=root_path, task=task)
        dataframe = loader.load_data()
        dataset = df_to_dataset(dataframe, 
                                target=features.target,
                                )
        
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
    
    
    def fit(self, df: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        if y is None:
            y = df[self.features.target]
            
        dataset = df_to_dataset(df.drop(columns=[self.features.target]), target=y)
        self.model.fit(dataset)
    
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass
    
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass
    