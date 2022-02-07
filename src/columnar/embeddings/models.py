"""
defines an abstract FeatureTransformStrategy
"""
from abc import ABC, abstractmethod
from typing import Annotated
from dataclasses import dataclass

import tensorflow as tf
from tensorflow.keras import layers

from ..feature_selection import FeatureSelection, DatasetColumn
from ..utils import Repr, set_repr
from .layers import TFEmbeddingLayer, TFNormalizationLayer, EmbSizeStrategyName

# typing definitions
Inputs = Annotated[dict[DatasetColumn, tf.Tensor], 
                   "inputs to a tensorflow model presented as a dictionary with features as keys and tensors for values"]

@dataclass
class FeatureTransformStrategy(ABC):
    """a dictionary defining the type of tensorflow 
    transformation to apply to each input feature"""
    name: str
    
    @abstractmethod
    def get_encoding_layers(self) -> dict[str, str]:
        pass

@dataclass    
class BaseTransformStrategy(FeatureTransformStrategy):
    emb_size_strategy: str
    def __init__(self, 
                 features: FeatureSelection, 
                 emb_size_strategy: EmbSizeStrategyName):
        """transforms all categorical features into embeddings while all
        numerical features are normalized.
        Args:
            features: defines which features are considered as categoricals or numericals
            emb_size_strategy: the strategy used to define the number of """
        super().__init__(self)
        self.name = 'base_strategy'
        self.features = features
        self.emb_size_strategy = emb_size_strategy
        categorical_layers = {name: TFEmbeddingLayer(name, emb_size_strategy) for name in features.categoricals}
        numerical_layers = {name: TFNormalizationLayer(name) for name in features.numericals}
        self._encoding_layers = categorical_layers
        self._encoding_layers.update(numerical_layers)
        
    def get_encoding_layers(self) -> dict[DatasetColumn, layers.Layer]:
        return self._encoding_layers
    

        

class TFCatEmbsEncoder(tf.keras.Model):
    """encoder that takes as input a dictionary with feature names as keys
    and tensors of feature values as values and returns a single tensor to pass
    into a classifier."""
    def __init__(self, transform_strategy: FeatureTransformStrategy):
        super().__init__()
        self.transform_strategy = transform_strategy
        self.encoding_layers = transform_strategy.get_encoding_layers()
        
    def adapt(self, dataset: tf.data.Dataset) -> None:
        for col_name, encoder in self.encoding_layers.items():
            encoder.adapt(dataset)
        
    def call(self, inputs: Inputs):
        """defines the graph to compute outputs from inputs.
        Note: the encoding can be done on a subset of all the features."""
        
        
        # encode and concatenate features
        encoded_features = []
        
        for col_name in inputs.keys():
            # apply the relevant encoder to the relevant column
            encoded_col = self.encoding_layers[col_name](inputs[col_name])
            # add to list of encoded features
            encoded_features.append(encoded_col)

        all_features = layers.concatenate(encoded_features)
        
        return all_features
    
    def __repr__(self) -> str:
        return self.__class__.__name__ + f'(strategy={self.transform_strategy})'
        
        
        
        
        
class TFCatEmbsClassifier(tf.keras.Model):
    def __init__(self, 
                 encoder: TFCatEmbsEncoder,
                 hidden_size: int = 32,
                 dropout_rate: float = .5,
                ):
        super().__init__()
        self.encoder = encoder
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        
        # build classifier head
        self.dense1 = tf.keras.layers.Dense(hidden_size, activation="relu")
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dense2 = tf.keras.layers.Dense(1)
        
    def adapt(self, dataset: tf.data.Dataset) -> None:
        self.encoder.adapt(dataset)
        
    def call(self, inputs: Inputs):
        """defines the graph to compute outputs from inputs"""
        # encode and concatenate features
        all_features = self.encoder(inputs)
        
        # apply dense layer with dropout 
        x = self.dense1(all_features)
        x = self.dropout(x)
        
        # final prediction layer
        output = self.dense2(x)
        return output      
    
    def __repr__(self) -> str:
        return set_repr(self, ['hidden_size', 'dropout_rate', 'encoder'])