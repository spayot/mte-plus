"""
defines an abstract TFTransformStrategy defining the type of tensorflow 
transformation to apply to each input feature.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

from tensorflow.keras import layers

from .. import utils, feature_selection as features
from . import emb_size, layers as custom_layers 



@dataclass
class TFTransformStrategy(ABC):
    """a dictionary defining the type of tensorflow 
    transformation to apply to each input feature"""
    name: str
    
    @abstractmethod
    def get_encoding_layers(self) -> dict[str, str]:
        pass

    
@dataclass    
class BaseTFTransformStrategy(TFTransformStrategy):
    emb_size_strategy: str
    def __init__(self, 
                 categoricals: list[features.DatasetColumn], 
                 numericals: list[features.DatasetColumn], 
                 emb_size_strategy: emb_size.EmbSizeStrategyName):
        """transforms all categorical features into embeddings while all
        numerical features are normalized.
        Args:
            features: defines which features are considered as categoricals or numericals
            emb_size_strategy: the strategy used to define the number of """
        super().__init__(self)
        self.name = 'base'
        self.categoricals = categoricals
        self.numericals = numericals
        self.emb_size_strategy = emb_size_strategy
        categorical_layers = {name: custom_layers.TFEmbeddingLayer(name, emb_size_strategy) for name in categoricals}
        numerical_layers = {name: custom_layers.TFNormalizationLayer(name) for name in numericals}
        self._encoding_layers = categorical_layers
        self._encoding_layers.update(numerical_layers)
        
    def get_encoding_layers(self) -> dict[features.DatasetColumn, layers.Layer]:
        return self._encoding_layers