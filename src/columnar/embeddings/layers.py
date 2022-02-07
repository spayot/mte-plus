"""

inspired by https://www.tensorflow.org/tutorials/structured_data/preprocessing_layers
"""

from enum import Enum
from typing import Annotated, Callable


import tensorflow as tf
from tensorflow.keras import layers

from ..feature_selection import DatasetColumn
from ..utils import set_repr, Repr


# define method to choose the size of the embedding based on the cardinality
EmbSizeStrategyName = Annotated[str, "strategy to define the embedding size, given a categorical features cardinality"]


class EmbSizeFactory:
    def __init__(self):
        self._builders = {}

    def register_builder(self, 
                         key: EmbSizeStrategyName, 
                         builder: Callable[[int], int]):
        self._builders[key] = builder

    def calculate_emb_size(self, key, cardinality):
        builder = self._builders.get(key)
        if not builder:
            raise ValueError(key)
        return builder(cardinality)
    
emb_size_factory = EmbSizeFactory()

emb_size_factory.register_builder('max50', lambda cardinality: min(cardinality // 2, 50))
emb_size_factory.register_builder('single', lambda cardinality: 1)
emb_size_factory.register_builder('max2', lambda cardinality: min(cardinality // 2, 2))



def _get_dtype(ds: tf.data.Dataset, header: str):
    """helper function to grab the dtype for a given """
    return ds.element_spec[0][header].dtype



class TFEmbeddingLayer(layers.Layer):
    def __init__(self, 
                 col_name: DatasetColumn, 
                 strategy: EmbSizeStrategyName = 'max50',
                 **kwargs):
        super().__init__(**kwargs)
        self.col_name = col_name
        self.strategy = strategy
        
        self.index = None
        self.emb_dim = None
        self.embedding= None
    
    
    @tf.autograph.experimental.do_not_convert
    def adapt(self, dataset: tf.data.Dataset) -> None:
        """creates embedding"""
        dtype = _get_dtype(dataset, self.col_name)
        lookup = layers.StringLookup if dtype == tf.string else layers.IntegerLookup
        self.index = lookup(num_oov_indices=1, name=self.col_name)
        
        
        # Prepare a `tf.data.Dataset` that only yields the feature.
        feature_ds = dataset.map(lambda x, y: x[self.col_name])
        self.index.adapt(feature_ds)
        voc_size = self.index.vocabulary_size()
        self.emb_dim = emb_size_factory.calculate_emb_size(self.strategy, voc_size)
        self.embedding = layers.Embedding(input_dim=voc_size, 
                                          output_dim=self.emb_dim, 
                                          name=f'{self.col_name}_embedding')
    
    def call(self, x):
        """defines the graph to compute outputs from inputs"""
        x = self.index(x)
        x = self.embedding(x)
        # remove unnecessary second dimension (catered to input_length > 1)
        x = x[:,0,:]
        return x
    
    def __repr__(self) -> str:
        return set_repr(self, ['col_name', 'strategy'])
    
    

class TFNormalizationLayer(layers.Layer, Repr):
    def __init__(self, col_name: DatasetColumn,
                 **kwargs):
        super().__init__(**kwargs)
        self.col_name = col_name
        self.normalizer = layers.Normalization(axis=None)
        
    def adapt(self, dataset: tf.data.Dataset) -> None:
        # sample only the feature of interest from the Dataset
        feature_ds = dataset.map(lambda x, y: x[self.col_name])
        
        # Learn the statistics of the data.
        self.normalizer.adapt(feature_ds)
        
    def call(self, x):
        """defines the graph to compute outputs from inputs"""
        return self.normalizer(x)
    
    
    
# class PassThroughLayer(layers.Layer, Repr):
#     def __init__(self, col_name: DatasetColumn, **kwargs):
#         super().__init__(**kwargs)
#         self.col_name = col_name
        
#     def adapt(self, datset: tf.data.Dataset) -> None:
#         pass
    
#     def call(self, x: tf.Tensor) -> tf.Tensor:
#         return x
    
    
        