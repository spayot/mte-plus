import tensorflow as tf
from tensorflow.keras import layers

from ..feature_selection import DatasetColumn

def _calculate_embedding_size(cardinality: int) -> int:
    """strategy to define the embedding size, given the 
    cardinality of a categorical feature"""
    return min(cardinality // 2, 50)    

def _get_dtype(ds: tf.data.Dataset, header: str):
    """helper function to grab the dtype for a given """
    return ds.element_spec[0][header].dtype

class TFEmbeddingLayer(layers.Layer):
    def __init__(self, 
                 col_name: DatasetColumn, 
                 dataset: tf.data.Dataset, 
                 **kwargs):
        super().__init__(**kwargs)
        self.col_name = col_name
        dtype = _get_dtype(dataset, col_name)
        lookup = layers.StringLookup if dtype == tf.string else layers.IntegerLookup
        self.index = lookup(num_oov_indices=1, name=col_name)
        
        
        # Prepare a `tf.data.Dataset` that only yields the feature.
        feature_ds = dataset.map(lambda x, y: x[col_name])
        self.index.adapt(feature_ds)
        voc_size = self.index.vocabulary_size()
        self.emb_size = _calculate_embedding_size(voc_size)
        self.embedding = layers.Embedding(input_dim=voc_size, 
                                          output_dim=self.emb_size, 
                                          name=f'{col_name}_embedding')
        
    def call(self, x):
        """defines the graph to compute outputs from inputs"""
        x = self.index(x)
        x = self.embedding(x)
        # remove unnecessary second dimension (catered to input_length > 1)
        x = x[:,0,:]
        return x
    
    
    
class TFNormalizationLayer(layers.Layer):
    def __init__(self, 
                 col_name: DatasetColumn, 
                 dataset: tf.data.Dataset, 
                 **kwargs):
        super().__init__(**kwargs)
        self.col_name = col_name
        self.normalizer = layers.Normalization(axis=None)
        
        # Prepare a Dataset that only yields the feature.
        feature_ds = dataset.map(lambda x, y: x[col_name])
        
        # Learn the statistics of the data.
        self.normalizer.adapt(feature_ds)
        
    def call(self, x):
        """defines the graph to compute outputs from inputs"""
        return self.normalizer(x)