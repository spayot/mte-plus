"""
defines an abstract FeatureTransformStrategy
"""

from typing import Annotated

import tensorflow as tf
from tensorflow.keras import layers

from .. import feature_selection as features, utils
from . import emb_size, tf_strategy, layers as custom_layers

from . import tf_strategy

# typing definitions
Inputs = Annotated[dict[features.DatasetColumn, tf.Tensor], 
                   "inputs to a tensorflow model presented as a dictionary with features as keys and tensors for values"]

        

class TFCatEmbsEncoder(tf.keras.Model):
    """encoder that takes as input a dictionary with feature names as keys
    and tensors of feature values as values and returns a single tensor to pass
    into a classifier."""
    def __init__(self, tf_strategy: tf_strategy.TFTransformStrategy):
        super().__init__()
        self.tf_strategy = tf_strategy
        self.encoding_layers = tf_strategy.get_encoding_layers()
        
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
        return self.__class__.__name__ + f'(strategy={self.tf_strategy.name})'
        
        
        
        
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
        self.dense1 = layers.Dense(hidden_size, activation="relu")
        self.dropout = layers.Dropout(dropout_rate)
        self.dense2 = layers.Dense(1)
        
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
        return utils.set_repr(self, ['hidden_size', 'dropout_rate', 'encoder'])