import tensorflow as tf
from tensorflow.keras import layers

from ..feature_selection import FeatureSelection, DatasetColumn
from .tf_preprocessing import TFEmbeddingLayer, TFNormalizationLayer

Inputs = dict[DatasetColumn, tf.Tensor]

class TFCatEmbsModel(tf.keras.Model):
    def __init__(self, 
                 train_ds: tf.data.Dataset, 
                 feature_selection: FeatureSelection,
                 hidden_size = 32,
                 dropout_rate=.5,
                ):
        super().__init__()
        self.feature_selection = feature_selection
        self.encoding_layers = dict()
        
        # categorical encoding
        for col_name in feature_selection.categoricals:
            self.encoding_layers[col_name] = TFEmbeddingLayer(col_name=col_name, dataset=train_ds)
        
        # numerical encoding
        for col_name in feature_selection.numericals:
            self.encoding_layers[col_name] = TFNormalizationLayer(col_name=col_name, dataset=train_ds)
            
        self.dense1 = tf.keras.layers.Dense(hidden_size, activation="relu")
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dense2 = tf.keras.layers.Dense(1)
        
    def call(self, inputs: Inputs):
        """defines the graph to compute outputs from inputs"""
        # encode and concatenate features
        all_features = self.encode_features(inputs)
        
        # apply dense layers with dropout 
        x = self.dense1(all_features)
        x = self.dropout(x)
        
        # final prediction layer
        output = self.dense2(x)
        return output
    
    def encode_features(self, inputs: Inputs, columns: list[DatasetColumn] = None) -> tf.Tensor:
        """applies encoding layers to the inputs. if the columns argument is not defined, it is applied to the whole """
        if columns is None:
            columns = self.encoding_layers.keys()
        
        encoded_features = []
        
        for col_name in columns:
            # apply the relevant encoder to the relevant column
            encoded_col = self.encoding_layers[col_name](inputs[col_name])
            # add to list of encoded features
            encoded_features.append(encoded_col)

        all_features = layers.concatenate(encoded_features)
        
        return all_features