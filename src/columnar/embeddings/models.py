import tensorflow as tf
from tensorflow.keras import layers

from ..feature_selection import FeatureSelection, DatasetColumn
from .layers import TFEmbeddingLayer, TFNormalizationLayer

Inputs = dict[DatasetColumn, tf.Tensor]

EncodingStrategy = dict[DatasetColumn, layers.Layer]

def get_base_strategy(feature_selection: FeatureSelection) -> EncodingStrategy:
    # embeddings for all categorical variables
    base_strategy = {name: TFEmbeddingLayer for name in feature_selection.categoricals}
    # Normalization for others
    base_strategy.update({name: TFNormalizationLayer for name in feature_selection.numericals})
    return base_strategy



class TFCatEmbsClassifier(tf.keras.Model):
    def __init__(self, 
                 train_ds: tf.data.Dataset, 
                 encoding_strategy: EncodingStrategy,
                 hidden_size = 32,
                 dropout_rate=.5,
                ):
        super().__init__()
        self.feature_selection = feature_selection
        self.encoding_layers = dict()
        
        # categorical encoding
        for col_name, layer in encoding_strategy.items():
            feature_ds = dataset.map(lambda x, y: x[col_name])
            self.encoding_layers[col_name] = layer(col_name=col_name, dataset=train_ds)
        
        # numerical encoding
        for col_name in feature_selection.numericals:
            feature_ds = feature_ds = dataset.map(lambda x, y: x[col_name])
            self.encoding_layers[col_name] = TFNormalizationLayer(col_name=col_name, dataset=train_ds)
            
        self.dense1 = tf.keras.layers.Dense(hidden_size, activation="relu")
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dense2 = tf.keras.layers.Dense(1)
        
    def call(self, inputs: Inputs):
        """defines the graph to compute outputs from inputs"""
        # encode and concatenate features
        all_features = self._encode_features(inputs)
        
        # apply dense layers with dropout 
        x = self.dense1(all_features)
        x = self.dropout(x)
        
        # final prediction layer
        output = self.dense2(x)
        return output
    
    def _encode_features(self, inputs: Inputs, columns: list[DatasetColumn] = None) -> tf.Tensor:
        """applies encoding layers to the inputs. if the columns argument is not defined, it is applied to the whole """
        if columns is None:
            columns = inputs.keys()
        
        encoded_features = []
        
        for col_name in columns:
            # apply the relevant encoder to the relevant column
            encoded_col = self.encoding_layers[col_name](inputs[col_name])
            # add to list of encoded features
            encoded_features.append(encoded_col)

        all_features = layers.concatenate(encoded_features)
        
        return all_features

    
        
        
        
class TFCatEmbeddingEncoder(tf.keras.Model):
    def __init__(self, encoding_strategy: EncodingStrategy):
        self.encoding_layers = encoding_layers
        self.features = features
        
    def call(self, inputs: Inputs) -> tf.Tensor:
        """"""
        encoded_features = []
        
        for col_name in columns:
            # apply the relevant encoder to the relevant column
            encoded_col = self.encoding_layers[col_name](inputs[col_name])
            # add to list of encoded features
            encoded_features.append(encoded_col)

        all_features = layers.concatenate(encoded_features)
        
        return all_features
        
        