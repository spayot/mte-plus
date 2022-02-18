"""module dedicated to TensorFlow based embedding transformers.
MonoEmbeddings is the wrapper class that makes it behave like a MonoTransformer,
and allows to integrate it with other MonoTransformers into a CompositeTransformer"""

from . import data, layers, emb_size, tf_strategy, models, wrapper
from .wrapper import MonoEmbeddings
from .models import TFCatEmbsClassifier
from .emb_size import emb_size_factory
