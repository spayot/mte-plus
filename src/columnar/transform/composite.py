"""Defines the CompositeTransformer class, which allows to define the 
transformation strategy for each individual column, using a TransformerStrategy instance"""
from dataclasses import dataclass
from typing import Any, Union

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import TransformerMixin

from ..feature_selection import DatasetColumn, FeatureSelection
from .mono import assert_fitted, MonoTransformer, PassThrough
from .strategy import MonoStrategy, TransformerStrategy


class CompositeTransformer(TransformerMixin):
    """similar to sklearn's ColumnTransformer class, but allows to 
    fit using all the input data (necessary for embeddings transformations)
    
    Example: 
    >>> from columnar import transform
    >>> strategy = transform.composite.TransformerStrategy.from_tuples(
            ('cats', transform.mono.MeanTargetEncoder(), ['cat_feature1', 'cat_feature2']),
            ('nums', transform.mono.PassThrough(), ['num_feature1']))
    >>> transformer = CompositeTransformer(strategy)
    >>> transformer.fit(X, y)
    >>> X_ = transformer.transform(X_)
    
    """
    def __init__(self, strategy: TransformerStrategy):
        self.strategy = strategy
        self._fitted = None
        
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        for monostrat in self.strategy:
            monostrat.transformer.fit(X, y, columns=monostrat.columns)
            
        self._fitted = True
        return self
            
    def transform(self, X: pd.DataFrame) -> Union[pd.DataFrame, sparse.csr_matrix, np.ndarray]:
        """applies composite transformation to each features and concatenate into a final transformed output.
        output type depends on transformation types:
        - if any of the transformations outputs a sparse format, the output will be sparse. 
        - if all mono-transformations outputs dataframes, the output will be a dataframe.
        - otherwise, the output will be a numpy array."""
        assert_fitted(self)
        X_transformed = [monostrat.transformer.transform(X) for monostrat in self.strategy]
        
        if _is_all_dataframe(X_transformed):
            return pd.concat(X_transformed, axis=1)
        elif _any_sparse(X_transformed):
            return sparse.hstack(X_transformed).tocsr()
        else:
            return np.hstack(X_transformed)
        
    def get_params(self, deep: bool):
        return {"strategy": self.strategy}
    
    def __repr__(self) -> str:
        return f"CompositeTransformer(strategy={self.strategy})"
            
    
class SimpleCompositeTransformer(CompositeTransformer):
    """A CompositeTransformer with only 2 transformation sub-strategies: 
    - one for categoricals (defined during instantiation)
    - one for numericals (passthrough)"""
    def __init__(self, cat_transformer: MonoTransformer, features: FeatureSelection):
        self.cat_transformer = cat_transformer
        self.features = features
        self.strategy = TransformerStrategy.from_tuples(
            ('categoricals', cat_transformer, features.categoricals),
            ('numericals', PassThrough(), features.numericals))
        
    def __repr__(self) -> str:
        return f"SimpleComposite_{self.cat_transformer}"
        
    
def _any_sparse(l: list[Any]):
    return any(type(element) == sparse.csr_matrix for element in l)

def _is_all_dataframe(l: list[Any]):
    return all(type(element) == pd.DataFrame for element in l)    
