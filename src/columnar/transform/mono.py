"""
Defines the MonoTransformer Interface as well as the MeanTargetEncoder implementation.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Any

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector as selector


from ..feature_selection import DatasetColumn, FeatureSelection
from ..utils import set_repr


def assert_fitted(obj):
    """helper function to validate a transformer has been fitted before 
    it is used for transformation"""
    assert obj._fitted == True, f"{obj.__class__} needs to be fitted before transformation"
    
    
    
class MonoTransformer(TransformerMixin, ABC):
    """Abstract class to define a single transformation technique 
    applied on a defined set of columns."""
    columns: list[DatasetColumn] = None
    suffix: str = '_'
    _fitted: bool = False
            
    @abstractmethod
    def get_params(self, deep: bool):
        pass
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, columns: list[DatasetColumn]) -> None:
        self.columns = columns
        self._fitted = True
    
    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """takes as input the full dataset but only applies its 
        transformatinon to the features listed in its `columns` 
        attribute"""
        assert_fitted(self)
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series, 
                      columns: list[DatasetColumn], **kwargs) -> pd.DataFrame:
        return self.fit(X, y, columns, **kwargs).transform(X)
    
    
class MTEMapper:
    """a Mapper object.
        each attribute"""
    def add_map(self, column: DatasetColumn, map: dict[str, float]) -> None:
        setattr(self, column, map)
        
    def items(self):
        return self.__dict__.items()
    
    def __repr__(self) -> str:
        return f"Mapper(cols={list(self.__dict__.keys())})"    
    
    
    
class MeanTargetEncoder(MonoTransformer):
    """Applies Mean Target Encoding to selected columns in a dataset.
    """
    def __init__(self, alpha: int = 5):
        """
        Args:
            alpha: a smoothing parameter corresponding to a global sample size
        """
        self.alpha : int = alpha
        self.mapper : MTEMapper = MTEMapper()
        self.global_mean: float = None
        self._fitted: bool = False
        self.columns = None

    def get_params(self, deep: bool):
        """returns the parameters used to initialize this MTE object.
        Note: this method is necessary to integrate MTE into a sklearn Pipeline."""
        return {'alpha': self.alpha}

    
    def fit(self, X: pd.DataFrame, y: pd.Series, columns=list[DatasetColumn]) -> None:
        """creates a mapper object. The mapper is a nested dictionary
        where:
        - keys are categorical column names
        - values are a dictionary with:
            - keys being the feature values associated with that column
            - values are the mean target value for the subgroup of observations
            in the data with that column value.
        A smoothing function is applied with `alpha` representing the smoothing
        sample 'size'.
        This mapper object can be used to transform a categorical column in a numerical
        one."""

        self.global_mean = y.mean()
        
        for col in columns:
            # Group by the categorical feature and calculate its properties
            train_groups = y.groupby(X.loc[:,col])
            category_sum = train_groups.sum()
            category_size = train_groups.size()

            # Calculate smoothed mean target statistics
            train_statistics = (category_sum + self.global_mean * self.alpha) / (category_size + self.alpha)
            self.mapper.add_map(col, train_statistics.to_dict())
        
        # set _fitted and columns class attributes
        super().fit(X, y, columns) 
        
        return self
    

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """returns a dataframe similar to the input df, but augmented with
        mean-target encoded categorical features"""
        super().transform(X)
        output = pd.DataFrame()
        for col, train_statistics in self.mapper.items():
            output[col + self.suffix] = X[col].map(train_statistics).fillna(self.global_mean)
        
        return output
    

    def get_feature_names_out(self):
        """Note: necessary method to integrate within Pipeline """
        assert_fitted(self)
        return [cat + self.suffix for cat in self.columns]

    
    def __repr__(self):
        return set_repr(self, ['alpha']) 
    
    
class PassThrough(MonoTransformer):
    def fit(self, X: pd.DataFrame, y: pd.Series, columns=list[DatasetColumn]) -> None:
        super().fit(X, y, columns)
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # validate transformer has been fitted first
        super().transform(X)
        
        return X.loc[:, self.columns]
    
    def get_params(self):
        return {}
    
    def __repr__(self):
        return set_repr(self, [])
    
    
class MonoFromSklearn(MonoTransformer):
    """modifies class behavior to limit its transformation to a defined set of columns"""
    def __init__(self, sk_transformer: TransformerMixin):
        self.sk_transformer = sk_transformer
        self._fitted = False
        
    def fit(self, X: pd.DataFrame, y: pd.Series, columns=list[DatasetColumn] ,*args, **kwargs) -> None:
        """"""
        self.sk_transformer.fit(X.loc[:, columns], y, *args, **kwargs)
        
        # set _fitted and columns class attributes
        super().fit(X, y, columns)
        return self
    
    def transform(self, X: pd.DataFrame):
        """"""
        # validate transformer has been fitted first
        super().transform(X)
        
        return self.sk_transformer.transform(X.loc[:, self.columns])
    
    def get_params(self, deep: bool) -> dict:
        return {'sk_transformer': self.sk_transformer}
    
    def __repr__(self) -> str:
        return f'Mono_{self.sk_transformer}'





        
    