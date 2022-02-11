"""
Defines the CategoricalEncoder Interface as well as the MeanTargetEncoder implementation.
"""
from abc import ABC, abstractmethod
from typing import Optional, Any

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer

from .feature_selection import FeatureSelection, DatasetColumn
from .utils import set_repr

class Mapper:
    """a Mapper object.
        each attribute"""
    def add_map(self, column: DatasetColumn, map: dict[str, float]) -> None:
        setattr(self, column, map)
        
    def items(self):
        return self.__dict__.items()
    
    def __repr__(self) -> str:
        return f"Mapper(cols={list(self.__dict__.keys())})"

class CategoricalTransformer(TransformerMixin, ABC):
    """Abstract class for categorical encoders"""
    suffix: str = '_'
    fitted: bool = False
    @abstractmethod
    def get_params(self, deep: bool):
        pass
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series,
            features: FeatureSelection) -> None:
        pass
    
    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        assert self.fitted == True, "Model needs to be fitted before transformation"
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series,
            features: FeatureSelection) -> pd.DataFrame:
        return self.fit(X, y, features).transform(X)
    

class MeanTargetEncoder(CategoricalTransformer):
    """Applies Mean Target Encoding to categorical data and passes through numerical ones."""
    def __init__(self, 
                 alpha: int = 5):
        """"""
        self.alpha = alpha
        self.mapper : Mapper = Mapper()
        self.global_mean: float = None
        self.fitted: bool = False
        self.features: FeatureSelection = None
        

    def get_params(self, deep: bool):
        """returns the parameters used to initialize this MTE object.
        Note: this method is necessary to integrate MTE into a sklearn Pipeline."""
        return {'alpha': self.alpha}

    
    def fit(self, X: pd.DataFrame, y: pd.Series,
            features: FeatureSelection) -> None:
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
        self.features = features
        
        for col in features.categoricals:
            # Group by the categorical feature and calculate its properties
            train_groups = y.groupby(X.loc[:,col])
            category_sum = train_groups.sum()
            category_size = train_groups.size()

            # Calculate smoothed mean target statistics
            train_statistics = (category_sum + self.global_mean * self.alpha) / (category_size + self.alpha)
            self.mapper.add_map(col, train_statistics.to_dict())
            
        self.fitted = True
            
        return self
    

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """returns a dataframe similar to the input df, but augmented with
        mean-target encoded categorical features"""
        super().transform(X)
        output = pd.DataFrame()
        for col, train_statistics in self.mapper.items():
            output[col + self.suffix] = X[col].map(train_statistics).fillna(self.global_mean)
        
        # pass through numerical features
        for col in self.features.numericals:
            output[col] = X[col]
        
        return output
    

#     def fit_transform(self, X: pd.DataFrame, y: pd.Series,
#                       features: FeatureSelection) -> pd.DataFrame:
#         """fit and transform the data in one go."""
        
#         return self.fit(X, y, features).transform(X)
    

    def get_feature_names(self):
        """Note: necessary method to integrate within Pipeline """
        return [cat + self.suffix for cat in self.features.categoricals]

    
    def __repr__(self):
        return set_repr(self, ['alpha'])
    

    
    
class TransformStrategy(CategoricalTransformer):
    """allows to apply a transformation strategy only to
    categorical columns, while numerical columns are passed through"""
    def __init__(self, cat_encoder: TransformerMixin):
        self.features = None
        self.cat_encoder = cat_encoder
        self.fitted = False
        
    
    def fit(self, X: pd.DataFrame, y: pd.Series, 
            features: FeatureSelection):
        """fits the categorical encoder to the categorical features.
        creates a passthrough strategy for all other features."""
        self.features = features
        self.transformer = ColumnTransformer(
            [('categories', self.cat_encoder, features.categoricals)], 
            remainder='passthrough')
        self.fitted = True
        
        return self.transformer.fit(X, y)
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.transformer.transform(X)
    
    def get_params(self, deep: bool = True) -> dict[str, Any]:
        # cat_encoder = clone(self.cat_encoder) if deep else self.cat_encoder
        # cat_encoder = self.cat_encoder
        return {'cat_encoder': self.cat_encoder}
    
    def __repr__(self) -> str:
        return f"TransformStrategy_{self.cat_encoder}()"
        
    