"""
Defines the CategoricalEncoder Interface as well as the MeanTargetEncoder implementation.
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict

import pandas as pd

from . import feature_selection

class Mapper:
    """a Mapper object.
        each attribute"""
    def add_map(self, column: feature_selection.DatasetColumn, map: Dict[str, float]) -> None:
        setattr(self, column, map)
        
    def items(self):
        return self.__dict__.items()
    
    def __repr__(self) -> str:
        return f"Mapper(cols={list(self.__dict__.keys())})"
    

class CategoricalEncoder(ABC):
    """Abstract class for categorical encoders"""
    suffix: str = '_'
    @abstractmethod
    def get_params(self, deep: bool):
        pass
    
    @abstractmethod
    def fit(self, df: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        pass
    
    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass
    
    @abstractmethod
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass
    

class MeanTargetEncoder(CategoricalEncoder):
    """Allows to apply Mean Target Encoding to categorical """
    def __init__(self, 
                 features: feature_selection.FeatureSelection, 
                 alpha: int = 5):
        self.mapper : Mapper = Mapper()
        self.global_mean = None
        self.features = features
        self.categoricals = features.categoricals
        self.target = features.target
        self.status = 'not_fitted'
        self.alpha = alpha

    def get_params(self, deep: bool):
        """returns the parameters used to initialize this MTE object.
        Note: this method is necessary to integrate MTE into a sklearn Pipeline."""
        return {'features': self.features, 'alpha': self.alpha}

    def fit(self, df: pd.DataFrame, y: Optional[pd.Series] = None):
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
        if not isinstance(y, pd.Series):
            y = df[self.target]

        self.global_mean = y.mean()

        for col in self.categoricals:
            # Group by the categorical feature and calculate its properties
            train_groups = y.groupby(df.loc[:,col])
            category_sum = train_groups.sum()
            category_size = train_groups.size()

            # Calculate smoothed mean target statistics
            train_statistics = (category_sum + self.global_mean * self.alpha) / (category_size + self.alpha)
            self.mapper.add_map(col, train_statistics.to_dict())
            self.status = 'fitted'

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """returns a dataframe similar to the input df, but augmented with
        mean-target encoded categorical features, as """
        assert self.status == 'fitted', "model not fitted"
        output = pd.DataFrame()
        for col, train_statistics in self.mapper.items():
            output[col + self.suffix] = df[col].map(train_statistics).fillna(self.global_mean)
        return output

    def fit_transform(self, df: pd.DataFrame,
                      y: Optional[pd.Series] = None) -> pd.DataFrame:
        """fit and transform the data in one go."""
        self.fit(df, y)
        return self.transform(df)

    def get_feature_names(self):
        """Note: necessary method to integrate within Pipeline """
        return [cat + self.suffix for cat in self.categoricals]

    def __repr__(self):
        return f"MeanTargetEncoder(target={self.target}, alpha={self.alpha})"