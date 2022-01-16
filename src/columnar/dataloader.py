"""
Factory for loading datasets. the factory defines the processes to download the data on disk, load the data in memory in a dataframe and select the features to use to fit the classification model.
"""
from dataclasses import dataclass
from typing import Callable

import pandas as pd

from . import loaders
from .feature_selection import FeatureSelection

DataLoadingFunc = Callable[[], pd.DataFrame]

# each factory is defined by a tuple of callables. 
# the first is downloading the data
# FACTORY = {
#     'adults': (_load, _select),
# }

@dataclass
class DataLoader:
    root: str
    
    def __post_init__(self):
        # load_func, select_features = FACTORY.get(self.root)
        self._load = eval(f"loaders.{self.root}._load")
        self._select_features = eval(f"loaders.{self.root}._select_features")
    
    def load_data(self) -> pd.DataFrame:
        return self._load()
    
    def get_selected_features(self, df: pd.DataFrame) -> FeatureSelection:
        return self._select_features(df)
        