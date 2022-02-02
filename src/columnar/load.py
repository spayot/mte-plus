"""
Factory for loading datasets. the factory defines the processes to download the data on disk, load the data in memory in a dataframe and select the features to use to fit the classification model.
"""
from dataclasses import dataclass
from typing import Callable, Any

import pandas as pd

from . import loaders

DataLoadingFunc = Callable[[], pd.DataFrame]


@dataclass
class DataLoader:
    task: str # the task name
    root: str = 'data/'# the root path to data
    
    def __post_init__(self):
        # load_func, select_features = FACTORY.get(self.root)
        self._load = eval(f"loaders.{self.task}._load")
        self._select_features = eval(f"loaders.{self.task}._select_features")
    
    def load_data(self) -> pd.DataFrame:
        return self._load(self.root)
    
    def get_selected_features(self, df: pd.DataFrame) -> dict[str, Any]:
        """returns a dictionary with 3 keys: categoricals, numericals and target.
        this dictionary can be passed as arguments for a FeatureSelection object"""
        return self._select_features(df)
        