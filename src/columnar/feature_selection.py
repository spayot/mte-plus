"""
Defines a Feature Selection class to store which features should be treated as categorical / numerical 
and used for training the classifier, and which corresponds to the target feature.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Annotated

import pandas as pd

DatasetColumn = Annotated[str, "column name in a dataset"]

@dataclass
class FeatureSelection:
    categoricals: List[DatasetColumn] = field(default_factory=list)
    numericals: List[DatasetColumn] = field(default_factory=list)
    target: DatasetColumn = None
    
    def select_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """splits a dataset between input features and target output. 
        if target output is not present in dataset, the second element in the tuple is None.
        """
        all_cols = self.categoricals + self.numericals
        y = None if self.target not in df.columns else df[self.target]
        return df[all_cols], y
    