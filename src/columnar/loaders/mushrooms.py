import os
from typing import Any

import numpy as np
import pandas as pd

def _load(root_path: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(root_path,'data/mushrooms/mushrooms.csv'))
    df['class_ep'] = df['class'].map({'p': 0, 'e': 1})
    df.columns = [col.replace('-', '_') for col in df.columns]
    df.drop(['class'], axis=1, inplace=True)
    return df

def _select_features(df: pd.DataFrame) -> dict[str, Any]:
    return dict(
        categoricals=list(df.columns.drop('class_ep')),
        numericals= [],
        target='class_ep',
    )
    
