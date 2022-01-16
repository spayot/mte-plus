"""
defines process to load Adults dataset and select features to fit a classifier on.
"""
import os
from typing import Any

import numpy as np
import pandas as pd


from ..feature_selection import FeatureSelection

DATASET_PATH = 'data/adults/adult.data'

def _load(root_path: str) -> pd.DataFrame:
    """load Adults dataset and transform columns as needed"""
    path = os.path.join(root_path, DATASET_PATH)
    COL_NAMES = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital-status', 'occupation',
             'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours-per-week', 'native-country', 'income']
    df = pd.read_csv(path, header=None)
    df.columns = COL_NAMES
    df['target'] = (df.income.str.strip() == '>50K')
    
    return df
    
    
def _select_features(df: pd.DataFrame) -> dict[str, Any]:
    numericals = df.select_dtypes(np.number).columns.tolist()
    categoricals = df.select_dtypes('object').columns.tolist()
    categoricals.remove('income')

    # select features to use in the model
    feature_selection = dict(
        categoricals=categoricals,
        numericals=numericals,
        target= 'target')

    return feature_selection
