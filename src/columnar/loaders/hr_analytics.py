import os
from typing import Any

import numpy as np
import pandas as pd

def _load(root_path: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(root_path,'data/hr_analytics/hr_analytics.csv'))
    df = df.drop(['enrollee_id'], axis=1)
    df.fillna('missing', inplace=True)
    return df


def _select_features(df: pd.DataFrame) -> dict[str, Any]:   
    numericals = ['city_development_index', 'training_hours']
    target = 'target'
    categoricals = [col for col in df.columns if col not in numericals + [target]]
    return dict(numericals=numericals, categoricals=categoricals, target=target)