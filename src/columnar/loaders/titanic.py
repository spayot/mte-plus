import os
from typing import Any

import numpy as np
import pandas as pd

def _load(root_path: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(root_path,'data/titanic/titanic.csv'), index_col='PassengerId')
    
    df.columns = [col.lower() for col in df.columns]
    df['missing_age'] = df.age.isnull() * 1
    df.age.fillna(df.age.mean(), inplace=True)
    df.fillna('other', inplace=True)
    df.drop(['ticket', 'name'], axis=1, inplace=True)
    df['cabin_type'] = df.cabin.apply(lambda s: s[0])
    
    return df

def _select_features(df: pd.DataFrame) -> dict[str, Any]:   
    numericals = ['age', 'fare']
    target = 'survived'
    categoricals = [col for col in df.columns if col not in numericals + [target]]
    return dict(numericals=numericals, categoricals=categoricals, target=target)
    
