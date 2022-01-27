import os
from typing import Any

import numpy as np
import pandas as pd


FILEPATH = 'data/petfinder/petfinder.csv'
def _load(root_path: str) -> pd.DataFrame:
    # load data
    df = pd.read_csv(os.path.join(root_path, FILEPATH), index_col='PetID')
    
    df.columns = [col.lower() for col in df.columns]
    df.name = df.name.str.lower()
    df = df.fillna('no name')
    df['has_name'] = ~df.name.str.contains('no name', case=False) * 1
    df['target'] = np.where(df['adoptionspeed']==4, 0, 1)
    df = df.drop(columns=['adoptionspeed', 'description', 'name'])
    
    return df

def _select_features(df: pd.DataFrame) -> dict[str, Any]:
    
    return dict(
        categoricals=['type', 'breed1', 'breed2', 'gender', 'color1', 
                      'color2', 'color3', 'maturitysize', 'furlength', 
                      'vaccinated', 'dewormed', 'sterilized', 'health', 
                      'quantity', 'state', 'rescuerid', 'has_name'], 
         numericals=['photoamt', 'videoamt', 'fee', 'age'], 
         target='target')