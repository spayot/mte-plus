"""
TO DO: generalize df_to_dataset to handle dataframes, without targets.
"""

from typing import Union, Optional

import pandas as pd
import tensorflow as tf

from ..feature_selection import DatasetColumn



def df_to_dataset(dataframe: pd.DataFrame, 
                  labels: Optional[pd.Series] = None,
                  shuffle: bool = True,
                  batch_size: int = 32) -> tf.data.Dataset:
    """transforms a pandas dataframe into a tensorflow dataset"""
    df = dataframe.copy()
    
    # turn dataframe into dictionary
    df = {key: value.values.reshape(-1,1) for key, value in df.items()}
    
    if labels is None:
        data = df
    else:
        data = (df, labels)
    
    ds = tf.data.Dataset.from_tensor_slices(data)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(batch_size)
    return ds