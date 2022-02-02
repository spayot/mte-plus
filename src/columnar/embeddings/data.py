from typing import Union

import pandas as pd
import tensorflow as tf

from ..feature_selection import DatasetColumn

def df_to_dataset(dataframe: pd.DataFrame, 
                  target: Union[DatasetColumn, pd.Series], 
                  shuffle: bool = True,
                  batch_size: int = 32) -> tf.data.Dataset:
    """transforms a pandas dataframe into a tensorflow dataset"""
    df = dataframe.copy()
    if type(target) == pd.Series:
        labels = targets
    else:
        labels = df.pop(target)
    df = {key: value.values.reshape(-1,1) for key, value in df.items()}
    ds = tf.data.Dataset.from_tensor_slices((df, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
        ds = ds.batch(batch_size)
        ds = ds.prefetch(batch_size)
    return ds