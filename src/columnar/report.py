"""
Module dedicated to reporting capabilities.
The Report class allows to track experiments and print summaries.
"""
from typing import Any
from pathlib import Path
import numpy as np
import pandas as pd            

from . import score, model

class Report:
    """tracks results of various experiments and allows to produce 
    simple summary reports. each report entry is defined by a config 
    and a set of results.
    
    Attributes:
    - report (pd.DataFrame)
    - columns_to_show (list[str])
    - scorer (Scorer)
    
    Methods:
    - add_to_report(config, results): appends a Report object with a new 
    experiment defined by its config and its results
    """
    
    def __init__(self, scorer: score.Scorer):
        self.report = pd.DataFrame()
        self.columns_to_show : list[str] = None
        self.scorer = scorer
    
    
    def add_to_report(self, 
                      config: model.Config, 
                      results: pd.DataFrame, 
                      show: bool = True) -> None:
        """add an entry to the report. each entry is defined by a config and a 
        set of results from the reports scorer"""
        data = config.copy()
        # add mean values
        data.update(results.mean().to_dict())
        
        # add std deviation
        stds = results.std().to_dict()
        data.update({k + '-std': v for k, v in stds.items()})
        
        # add to report
        self.report = self.report.append(data, ignore_index=True)
        
        if show:
            display(self.show())
        
        
    def show(self) -> pd.DataFrame:
        if self.columns_to_show is None:
            return self.report
        return self.report[self.columns_to_show]
    
    
    def set_columns_to_show(self, columns: list[str]) -> None:
        """defines the columns in the report that should be displayed when calling the 
        .show() method"""
        if len(self.report) > 0:
            not_existing_cols = [col for col in columns if col not in self.report.columns]
            assert not_existing_cols, f"{not_existing_cols} are not valid column names"
        self.columns_to_show = columns
        
        
    def score(self, y_test: np.ndarray, y_preds: np.ndarray) -> pd.Series:
        return self.scorer.score(y_test, y_preds)
    
    
    def __repr__(self) -> str:
        return f"""Report(
        scorer: {self.scorer}, 
        to_show: [{','.join(self.columns_to_show)}]
        )"""
    
