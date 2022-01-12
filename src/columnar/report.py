from typing import List, Any

import numpy as np
import pandas as pd


class Scorer:
    """defines set of scoring functions to apply."""
    def __init__(self, **kwargs):
        self.scoring_fcts = kwargs
            
    def score(self, y_test: np.ndarray, y_preds: np.ndarray) -> pd.Series:
        """Calculates each scores given 2 arrays of results."""
        return {label: fct(y_test, y_preds) for label, fct in self.scoring_fcts.items()}
    
    def __repr__(self) -> str:
        return f"Scorer(scoring_fcts=[{','.join(self.scoring_fcts.keys())}])"
            
            

class Report:
    """stores results of various experiments.
    each entry is defined by a config and a set of results.
    
    Attributes:
    - report (pd.DataFrame)
    - columns ()"""
    
    def __init__(self, scorer: Scorer):
        self.report = pd.DataFrame()
        self.columns_to_show = None
        self.scorer = scorer
    
    
    def add_to_report(self, config: dict[str, Any], results: pd.Series) -> None:
        """add an entry to the report. each entry is defined by a config and a 
        set of results from the reports scorer"""
        data = config.copy()
        data.update(results.to_dict())
        self.report = self.report.append(data, ignore_index=True)
        
        
    def show(self) -> pd.DataFrame:
        if self.columns_to_show is None:
            return self.report
        return self.report[self.columns_to_show]
    
    
    def set_columns_to_show(self, columns: List[str]) -> None:
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