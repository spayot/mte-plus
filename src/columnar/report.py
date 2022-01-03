import pandas as pd
from typing import List

class Report:
    def __init__(self):
        self.report = pd.DataFrame()
        self.columns = None
        
    def add_to_report(self, config, results) -> None:
        data = {k: str(v) for k,v in config.__dict__.items()}
        data.update(results.to_dict())
        self.report = self.report.append(data, ignore_index=True)
        
    def show(self) -> pd.DataFrame:
        if self.columns is None:
            return self.report
        return self.report[self.columns]
    
    def set_columns_to_show(self, columns: List[str]) -> None:
        # for col in columns:
        #     assert col in self.report.columns, f"{col} is not a valid column name"
        self.columns = columns