import re
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

from . import report

def plot_model_encoder_pairs(reporter: report.Report, 
                             metrics: list[str] = None, 
                             figpath: Optional[str] = None) -> None:
    """plots metrics"""
    if metrics is None:
        metrics = list(reporter.scorer.scoring_fcts.keys())
    
    # create figure
    fig, axs = plt.subplots(1, len(metrics), figsize=(len(metrics) * 10,5))
    for ax, metric in zip(axs, metrics):
        # create summary view 
        summary = pd.pivot(reporter.show(), index='model', columns='encoder', values=metric)
        
        # clean up column and index names
        summary.columns = [_get_class_name_from_string(col) for col in summary.columns]
        summary.index = [_get_class_name_from_string(idx) for idx in summary.index]
        summary.plot.bar(ax=ax)
        ax.set_ylim([0.5,1])
        ax.set_xticklabels(summary.index, rotation=0)
        ax.set_title(metric.upper())
    
    if figpath is not None:
        plt.savefig(figpath, transparent=False, facecolor='black');
        
        
def _get_class_name_from_string(string : str) -> str:
    """extracts class name from a repr of an instance.
    Example: 
    >>> s = "RandomForestRegressor(n_estimators=100)"
    >>> _get_class_name_from_string(s)
    "RandomForestRegressor"
    """
    return re.match('[A-Za-z]+', string).group(0)
    
    