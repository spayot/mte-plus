import re
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

from . import report, model

def plot_model_encoder_pairs(reporter: report.Report, 
                             metrics: list[str] = None, 
                             figpath: Optional[str] = None,
                             title: Optional[str] = None,
                             show: bool = True,
                            ) -> plt.Figure:
    """plots metrics"""
    if metrics is None:
        metrics = list(reporter.scorer.scoring_fcts.keys())
    
    
    # create figure
    fig, axs = plt.subplots(1, len(metrics), figsize=(len(metrics) * 10,5))
    for ax, metric in zip(axs, metrics):
        # create summary view for mean value of this metric during cross validation
        summary = pd.pivot(reporter.report, index='classifier', columns='transformer', values=metric)
        
        # get std dev of this metric across cross validation
        err = pd.pivot(reporter.report, index='classifier', columns='transformer', values=metric + '-std')
        
        summary = _clean_index_column_names(summary)
        err = _clean_index_column_names(err)
        
        
        for table in [summary, err]:
            # clean up column and index names
            table.columns = [_get_class_name_from_string(col) for col in table.columns]
            table.index = [_get_class_name_from_string(idx) for idx in table.index]
        summary.plot.bar(ax=ax, yerr=err)
        ax.set_ylim([0.3,1])
        ax.set_xticklabels(summary.index, rotation=0)
        ax.set_title(metric.upper())
        ax.get_legend().remove()
        
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left')
    if title is not None:
        fig.suptitle(title, y=1.1)
    
    if figpath is not None:
        plt.savefig(figpath, transparent=False, facecolor='white');
    
    if show:
        plt.show()
    
    return fig
        
        
def _get_class_name_from_string(string : str) -> str:
    """extracts class name from a repr of an instance.
    Example: 
    >>> s = "RandomForestRegressor(n_estimators=100)"
    >>> _get_class_name_from_string(s)
    "RandomForestRegressor"
    """
    return re.match('[A-Za-z_]+', string).group(0)

def _clean_index_column_names(df: pd.DataFrame) -> None:
    table = df.copy()
    table.columns = [_get_class_name_from_string(col) for col in table.columns]
    table.index = [_get_class_name_from_string(idx) for idx in table.index]
    
    return table
    
    
    
    
def plot_feature_importance(pipe: model.CategoricalPipeline, *args, **kwargs):
    fi = (pd.Series(pipe.model.feature_importances_, 
                   index=pipe.features.categoricals + pipe.features.numericals)
          .sort_values())
    fi.plot.barh(*args, **kwargs)
    plt.show()