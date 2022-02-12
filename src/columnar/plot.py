
import math
import os
import re
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from . import report, pipeline

plt.style.use('fivethirtyeight')



    
def plot_model_encoder_pairs(reporter: report.Reporter, 
                             metrics: list[str] = None, 
                             figpath: Optional[str] = None,
                             title: Optional[str] = None,
                             show: bool = True,
                            ) -> plt.Figure:
    """plots metrics in a bar chart, including error bars"""
    if metrics is None:
        metrics = list(reporter.scorer.scoring_fcts.keys())
    
    # get ylim_min
    ylim_min = get_ylim_min(reporter.report[metrics])
    
    # create figure
    fig, axs = plt.subplots(1, len(metrics), figsize=(len(metrics) * 10,5))
    for ax, metric in zip(axs, metrics):
        # create summary view for mean value of this metric during cross validation
        summary = _get_summary(reporter.report, metric)
        
        # get std dev of this metric across cross validation
        err = _get_summary(reporter.report, metric +'-std')
        
        summary.plot.bar(ax=ax, yerr=err)
        
        # set y limits
        ax.set_ylim([ylim_min,1])
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
    return re.match('[A-Za-z0-9_]+', string).group(0)

def _clean_index_column_names(df: pd.DataFrame) -> None:
    """helper function to simplify the columns and index names in summary tables"""
    table = df.copy()
    table.columns = [_get_class_name_from_string(col) for col in table.columns]
    table.index = [_get_class_name_from_string(idx) for idx in table.index]
    
    return table
    
def get_ylim_min(df: pd.DataFrame) -> float:
    y_min = df.min().min()
    ylim_min = max(0, math.floor(10 * y_min - 1) / 10)
    return ylim_min

def _get_summary(df: pd.DataFrame, metric: str):
    """returns a table with classifier as columns and transformers as rows populated
    with the metric of interest"""
    summary = pd.pivot(df, index='classifier', columns='transformer', values=metric)
    summary = _clean_index_column_names(summary)
    return summary    
    
    
    
#==========================================================================================
# Feature Importance
#==========================================================================================
    
def plot_feature_importance(pipe: pipeline.CategoricalPipeline, *args, **kwargs):
    """note: only works with pipes with a classifier having a 
    feature_importances_ class attribute"""
    fi = (pd.Series(pipe.classifier.feature_importances_, 
                   index=pipe.features.categoricals + pipe.features.numericals)
          .sort_values())
    fi.plot.barh(*args, **kwargs)
    plt.show()
    

    
#==========================================================================================
# Summary Heatmaps 
#==========================================================================================
    
TRANSFORMER_RENAMING = {
    'MeanTargetEncoder': 'MeanTarget',
 "TransformStrategy_OneHotEncoder": 'OneHot',
 "TransformStrategy_OrdinalEncoder": "Ordinal",
 'TFEmbeddingWrapper_SingleStrategy': "Embeddings-1dim",
 'TFEmbeddingWrapper_Max2Strategy': "Embeddings-Max2",
 'TFEmbeddingWrapper_Max50Strategy': "Embeddings-Max50",
}


    
def plot_heatmap(task, metric, clf, ax, reports_path: str, yticks: bool = False) -> None:
    """returns a single column heatmap for hte metric of interest."""
    filepath = os.path.join(reports_path, f'{task}.csv')
    
    report = pd.read_csv(filepath, sep=';')

    summary = _get_summary(report, metric).T
    
    baseline = summary.loc['TransformStrategy_OneHotEncoder', 'LogisticRegression']
    
    summary = summary[[clf]]
    summary.columns = [f'{task:<10}']
    summary.index = summary.index.map(TRANSFORMER_RENAMING)
    
    delta = .05
    sns.heatmap(summary, cmap='vlag', annot=True, fmt='.1%', cbar=False, ax=ax, center=baseline, vmin=baseline -delta, vmax=baseline + delta)
    
    if not yticks:
        ax.set_yticklabels([])
        
        
def generate_heatmaps(tasks: list[str], 
                      metric: str, 
                      reports_path: str,
                      figures_path: str,
                      clfs: list[str] = None) -> None:
    """generate a heatmap for each separate classifier."""
    
    if clfs is None:
        clfs = ['KNeighborsClassifier', 'LGBMClassifier', 'LogisticRegression', 'RandomForestClassifier']
    
    for j, clf in enumerate(clfs):
        fig, ax = plt.subplots(1,5, figsize=(5,4))
        
        for i, task in enumerate(tasks):
            plot_heatmap(task, metric, clf=clf, ax=ax[i], yticks=(i==0) & (j==0), reports_path=reports_path)

        plt.subplots_adjust(wspace=0, hspace=0)
        ax[2].set_title(clf)

        # save figure
        figpath = os.path.join(figures_path, f'heatmap_{metric}_{clf}.png')
        plt.savefig(figpath, transparent=False, facecolor='white')
        plt.show()