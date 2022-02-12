"""various functions to load transformers, classifiers and define metrics based on config file"""
from typing import Callable

# classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from lightgbm import LGBMClassifier

# transformers
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# metrics
from sklearn import metrics

# project-specific transformers
from .config import BenchmarkConfig
from ..encode import MeanTargetEncoder, FilteredCategoricalTransformer
from ..embeddings import TFEmbeddingWrapper


def get_transformers_from_config(cfg: BenchmarkConfig) -> list[TransformerMixin]:
    TRANSFORMER_OPTIONS = {
        'onehot': FilteredCategoricalTransformer(cat_encoder=OneHotEncoder(handle_unknown='ignore')),
        'ordinal': FilteredCategoricalTransformer(cat_encoder=OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
        'tfembeddings_single': TFEmbeddingWrapper(emb_size_strategy='single'),
        'tfembeddings_max2': TFEmbeddingWrapper(emb_size_strategy='max2'),
        'tfembeddings_max50': TFEmbeddingWrapper(emb_size_strategy='max50'),
        'mte': MeanTargetEncoder(alpha=cfg.transformers.mte__alpha),
    }
    transformers = []
    for name, transformer in TRANSFORMER_OPTIONS.items():
        if getattr(cfg.transformers, f'include_{name}'):
            transformers.append(transformer)
    return transformers


def get_classifiers_from_config(cfg: BenchmarkConfig) -> list[BaseEstimator]:
    """returns a list of metrics absed on """
    cfgc = cfg.classifiers
    classifiers = [
        RandomForestClassifier(n_estimators=cfgc.rf__n_estimators,
                              max_depth=cfgc.rf__max_depth),
        LogisticRegression(max_iter=cfgc.lr__max_iter),
        KNeighborsClassifier(n_neighbors=cfgc.knn__n_neighbors),
        LGBMClassifier(),
    ]
    return classifiers

def get_metrics_from_config(cfg: BenchmarkConfig) -> dict[str, Callable]:
    METRICS_OPTIONS = dict(
        acc=lambda x, y: metrics.accuracy_score(x,y>.5),
        f1=lambda x, y: metrics.f1_score(x,y>.5),
        auc=metrics.roc_auc_score)
    
    m = dict()
    for name, metric_fcn in METRICS_OPTIONS.items():
        if getattr(cfg.metrics, f'include_{name}'):
            m[name] = metric_fcn
    return m