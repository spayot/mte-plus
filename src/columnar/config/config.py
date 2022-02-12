import yaml
from dataclasses import dataclass, is_dataclass


def nested_dataclass(*args, **kwargs):
    """generates a nested dataclass from a nested dictionary"""
    def wrapper(cls):
        cls = dataclass(cls, **kwargs)
        original_init = cls.__init__
        def __init__(self, *args, **kwargs):
            for name, value in kwargs.items():
                field_type = cls.__annotations__.get(name, None)
                if is_dataclass(field_type) and isinstance(value, dict):
                     new_obj = field_type(**value)
                     kwargs[name] = new_obj
            original_init(self, *args, **kwargs)
        cls.__init__ = __init__
        return cls
    return wrapper(args[0]) if args else wrapper


@dataclass
class Paths:
    root: str = './'
    figures: str = 'figures/'
    reports: str = 'reports/'

    
@dataclass    
class MetricsConfig:
    """defines which metrics should be included"""
    include_acc: bool = True
    include_f1: bool = True
    include_auc: bool = True

    
@dataclass
class ClassifiersConfig:
    rf__n_estimators: int = 100 # rf: random forest classifier
    rf__max_depth: int = 10
    lr__max_iter: int = 500     # lr: logistic regression 
    knn__n_neighbors: int = 10  # knn: K-nearest neighbors

    
@dataclass
class TransformersConfig:
    include_mte: bool = True
    include_tfembeddings_single: bool = True
    include_tfembeddings_max2: bool = True
    include_tfembeddings_max50: bool = True
    include_onehot: bool = True
    include_ordinal: bool = True
    mte__alpha: int = 5
    tfclassifier__hidden_size: int = 32
    tfclassifier__dropout_rate: float = .5
        
    
@dataclass
class CrossValidationParams:
    n_splits: int = 5

    
@dataclass
class FitParams:
    tf__epochs: int = 3
    tf__verbose: int = 1

    
@dataclass
class PlotConfig:
    style: str = 'fivethirtyeight'

    
@nested_dataclass
class BenchmarkConfig:
    paths: Paths
    metrics: MetricsConfig
    classifiers: ClassifiersConfig
    transformers: TransformersConfig
    cross_validation: CrossValidationParams
    fit_params: FitParams
    plot: PlotConfig
    
    @classmethod
    def load(cls, filepath: str):
        with open(filepath, 'r') as f:
            cfg = yaml.safe_load(f)
        
        return cls(**cfg)

    
    
    
    