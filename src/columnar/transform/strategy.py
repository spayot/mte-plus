from dataclasses import dataclass



from ..feature_selection import DatasetColumn
from .mono import MonoTransformer


@dataclass
class MonoStrategy:
    """defines on which columns a MonoTransformer is applied."""
    name: str
    transformer: MonoTransformer
    columns: list[DatasetColumn]


class TransformerStrategy:
    def __init__(self, transformations: list[MonoStrategy]):
        """defines all the transformations to be performed as a 
        list of SubStrategies"""
        self.transformations = transformations            
        self.mapping = self.build_map()
        
    @classmethod
    def from_tuples(cls, *tuples: list[tuple[str, MonoTransformer, list[DatasetColumn]]]):
        """each tuple must correspond to the initializer values of a SubStrategy object.
        
        Example: 
        >>> strategy = TransformerStrategy.from_tuples(('cats', OneHotEncoder(), ['cat_feature1', 'cat_feature2']))
        """
        return cls([MonoStrategy(name, transformer, columns) for name, transformer, columns in tuples])
    
    def __iter__(self):
        return self.transformations.__iter__()
    
    def to_dict(self):
        """returns a dictionary with transformation labels as keys and 
        a tuple (MonoTransformer, list of columns) as values"""
        d = dict()
        for t in self.transformations:
            d[t.name] = (t.transformer, t.columns)
        return d
    
    def build_map(self) -> dict[DatasetColumn, str]:
        """returns a dictionary with dataset columns as keys and
        transformer repr as a value."""
        _map = dict()
        for t in self.transformations:
            for column in t.columns:
                _map[column] = str(t.transformer)
        
        return _map

    def __getitem__(self, column: DatasetColumn) -> str:
        return self.mapping[column]
    
    def __repr__(self) -> str:
        return "TransformerStrategy(\n\t" + "\n\t".join([str(t) for t in self.transformations]) + ")"