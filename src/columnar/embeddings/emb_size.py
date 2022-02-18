from typing import Callable, Annotated

# define method to choose the size of the embedding based on the cardinality
EmbSizeStrategyName = Annotated[str, "name of the strategy to implement"]


class EmbSizeFactory:
    def __init__(self):
        self._builders = {}

    def register_builder(self, 
                         key: EmbSizeStrategyName, 
                         builder: Callable[[int], int]):
        self._builders[key] = builder

    def calculate_emb_size(self, key, cardinality):
        builder = self._builders.get(key)
        if not builder:
            raise ValueError(key)
        return builder(cardinality)


emb_size_factory = EmbSizeFactory()

emb_size_factory.register_builder(key='single', builder=lambda cardinality: 1)
emb_size_factory.register_builder(key='max2', builder=lambda cardinality: min(cardinality // 2, 2))
emb_size_factory.register_builder(key='max50', builder=lambda cardinality: min(cardinality // 2, 50))