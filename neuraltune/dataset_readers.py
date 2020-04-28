from typing import List, Iterable
from allennlp.data import DatasetReader
from allennlp.data import Instance
from pathlib import Path


class NeuraltuneReader(DatasetReader):
    def _read(path: str) -> Iterable[Instance]:
        pass
