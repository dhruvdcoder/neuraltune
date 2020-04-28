from typing import List, Iterable
from allennlp.data import DatasetReader
from allennlp.data import Instance
from allennlp.data.fields import ArrayField
from pathlib import Path
from .data_utils import NeuralData
import numpy as np
import logging
logger = logging.getLogger(__name__)


@DatasetReader.register('neuraltune-reader')
class NeuraltuneReader(DatasetReader):
    def __init__(self, set_size: int = 5, train: bool = True) -> None:
        super().__init__(lazy=train)
        self.train = train
        self.lazy_reader = NeuralData(set_size=set_size)
        self.done_read = False

    def _read(self, path: str) -> Iterable[Instance]:
        if not self.done_read:
            logger.info(f"Reading data from {path}")
            self.lazy_reader.read_data(path)

        for sample in self.lazy_reader:
            a, b, y = sample[0]
            breakpoint()
            a_f = ArrayField(a, dtype=np.float)
            b_f = ArrayField(b, dtype=np.float)

            if y is not None:
                yield Instance(
                    dict(a=a_f, b=b_f, y=ArrayField(y, dtype=np.float)))
            else:
                assert not self.train
                yield Instance(dict(
                    a=a_f,
                    b=b_f,
                ))
