from typing import List, Iterable
from allennlp.data import DatasetReader
from allennlp.data import Instance
from allennlp.data.fields import ArrayField, MetadataField
from pathlib import Path
from .data_utils import NeuralData
import numpy as np
from joblib import dump, load

import pickle
import pandas as pd
import logging
logger = logging.getLogger(__name__)

np.random.seed(123)
logger = logging.getLogger(__name__)

KNOBS = [f'k{i}' for i in range(1, 9)] + [f's{i}' for i in range(1, 5)]
LATENCY = 'latency'
METRICS_START = 13

default_metrics = [
    'executor.jvm.heap.committed.avg', 'worker_1.Disk_Write_KB/s.sdi',
    'worker_1.Disk_Block_Size.sdi2', 'executor.runTime.avg',
    'worker_2.Memory_MB.cached', 'mimic_cpu_util',
    'worker_1.Paging_and_Virtual_Memory.pgpgout',
    'executor.resultSerializationTime.avg',
    'driver.LiveListenerBus.numEventsPosted.avg_increase',
    'executor.jvm.non-heap.committed.avg_period', 'latency'
]


@DatasetReader.register('neuraltune-reader')
class NeuraltuneReader(DatasetReader):
    def __init__(self,
                 pruned_metrics: List,
                 set_size: int = 5,
                 type_flag: str = 'train',
                 scaler_path: str = None) -> None:
        super().__init__(lazy=True)
        self.type_flag = type_flag

        if scaler_path is None:
            scaler_path = 'metrics_scaler__' + '__'.join(
                pruned_metrics) + '.pkl'

        self.lazy_reader = NeuralData(
            set_size=set_size,
            type_flag=type_flag,
            scaler_path=scaler_path,
            pruned_metrics=pruned_metrics)
        self.done_read = False

    def _read(self, path: str) -> Iterable[Instance]:
        if not self.done_read:
            logger.info(f"Reading data from {path}")
            self.lazy_reader.read_data(path)
            self.done_read = True

        for sample in self.lazy_reader:
            a, b, y = sample[0]
            # breakpoint()
            a_f = ArrayField(a.astype('float'), dtype=np.float32)
            b_f = ArrayField(b.astype('float'), dtype=np.float32)

            if y is not None:
                yield Instance(
                    dict(
                        a=a_f,
                        b=b_f,
                        y=ArrayField(
                            np.array(y).astype('float'), dtype=np.float32)))
            else:
                assert not self.train
                yield Instance(dict(
                    a=a_f,
                    b=b_f,
                ))


@DatasetReader.register('neuraltune-static-reader')
class NeuraltuneStaticReader(DatasetReader):
    def _read(self, path: str) -> Iterable[Instance]:
        with open(path, 'rb') as f:
            instances = pickle.load(f)

        return instances


@DatasetReader.register('neuraltune-test_reader')
class NeuraltuneTestReader(DatasetReader):
    def __init__(
            self,
            scaler_path: str,
            pruned_metrics: List,
            set_size: int = 5,
    ):
        super().__init__(lazy=False)
        self.pruned_metrics = pruned_metrics
        self.set_size = set_size
        self.scaler_path = scaler_path
        self.metrics_scaler = load(self.scaler_path)
        self.knob_scaler = load('knob_scaler.pkl')

    def _read(self, path: str):
        path_a = Path(path) / 'online_workload_C.CSV'
        path_b = Path(path) / 'test.CSV'
        df_a = pd.read_csv(path_a)
        df_b = pd.read_csv(path_b)
        pruned_idxs = [
            i for i, name in enumerate(df_a.columns)

            if name in self.pruned_metrics
        ]
        pruned_idxs = list(range(0, METRICS_START)) + pruned_idxs
        df_a = df_a[df_a.columns[pruned_idxs]]

        for idx, row in df_b.iterrows():
            w_id = row['workload id']
            a = df_a.loc[df_a['workload id'] == row['workload id']]
            a = a[a.columns[1:]].to_numpy().astype('float')
            b = row[df_b.columns[1:-1]].to_numpy().astype('float')
            # normalize
            a[:, :METRICS_START - 1] = self.knob_scaler.transform(
                a[:, :METRICS_START - 1])
            a[:, METRICS_START - 1:] = self.metrics_scaler.transform(
                a[:, METRICS_START - 1:])
            b = self.knob_scaler.transform(b.reshape(1, -1)).reshape(-1)
            yield Instance(
                dict(
                    a=ArrayField(a, dtype=np.float32),
                    b=ArrayField(b, dtype=np.float32),
                    meta=MetadataField(dict(workload_id=w_id))))
