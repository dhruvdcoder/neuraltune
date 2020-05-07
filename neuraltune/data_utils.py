import pandas as pd
import os
from typing import List, Any
import numpy as np
import logging

from joblib import dump, load
import argparse
import math
import itertools
from pathlib import Path
from sklearn.preprocessing import StandardScaler

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


class NeuralData:
    def __init__(self, **parameters: Any) -> None:
        self.set_size = 5
        self.batch_size = 1

        self.type_flag = 'train'
        self.train_data = None
        self.train_labels = None
        self.dev_data = None
        self.dev_labels = None
        self.test_data = None
        self.test_labels = None
        self.map_wid_idxs = dict()
        self.map_idx_wid = dict()
        self.shuffled_idxs = None
        self.index = 0
        self.knob_scaler = StandardScaler(copy=False)

        for param, val in list(parameters.items()):
            setattr(self, param, val)

        self.pruned_metrics: List[str] = parameters.get(
            'pruned_metrics', None) or default_metrics

        if (self.scaler_path is not None) and Path(self.scaler_path).is_file():
            logger.info(f"Loading scaler from {self.scaler_path}")
            self.metrics_scaler = load(self.scaler_path)
        else:
            self.metrics_scaler = StandardScaler(copy=False)
            logger.info("Created a new scaler")
        self.latency_idx = None

    def read_data_train(self, folder_path: str) -> None:
        df = pd.read_csv(os.path.join(folder_path, 'train.csv'))
        pruned_idxs = [
            i for i, name in enumerate(df.columns)

            if name in self.pruned_metrics
        ]
        pruned_idxs = list(range(0, METRICS_START)) + pruned_idxs
        df = df[df.columns[pruned_idxs]]
        self.train_labels = df.columns
        self.latency_idx = [
            i for i, name in enumerate(self.train_labels) if name == LATENCY
        ][0]
        self.train_data = df.to_numpy()
        self.train_data[:, 1:] = self.train_data[:, 1:].astype(float)
        logger.info(f"Finished reading train data")

    def read_data(self, folder_path: str) -> None:
        self.read_data_train(folder_path)

        if self.type_flag != 'train':
            df = pd.read_csv(
                os.path.join(folder_path, f'{self.type_flag}.csv'))
            pruned_idxs = [
                i for i, name in enumerate(df.columns)

                if name in self.pruned_metrics
            ]
            pruned_idxs = list(range(0, METRICS_START)) + pruned_idxs
            df = df[df.columns[pruned_idxs]]
            setattr(self, f'{self.type_flag}_labels', df.columns)
            setattr(self, f'{self.type_flag}_data', df.to_numpy())
        data = getattr(self, f'{self.type_flag}_data')
        data[:, 1:] = data[:, 1:].astype(float)
        self.shuffled_idxs = np.arange(data.shape[0])
        np.random.default_rng().shuffle(self.shuffled_idxs)
        self.create_meta_objects()
        logger.info(f"Finished creating meta objects")
        self.normalize()
        logger.info(f"Finished data reading and normalisation")

    def normalize(self) -> None:
        self.knob_scaler.fit(self.train_data[:, 1:METRICS_START])
        self.metrics_scaler.fit(self.train_data[:, METRICS_START:])
        data = getattr(self, f'{self.type_flag}_data')
        data[:, 1:METRICS_START] = self.knob_scaler.transform(
            data[:, 1:METRICS_START])

        if self.type_flag != 'test':
            data[:, METRICS_START:] = self.metrics_scaler.transform(
                data[:, METRICS_START:])
        setattr(self, f'{self.type_flag}_data', data)
        logger.info(f"Saving metrics_scaler to {self.scaler_path}")
        dump(self.metrics_scaler, self.scaler_path)

        return

    def create_meta_objects(self) -> None:
        data = getattr(self, f'{self.type_flag}_data')

        for id, entry in enumerate(data):
            wid = entry[0]
            l = self.map_wid_idxs.get(wid, [])
            self.map_wid_idxs[wid] = l + [id]
            self.map_idx_wid[id] = wid

    def __iter__(self):
        self.index = 0

        return self

    def __next__(self):
        results = []
        data = getattr(self, f'{self.type_flag}_data')

        if self.index >= data.shape[0]:
            raise StopIteration

        for idx in range(self.index, self.index + self.batch_size):
            if idx < data.shape[0]:
                b = data[idx][1:METRICS_START]

                y = data[idx][self.latency_idx]
                wid = self.map_idx_wid[idx]

                if self.type_flag == 'train':
                    a_idxs = self.map_wid_idxs[wid]
                else:
                    a_idxs = self.map_wid_idxs[wid].copy()
                    a_idxs.remove(idx)
                a_idxs = np.random.choice(a_idxs, self.set_size, replace=False)
                a = data[a_idxs][:, 1:]
                results.append((a, b, y))
                self.index += 1
            else:
                break

        return results


if __name__ == '__main__':
    train_path = '.data/offline_workload.CSV'
    dev_path = '.data/online_workload_B.csv'
    test_path = ['.data/online_workload_C.csv', '.data/test.csv']
    pruned_metrics = [
        'executor.jvm.heap.committed.avg', 'worker_1.Disk_Write_KB/s.sdi',
        'worker_1.Disk_Block_Size.sdi2', 'executor.runTime.avg',
        'worker_2.Memory_MB.cached', 'mimic_cpu_util',
        'worker_1.Paging_and_Virtual_Memory.pgpgout',
        'executor.resultSerializationTime.avg',
        'driver.LiveListenerBus.numEventsPosted.avg_increase',
        'executor.jvm.non-heap.committed.avg_period', 'latency'
    ]
    output_path = '.'
    p = NeuarlData(
        train_path=train_path,
        dev_path=dev_path,
        test_path=test_path,
        pruned_metrics=pruned_metrics,
        output_path=output_path)

    for i in range(2):
        count = 0

        for x in p:
            count += 1
        print(i, count)
