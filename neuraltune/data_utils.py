import pandas as pd
from pathlib import Path
from typing import List, Any
import numpy as np
import logging
import argparse
import math
import itertools
from sklearn.preprocessing import StandardScaler

np.random.seed(123)
logger = logging.getLogger(__name__)

KNOBS = [f'k{i}' for i in range(1, 9)] + [f's{i}' for i in range(1, 5)]
LATENCY = 'latency'
METRICS_START = 13


class NeuralData:
    def __init__(self, **parameters: Any) -> None:
        self.set_size = 5
        self.batch_size = 1
        self.pruned_metrics: List[str] = [
            'executor.jvm.heap.committed.avg', 'worker_1.Disk_Write_KB/s.sdi',
            'worker_1.Disk_Block_Size.sdi2', 'executor.runTime.avg',
            'worker_2.Memory_MB.cached', 'mimic_cpu_util',
            'worker_1.Paging_and_Virtual_Memory.pgpgout',
            'executor.resultSerializationTime.avg',
            'driver.LiveListenerBus.numEventsPosted.avg_increase',
            'executor.jvm.non-heap.committed.avg_period', 'latency'
        ]

        self.train_data = None
        self.train_labels = None
        self.map_wid_idxs = dict()
        self.map_idx_wid = dict()
        self.shuffled_idxs = None
        self.index = 0
        self.train_flag = True
        self.scaler = StandardScaler(copy=False
                                    )

        for param, val in list(parameters.items()):
            setattr(self, param, val)
        self.latency_idx = None

    def read_data(self, train_path: str) -> None:
        df = pd.read_csv(train_path)
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
        self.shuffled_idxs = np.arange(self.train_data.shape[0])
        np.random.default_rng().shuffle(self.shuffled_idxs)
        logger.info(f"Finished reading data")
        self.create_meta_objects()
        logger.info(f"Finished creating meta objects")

     def read_dev_data(self, dev_path: str) -> None:
        df = pd.read_csv(dev_path)
        # prune
        pruned_idxs = [
            i for i, name in enumerate(df.columns)

            if name in self.pruned_metrics
        ]
        pruned_idxs = list(range(0, METRICS_START)) + pruned_idxs

        df = df[df.columns[pruned_idxs]]
        self.dev_labels = df.columns
        self.dev_latency_idx = [
            i for i, name in enumerate(self.dev_labels) if name == LATENCY
        ][0]
        self.dev_data = df.to_numpy()
        logger.info(f"Finished reading data")

    def create_meta_objects(self) -> None:
        for id, entry in enumerate(self.train_data):
            wid = entry[0]
            l = self.map_wid_idxs.get(wid, [])
            self.map_wid_idxs[wid] = l + [id]
            self.map_idx_wid[id] = wid

    def __iter__(self):
        self.index = 0

        return self

    def __next__(self):
        results = []

        if self.index >= self.train_data.shape[0]:
            raise StopIteration

        for idx in range(self.index, self.index + self.batch_size):
            if idx < self.train_data.shape[0]:
                b = np.concatenate([
                    self.train_data[idx][1:self.latency_idx],
                    self.train_data[idx][self.latency_idx + 1:]
                ])
                y = self.train_data[idx][self.latency_idx]
                wid = self.map_idx_wid[idx]

                if self.train_flag:
                    a_idxs = self.map_wid_idxs[wid]
                else:
                    a_idxs = self.map_wid_idxs[wid].copy()
                    a_idxs.remove(idx)
                a_idxs = np.random.choice(a_idxs, self.set_size, replace=False)
                a = self.train_data[a_idxs][:, 1:]
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
