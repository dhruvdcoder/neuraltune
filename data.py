import pandas as pd
from pathlib import Path
from typing import List
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler
from server.analysis.preprocessing import Bin, get_shuffle_indices
import logging
logger = logging.getLogger(__name__)

KNOBS = [f'k{i}' for i in range(1, 9)] + [f's{i}' for i in range(1, 5)]
LATENCY = 'latency'
METRICS_START = 13


class Data:
    def __init__(self,
                 remove_duplicates=True,
                 remove_const=True,
                 normalizer=None):
        self.remove_const = remove_const
        self.remove_duplicates = remove_duplicates
        self.all_data: np.ndarray = None
        self.all_metric_names: List[str] = None
        self.X = None
        self.metric_names = None
        self.scaler = StandardScaler(copy=False)
        if normalizer is None:

            self.normalizer = Bin(bin_start=1, axis=0)
            #self.normalizer = StandardScaler(copy=False)
            logger.info("Created default Bin normalizer")
        else:
            self.normalizer = normalizer

    def read(self, path: Path) -> None:
        df = pd.read_csv(path)
        self.all_data = (df[df.columns[METRICS_START:]]).values
        self.all_metric_names = df.columns[METRICS_START:]
        

    def preprocess(self) -> None:
        logger.info(f"Data shape before preprocessing {self.all_data.shape}")
        logger.info(f"Data shape before normalizing {self.all_data.shape}")
        #breakpoint()
        
        scaled_data = self.scaler.fit_transform(self.all_data)
        normal = self.normalizer.fit_transform(scaled_data)
        logger.info(f"Data shape after normalizing {normal.shape}")
        # remove const metrics
        non_const_matrix = []
        non_const_labels = []
        logger.info(f"Data shape before removing const {normal.shape}")

        for col, label in zip(normal.T, self.all_metric_names):
            if np.any(col != col[0]):  # means non-const
                non_const_matrix.append(col.reshape(-1, 1))
                non_const_labels.append(label)
        assert len(non_const_matrix) > 0
        non_const_matrix = np.hstack(non_const_matrix)
        logger.info(f"Data shape after removing "
                    f"const {non_const_matrix.shape}")  # type:ignore

        # remove dup cols
        unique_matrix, unique_idx = np.unique(
            non_const_matrix, axis=1, return_index=True)
        unique_labels = [non_const_labels[idx] for idx in unique_idx]
        # shuffle
        unique_matrix = unique_matrix[get_shuffle_indices(unique_matrix.
                                                          shape[0]), :]
        self.X = unique_matrix
        self.metric_names = unique_labels
        logger.info(f"Data shape after preprocessing {self.X.shape}")


def data(name: str) -> pd.DataFrame:
    input_data = Path(f'.data/{name}.CSV')
    inp_data = pd.read_csv(input_data)

    return inp_data


def for_fa(all_data: pd.DataFrame,
           metrics_start: int = METRICS_START,
           normalise: bool = True) -> np.ndarray:
    """
        Args:
            all_data: Complete DataFrame
            metrics_start: Column number from which the metrics start (def: 13)
        Returns:
            numpy.ndarray of shape (num_metrics, num_configs)
    """
    x = (all_data[all_data.columns[metrics_start:]]).values

    if normalise:
        x = RobustScaler(copy=False).fit_transform(x)

    return x.T


if __name__ == '__main__':
    X = for_fa(data('offline_workload'))
    assert X.shape == (572, 18349)
