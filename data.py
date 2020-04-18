import pandas as pd
from pathlib import Path
import numpy as np

KNOBS = [f'k{i}' for i in range(1, 9)] + [f's{i}' for i in range(1, 5)]
LATENCY = 'latency'
METRICS_START = 13


def data(name: str) -> pd.DataFrame:
    input_data = Path(f'.data/{name}.CSV')
    inp_data = pd.read_csv(input_data)

    return inp_data


def for_fa(all_data: pd.DataFrame,
           metrics_start: int = METRICS_START) -> np.ndarray:
    """
        Args:
            all_data: Complete DataFrame

            metrics_start: Column number from which the metrics start (def: 13)

        Returns:
            numpy.ndarray of shape (num_metrics, num_configs)

    """

    return (all_data[all_data.columns[metrics_start:]]).values.T


if __name__ == '__main__':
    X = for_fa(data('offline_workload'))
    assert X.shape == (572, 18349)
