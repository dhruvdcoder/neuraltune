import pandas as pd
from pathlib import Path
from typing import List
import numpy as np
import logging
import argparse
import math
import itertools 
from neuraltune.data_utils import NeuralData
np.random.seed(123)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO)
logger = logging.getLogger(__name__)

KNOBS = [f'k{i}' for i in range(1, 9)] + [f's{i}' for i in range(1, 5)]
LATENCY = 'latency'
METRICS_START = 13


if __name__ =='__main__':
    train_path = '.data/offline_workload.CSV'
    dev_path = '.data/online_workload_B.csv'
    test_path = ['.data/online_workload_C.csv','.data/test.csv']
    pruned_metrics = ['executor.jvm.heap.committed.avg', 'worker_1.Disk_Write_KB/s.sdi', 'worker_1.Disk_Block_Size.sdi2', 'executor.runTime.avg', 'worker_2.Memory_MB.cached', 'mimic_cpu_util', 'worker_1.Paging_and_Virtual_Memory.pgpgout', 'executor.resultSerializationTime.avg', 'driver.LiveListenerBus.numEventsPosted.avg_increase', 'executor.jvm.non-heap.committed.avg_period', 'latency']
    output_path = '.'
    p = NeuralData(type_flag='dev')
    p.read_data('.data')
    for i in range(2):
        count = 0
        for x in p:
            count+=1
        print(i, count)




            

        
