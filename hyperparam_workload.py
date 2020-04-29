from map_workload import Workload
import pandas as pd
from pathlib import Path
from typing import List
import numpy as np
import logging
import argparse
from fa import get_parser as fa_parser
from data import Data
from pathlib import Path
from server.analysis.factor_analysis import FactorAnalysis
from server.analysis.cluster import KMeansClusters, create_kselection_model
import wandb
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO)
logger = logging.getLogger(__name__)


def get_parser():
    parser = fa_parser()
    parser.add_argument('--length_scale', type=float, default=0.73)
    parser.add_argument('--output_variation', type=float, default=1.2)
    parser.add_argument('--noise', type=float, default=0.49)
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--threshold', type=float, default=0)
    parser.add_argument('--no_wandb', action='store_true')
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    np.random.seed(123)
    if not args.no_wandb:
        wandb.init()
    data = Data()
    data.read(args.input_data)
    data.preprocess()
    fa_model = FactorAnalysis()
    fa_model.fit(data.X, data.metric_names, n_components=args.n_components)
    components = fa_model.components_.T.copy()
    kmeans_models = KMeansClusters()
    kmeans_models.fit(
        components,
        min_cluster=1,
        max_cluster=10,
        sample_labels=data.metric_names,
        estimator_params=dict(n_init=args.kmeans_runs))
    gap_k = create_kselection_model("gap-statistic")
    gap_k.fit(components, kmeans_models.cluster_map_)
    logger.info(f"Optimal clusters is {gap_k.optimal_num_clusters_}")
    tempList = list(kmeans_models.cluster_map_[
        gap_k.optimal_num_clusters_].get_closest_samples())
    pruned_metrics = tempList + ['latency'] if 'latency' not in tempList else tempList

    #pruned_metrics = ['executor.jvm.heap.committed.avg', 'worker_1.Disk_Write_KB/s.sdi', 'worker_1.Disk_Block_Size.sdi2', 'executor.runTime.avg', 'worker_2.Memory_MB.cached', 'mimic_cpu_util', 'worker_1.Paging_and_Virtual_Memory.pgpgout', 'executor.resultSerializationTime.avg', 'driver.LiveListenerBus.numEventsPosted.avg_increase', 'executor.jvm.non-heap.committed.avg_period', 'latency']
    logger.info(f"pruned metrics: {pruned_metrics}")
    w = Workload(length_scale = args.length_scale, output_variation=args.output_variation,
                 pruned_metrics = pruned_metrics, noise=args.noise, n_jobs=args.workers,
                 topk=args.topk, threshold=args.threshold, chunk_size=args.chunk_size,
                 pool_workers=args.pool_workers)
    w.read(args.input_data)
    w.preprocess()
    w.train_models()
    w.read_dev_set(args.dev_data)
    mse = w.compute_score()
    logger.info(f'MSE: {mse}')
    if not args.no_wandb:
        wandb.log({'mse': mse})
