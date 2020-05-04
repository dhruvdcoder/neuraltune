from sklearn.decomposition import FactorAnalysis
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.datasets import load_digits
import pdb
from sklearn.model_selection import GridSearchCV
from joblib import Parallel, delayed
from functools import partial
import multiprocessing
import tqdm
import argparse
import logging
from data import data, for_fa
from pathlib import Path
import wandb
import math
#from clustering import KMeans
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_score(X, args):
    fa = FactorAnalysis(n_components=args.ncomponents)

    return np.mean(
        cross_val_score(fa, X, verbose=2, cv=args.folds, n_jobs=args.workers))


if __name__ == '__main__':
    wandb.init()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ncomponents',
        type=int,
        required=True,
        help='Number of components param to be passed to FA.')
    parser.add_argument(
        '--workers', type=int, default=5, help='pool size for multiprocessing')
    parser.add_argument('--folds', type=int, default=5, help='K for Kfolds')
    parser.add_argument('--input_file', default='fa_cache.npy')
    args = parser.parse_args()
    logging.info('Loading data!')
    # read inp

    inp = np.load(args.input_file, allow_pickle=True)
    X = inp.item().get('metrics')

    score = compute_score(X.T, args)

    if score == -1 * math.inf:
        score = -1e40
    wandb.log({'score': score})
    logging.info(f'{args.ncomponents} : {score}')
