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
from clustering import kmeans, get_centers
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_score(X, folds, cv_jobs,  n_components ):
    fa = FactorAnalysis(n_components=n_components)
    return np.mean(cross_val_score(fa, X, verbose=2, cv=folds, n_jobs=cv_jobs))

def search(X, folds, max_components ,step_size, chunk_size, pool_workers, cv_jobs):
    score = partial(compute_score, X, folds, cv_jobs)
    grid = list(range(1, max_components, step_size))
    with multiprocessing.Pool(pool_workers) as pool:
        results = list(tqdm.tqdm(pool.imap(score,grid,chunksize=chunk_size), total=len(grid)))
    return grid[np.argmax(results)]

def fa(x, folds=3, max_components=1000,step_size=1, chunk_size=2, pool_workers=5, cv_jobs=1):
    max_components = min(max_components, x.shape[-1])
    n_components_fa = search(x, folds, max_components, step_size=step_size, chunk_size=chunk_size, pool_workers=pool_workers, cv_jobs=cv_jobs)
    logger.info('n_components {}'.format(n_components_fa))
    transformer = FactorAnalysis(n_components=n_components_fa, random_state=123)
    X_transformed = transformer.fit_transform(x)
    return X_transformed

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_comp', type=int, default=1000,
            help='Max components to be considered for FA')   
    parser.add_argument('--step', type=int, default=1,
            help='Step size for grid')   
    parser.add_argument('--chunk', type=int, default=2,
            help='chunk size for imap multiptocessing') 
    parser.add_argument('--workers', type=int, default=5,
            help='pool size for multiprocessing')
    parser.add_argument('--cv_jobs', type=int, default=1,
            help='cpus to assign to each param evaluation')
    parser.add_argument('--folds', type=int, default=3,
            help='K for Kfolds')    
    parser.add_argument('--output', type=Path, default='output',
            help='Filepath to save transformed X')        
    parser.add_argument('--cache', type=Path, default='fa_cache',
            help='Filepath to save transformed X')

            

    args = parser.parse_args()
    #X,_ = load_digits(return_X_y=True)
    logging.info('Loading data!')
    X = for_fa(data('offline_workload'))
    logging.info(f'Shape of data {X.shape}')
    X_transformed = fa(X, folds=args.folds, max_components=args.max_comp, step_size=args.step, chunk_size=args.chunk, pool_workers=args.workers, cv_jobs=args.cv_jobs)
    np.save(args.output.with_suffix('.npy'), X_transformed)
    logger.info('Saved x_transformed to {}'.format(args.output.with_suffix('.npy')))
    logger.info(f'After Factor Analysis, X is of the dimensions {X_transformed.shape}')


