from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.metrics import pairwise_distances_argmin_min
from tqdm import tqdm
import logging
import argparse
from pathlib import Path
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def kmeans(X, args):
    distortions = []
    inertias = []
    mapping1 = {}
    mapping2 = {}
    K = range(1, args.max_k)
    for k in tqdm(K):
        # Building and fitting the model
        kmeanModel = KMeans(n_clusters=k,n_jobs=args.jobs).fit(X)
        kmeanModel.fit(X)
        distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_,
                                        'euclidean'), axis=1)) / X.shape[0])
        inertias.append(kmeanModel.inertia_)

        mapping1[k] = kmeanModel.inertia_
        mapping2[k] = sum(np.min(cdist(X, kmeanModel.cluster_centers_,
                     'euclidean'),axis=1)) / X.shape[0]

    plt.figure(1)
    plt.plot(K, inertias, 'bx-')
    plt.xlabel('Values of K')
    plt.ylabel('Inertia')
    plt.title('The Elbow Method using Inertia')
    plt.savefig("elbow_inertia.png")

    plt.figure(2)
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('Values of K')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method using Distortion')
    plt.savefig("elbow_distortion.png")


def get_centers(X,args):
    kmeans_ = KMeans(n_clusters=args.opt_k, random_state=0, n_jobs=args.jobs).fit(X)
    logger.info('labels',kmeans_.labels_)
    #logger.info(kmeans_.cluster_centers_)
    idx, dist = pairwise_distances_argmin_min(kmeans_.cluster_centers_, X)
    logger.info('Centres shape',idx.shape)
    np.save(args.output.with_suffix('.npy'), idx)
    return idx

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_k', type=int, default=11,
            help='Max centres to be considered for clustering')   
    parser.add_argument('--step', type=int, default=1,
            help='Step size for grid')   
    parser.add_argument('--chunk', type=int, default=2,
            help='chunk size for imap multiptocessing') 
    parser.add_argument('--workers', type=int, default=5,
            help='pool size for multiprocessing')
    parser.add_argument('--jobs', type=int, default=1,
            help='cpus to assign to each param evaluation')
    parser.add_argument('--opt_k', type=int, default=3,
            help='Optimal k for cluster centers')    
    parser.add_argument('--output', type=Path, default='output_centres',
            help='Filepath to save metric indices')        
    parser.add_argument('--datapath', type=Path, default='output.npy',
            help='Filepath to read x_transformed')     
    parser.add_argument('--cache', type=Path, default='fa_cache',
            help='Filepath to save transformed X')
    parser.add_argument('--elbow', action='store_true', 
            help='Set flag to visualize clustering on range of k to find ideal number of centres')
    parser.add_argument('--kmeans', action='store_true', 
            help='Set flag to perform KMeans')
            
    args = parser.parse_args()
    logger.info('Loading data X')
    X = np.load(args.datapath)
    if args.elbow:
        logger.info(f'Calling kmeans to visualize distorion inertia plots for k upto {args.max_k}')
        kmeans(X, args)
    if args.kmeans:
        logger.info(f'Calling get_centres to get centre labels')
        get_centers(X, args)
    
