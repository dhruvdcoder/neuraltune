import pandas as pd
from pathlib import Path
from typing import List
import numpy as np
import logging
from server.analysis.preprocessing import Bin
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from util import Util
import tqdm
import multiprocessing
import matplotlib.pyplot as plt
logger = logging.getLogger(__name__)

KNOBS = [f'k{i}' for i in range(1, 9)] + [f's{i}' for i in range(1, 5)]
LATENCY = 'latency'
METRICS_START = 13


class Workload:
    def __init__(self, **parameters):
        self.length_scale = None
        self.output_variation = None
        self.X_train = None
        self.y_train = None
        self.models: dict = dict()
        self.y_preds = None
        self.X_scaler = None
        self.y_scaler = None
        self.y_binner = None
        self.metric_names: List[str] = None
        self.pruned_metrics: List[str] = None
        self.knob_labels = KNOBS
        self.row_labels: List[str] = None
        self.unique_workloads_train: List[str] = None
        self.n_jobs = 1
        self.noise = None
        self.topk = 1
        self.threshold = 0
        self.pool_workers = 5
        self.chunk_size = 2
        self.mode = 'train' 
        self.method='baseline'
        for param, val in list(parameters.items()):
            setattr(self, param, val)
        self.X_scaler = self.X_scaler if self.X_scaler else StandardScaler(
            copy=False)
        self.y_scaler = self.y_scaler if self.y_scaler else StandardScaler(
            copy=False)
        self.y_binner = self.y_binner if self.y_binner else Bin(
            bin_start=1, axis=0)
        self.X_dev = None
        self.y_dev = None
        self.row_labels_dev: List[str] = None
        self.unique_workloads_dev: List[str] = None
        self.metric_names_dev: List[str] = None
        self.rbf = None
        self.distances = []
        self.neighbors = []
        

    def get_params(self, deep=True) -> dict:
        return {
            "length_scale": self.length_scale,
            "output_variation": self.output_variation,
            "X_train": self.X_train,
            "y_train": self.y_train
        }

    def read(self, path: Path) -> None:
        logger.info(f'Reading train data from {path}')
        df = pd.read_csv(path)
        self.X_train = (df[df.columns[1:METRICS_START]]).values.astype(float)
        self.row_labels = df[df.columns[0]]
        self.unique_workloads_train = np.unique(self.row_labels)
        self.y_train = (df[df.columns[METRICS_START:]]).values
        self.metric_names = df.columns[METRICS_START:]
        logger.info(
            f'Shapes after reading: X_train {self.X_train.shape},y_train {self.y_train.shape},unique_workloads {self.unique_workloads_train.shape}'
        )

    def preprocess(self) -> None:
        logger.info('Preprocessing step called in workload mapping')
        logger.info(
            f'Shapes before pruning: X:{self.X_train.shape}, Y: {self.y_train.shape}'
        )

        # Prune metrics to the ones found in Step 1: FA + KMeans
        pruned_metric_idxs = [
            i for i, name in enumerate(self.metric_names)

            if name in list(self.pruned_metrics)
        ]
        self.y_train = self.y_train[:, pruned_metric_idxs]
        self.metric_names = self.metric_names[pruned_metric_idxs]
        logger.info(
            f'Shapes after pruning: X:{self.X_train.shape}, Y: {self.y_train.shape}'
        )
        """
        ## Combine duplicate rows(rows with same knob settings)
        self.X_train, self.y_train, self.row_labels = Util.combine_duplicate_rows(
                                                        self.X_train, self.y_train, self.row_labels)
        logger.info(f'Shapes after dropping duplicate rows: X:{self.X_train.shape}, Y: {self.y_train.shape}')
        """

        # Scale the X & y values, then compute the deciles for each column in y
        # Fit first with training data to transorm the target later
        self.X_scaler.fit(self.X_train)
        self.y_scaler.fit(self.y_train)
        
        self.X_train = self.X_scaler.transform(self.X_train)
        self.y_train = self.y_scaler.transform(self.y_train)
        self.y_binner.fit(self.y_train)
        logger.info(
            f'Shapes after scaling and normalizing: X:{self.X_train.shape}, Y: {self.y_train.shape}'
        )
        logger.info('Preprocessing step done in workload mapping')
        return

    def train_models_helper(self, workload_id) -> None:
        idxs = np.where(self.row_labels == workload_id)
        self.models[workload_id] = dict()
        x_train = self.X_train[idxs]
        results = []
        for im, metric in enumerate(list(self.metric_names)):
            y_train = self.y_train[idxs][:, im]
            #logger.info(f'Shapes of x {x_train.shape}, y {y_train.shape}')
            gpr = GaussianProcessRegressor(
                kernel=self.rbf, alpha=self.noise**2)
            gpr.fit(x_train, y_train)
            results.append((workload_id, metric, gpr))    
        return results        

    def train_models(self) -> None:
        logger.info('Train models step called for the workload')
        self.rbf = ConstantKernel(
            self.output_variation) * RBF(length_scale=self.length_scale)
        with multiprocessing.Pool(self.pool_workers) as pool:
            results = list(
                tqdm.tqdm(
                    pool.imap(self.train_models_helper, self.unique_workloads_train, 
                              chunksize=self.chunk_size), 
                              total=len(self.unique_workloads_train)))
        for l in results:
            for pair in l:
                self.models[pair[0]] = self.models.get(pair[0], dict())
                self.models[pair[0]][pair[1]] = pair[2]
        logger.info('Finished training for unique train workloads')

    def map_target_workload(self, id, target_x, target_y) -> tuple():
        target_y = self.y_binner.transform(target_y) #Needs to be binned for euclidian distance
        scores = []
        for key in self.unique_workloads_train:
            y_pred = np.array([])
            for metric in list(self.metric_names):
                gpr = self.models[key][metric]
                outputs = gpr.predict(target_x)
                y_pred = outputs.reshape(
                    -1, 1) if not y_pred.shape[0] else np.concatenate(
                        [y_pred, outputs.reshape(-1, 1)], axis=1)
            binned_pred = self.y_binner.transform(y_pred)
            dists = np.sqrt(
                np.sum(np.square(np.subtract(binned_pred, target_y)), axis=0))
            scores.append(np.mean(dists))
        scores = np.array(scores)
        self.distances.append(scores)
        min_idx = []

        if self.threshold:
            min_idx = np.where(scores < self.threshold)[0]


        if len(min_idx) == 0:
            min_idx = np.argpartition(scores, self.topk)[:self.topk]


        workload_id = self.unique_workloads_train[min_idx]

        # Now we know which workload from the train set is the nearest neighbor. Need to augment its x and y values to test.
        idxs = np.isin(self.row_labels, workload_id)
        workload_x = self.X_train[idxs]
        workload_y = self.y_train[idxs]

        if self.mode=='test':
            f = open(f'temp/{self.method}_{id}.txt', 'w')
            f.write("["+",".join(workload_id)+"]")
            f.close()
        return (workload_id, workload_x, workload_y)

    def predict_target(self, id, target_x, target_y) -> tuple:
        # Prune target metrics
        pruned_metric_idxs = [
            i for i, name in enumerate(self.metric_names_dev)

            if name in list(self.pruned_metrics)
        ]
        target_y = target_y[:, pruned_metric_idxs]
        metric_names = np.array(self.metric_names_dev)[pruned_metric_idxs]
        latency_idx = np.where(metric_names == LATENCY)
        # scale x and y
        target_x = self.X_scaler.transform(target_x)
        target_y = self.y_scaler.transform(target_y)
        # Take the last config to predict. Rest can be used for workload mapping.
        test_x, test_y = target_x[-1, :].reshape(1,
                                                 -1), target_y[-1, :].reshape(
                                                     1, -1)
        target_x, target_y = target_x[:-1, :], target_y[:-1, :]

        # Map target to closest workload
        neighbor, x, pred_y = self.map_target_workload(id, target_x, target_y)

        # Concatenate target worload and map workload output. Retain original latency values if knob config repeats.
        train_x = np.concatenate([target_x, x], axis=0)
        train_y = np.concatenate([
            target_y[:, latency_idx].reshape(-1, 1),
            pred_y[:, latency_idx].reshape(-1, 1)
        ],
            axis=0)
        logger.debug(
            f'Shapes for training latency prediction{id}: X {train_x.shape}, Y {train_y.shape}'
        )

        temp_data = np.concatenate([train_x, train_y], axis=1)
        df = pd.DataFrame(temp_data)
        df = df.drop_duplicates(
            subset=df.columns[:-1], keep='first').to_numpy()
        train_x, train_y = df[:, :-1], df[:, -1:]

        # Train gpr model to predict latencyalpha
        gpr = GaussianProcessRegressor(kernel=self.rbf, alpha=self.noise**2)
        gpr.fit(train_x, train_y)
        y = gpr.predict(test_x)
        test_y = test_y.reshape(-1)[latency_idx].reshape(-1)
        
        return (id, y, test_y.reshape(-1))

    def read_dev_set(self, path) -> None:
        logger.info(f'Reading dev data from {path}')
        df = pd.read_csv(path)
        self.X_dev = (df[df.columns[1:METRICS_START]]).values.astype(float)
        self.row_labels_dev = df[df.columns[0]]
        self.unique_workloads_dev = np.unique(self.row_labels_dev)
        self.y_dev = (df[df.columns[METRICS_START:]]).values
        self.metric_names_dev = df.columns[METRICS_START:]

    def predict_target_helper(self, workload_id: str) -> None:
        idxs = np.where(self.row_labels_dev == workload_id)
        x_train = self.X_dev[idxs]
        y_train = self.y_dev[idxs]
        ypred, yarr = [], []
        workload = []
        for i in range(x_train.shape[0]):
            _, y_pred, y = self.predict_target(workload_id, x_train, y_train)
            yarr.append(y.reshape(-1)[0])
            ypred.append(y_pred.reshape(-1)[0])
            workload.append(workload_id)
            if self.mode == 'test':
                break
            x_train, y_train = np.roll(
                x_train, -1, axis=0), np.roll(y_train, -1, axis=0)
        ypred = np.array(ypred).reshape(-1,1).astype(float)
        yarr = np.array(yarr).reshape(-1,1).astype(float)
        workload = np.array(workload).reshape(-1,1)
        return np.concatenate([ypred,yarr, workload], axis=-1)

    def inverse_trainsform_helper(self, y):
        y = y.reshape(-1)
        temp = np.zeros((y.shape[0], len(self.pruned_metrics)))
        latency_idx = np.where(self.metric_names_dev == LATENCY)[0]
        temp[:, latency_idx] = y.reshape(-1,1)
        y = self.y_scaler.inverse_transform(temp)
        return y[:,latency_idx]

    def compute_score(self) -> float:
        logger.info('Computes score called.')
        y_arr, ypred_arr = [], []
        with multiprocessing.Pool(self.pool_workers) as pool:
            results = list(
                tqdm.tqdm(
                    pool.imap(self.predict_target_helper, self.unique_workloads_dev, chunksize=self.chunk_size),
                              total=len(self.unique_workloads_dev)))
        results = np.array(results)
        results = np.concatenate(results, axis=0) #y,ypred, workload_ids
        logger.info(f'Shapes final result {results.shape}')
        ypred = results[:, 0].astype(float)
        y = results[:, 1].astype(float)
        mse = np.mean((y-ypred)**2)
        mape = np.mean(np.abs((y -ypred)/y))*100
        np.save(f'{self.mode}_{self.method}_normalised.npy', results) 
        logger.info(f'Normalised MSE: {mse}, mape: {mape}')

        #Inverse transform
        ypred = self.inverse_trainsform_helper(ypred)
        y = self.inverse_trainsform_helper(y)
        mse_u = np.mean((y-ypred)**2)
        mape_u = np.mean(np.abs((y -ypred)/y))*100
        logger.info(f'After inverse transform mse: {mse_u}, mape: {mape_u}')
        results[:,:-1]=np.concatenate([ypred,y], axis=1)
        np.save(f'{self.mode}_{self.method}_transformed.npy', results) 

        #saving predictions
        if self.mode=='test':
            a= np.load(f'{self.mode}_{self.method}_transformed.npy', allow_pickle=True)
            df = pd.read_csv('.data/test.CSV')
            for index, row in df.iterrows(): 
                for entry in a:
                    if row['workload id'] == entry[-1]:
                        df.loc[index, "latency prediction"] = entry[0]
            df.to_csv(f'{self.method}_test.csv', index=False)

        return mape, mape_u, mse, mse_u
