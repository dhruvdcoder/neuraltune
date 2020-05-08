from typing import Optional, List, Any, Dict
import torch
from allennlp.models import Model
from allennlp.modules.feedforward import FeedForward
from allennlp.nn.activations import Activation
from allennlp.training.metrics import Average
import numpy as np
from joblib import dump, load
import logging
from sklearn.preprocessing import StandardScaler
logger = logging.getLogger(__name__)


def inverse_transform(y, scaler):
    return scaler.inverse_transform(y.detach().numpy()[:, np.newaxis].repeat(
        scaler.scale_.shape[0], axis=1))[:, 0]


def np_mape(y_pred, y, scaler):
    y_pred = inverse_transform(y_pred, scaler)
    y = inverse_transform(y, scaler)
    ratio = y_pred / y
    ref = np.ones_like(ratio)

    return float(np.linalg.norm(ref - ratio, 1)) / len(ref)


class NeuralTune(Model):
    """Base"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)


@Model.register('simple-nn')
class SimpleNN(Model):
    def __init__(self, num_samples: int, representation_network: FeedForward,
                 regression_network: FeedForward, scaler_path: str) -> None:
        super().__init__(vocab=None)
        self.num_samples = num_samples
        self.ff_rep = representation_network
        self.ff_reg = regression_network
        self.output = torch.nn.Linear(self.ff_reg.get_output_dim(), 1)
        self.scaler_path = scaler_path
        self.scaler = None
        self.loss_f = torch.nn.L1Loss()
        self.mape = Average()

    def check_dimensions(self,
                         a: torch.Tensor,
                         b: torch.Tensor,
                         y: torch.Tensor = None) -> None:
        batch, num_samples, num_features = a.shape
        assert num_samples == self.num_samples
        b_batch, b_dims = b.shape  # (batch, knobs)

        assert b_batch == batch

        if y is not None:
            assert y.shape[0] == b_batch

    def forward(self,
                a: torch.Tensor,
                b: torch.Tensor,
                y: Optional[torch.Tensor] = None,
                meta: Dict = None) -> Dict:

        if self.scaler is None:
            self.scaler = load(self.scaler_path)
        # do some checks
        batch, num_samples, num_features = a.shape
        self.check_dimensions(a, b, y)
        a = a.view(batch * num_samples, num_features)
        batch = b.shape[0]
        # apply ff
        hidden_a = self.ff_rep(a)  # (num_samples*batch, hidden_dim)
        hidden_a_grouped = hidden_a.view(
            (batch, self.num_samples, -1))  # (batch, num_samples, hidden_dim)
        hidden_a_mean = torch.mean(hidden_a_grouped, -2)  # (batch, hidden_dim)
        for_regression = torch.cat((hidden_a_mean, b),
                                   -1)  # (batch, hidden_dim + knobs)
        final_states = self.ff_reg(for_regression)  # (batch, hidden_dim)
        pred = self.output(final_states).view(-1)
        output_dict: Dict[str, Optional[torch.Tensor]] = {'pred': pred}

        if y is not None:
            ratio = pred / y
            ref = torch.ones_like(ratio)
            #output_dict['loss'] = self.loss_f(ratio, ref)
            output_dict['loss'] = self.loss_f(pred, y)
            with torch.no_grad():
                output_dict['mape'] = np_mape(pred, y, self.scaler)
            self.mape(output_dict['mape'])

        if meta is not None:
            pred = inverse_transform(pred, self.scaler)
            output_dict['pred'] = []
            output_dict['workload_id'] = []

            for w_id, pred_val in zip(meta, pred.tolist()):
                output_dict['pred'].append(pred_val)
                output_dict['workload_id'].append(w_id['workload_id'])

        return output_dict

    def get_metrics(self, reset: bool = False):
        return {'mape': self.mape.get_metric(reset=reset)}
