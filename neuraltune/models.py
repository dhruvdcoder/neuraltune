from typing import Optional, List, Any, Dict
import torch
from allennlp.models import Model
from allennlp.modules.feedforward import FeedForward
from allennlp.nn.activations import tanh, Activation
import logging
logger = logging.getLogger(__name__)


class NeuralTune(Model):
    """Base"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)


@NeuralTune.register('simple-nn')
class SimpleNN(NeuralTune):
    def __init__(self,
                 num_samples: int = 5,
                 representation_network: FeedForward = FeedForward(
                     input_dim=10,
                     hidden_size=10,
                     num_layers=1,
                     activations=tanh),
                 regression_network: FeedForward = FeedForward(
                     input_dim=10,
                     hidden_size=10,
                     num_layers=1,
                     activations=tanh)) -> None:
        super().__init__(vocab=None)
        self.num_samples = num_samples
        self.ff_rep = representation_network
        self.ff_reg = regression_network
        self.output = torch.nn.Linear(self.tt_reg.get_output_dim(), 1)
        self.loss_f = torch.nn.MSELoss()

    def check_dimensions(self,
                         a: torch.Tensor,
                         b: torch.Tensor,
                         y: torch.Tensor = None) -> None:
        a_batch, a_dims = a.shape  # (num_samples*batch, knobs+metrics)
        b_batch, b_dims = b.shape  # (batch, knobs)
        assert self.num_samples * b_batch == a_batch

        if y is not None:
            assert y.shape[0] == b_batch

    def forward(self,
                a: torch.Tensor,
                b: torch.Tensor,
                y: Optional[torch.Tensor] = None) -> Dict:
        # do some checks
        self.check_dimensions(a, b, y)
        batch = b.shape[0]
        # apply ff
        hidden_a = self.ff(a)  # (num_samples*batch, hidden_dim)
        hidden_a_grouped = hidden_a.view(
            (batch, self.num_samples, -1))  # (batch, num_samples, hidden_dim)
        hidden_a_mean = torch.mean(hidden_a_grouped, -2)  # (batch, hidden_dim)
        # (batch, hidden_dim + knobs)
        for_regression = torch.cat((hidden_a_mean, b), -1)
        final_states = self.ff_reg(for_regression)  # (batch, hidden_dim)
        pred = self.output(final_states)
        output_dict: Dict[str, Optional[torch.Tensor]] = {'pred': pred}

        if y is not None:
            output_dict['loss'] = self.loss_f(pred, y)
        else:
            output_dict['loss'] = None

        return output_dict
