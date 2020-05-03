from typing import List
from collections import defaultdict

import numpy as np

from catalyst.core import Callback, CallbackOrder, State
from catalyst.dl.utils import get_activation_fn


class PredictionCallback(Callback):
    """
    A callback that tracks metrics through meters and prints metrics for
    each class on `state.on_loader_end`.

    .. note::
        This callback works for both single metric and multi-metric meters.
    """

    def __init__(
        self,
        input_key: str = "filepath",
        output_key: str = "logits",
        class_names: List[str] = None,
        activation: str = "Softmax",
    ):
        """
        Args:
            metric_names (List[str]): of metrics to print
                Make sure that they are in the same order that metrics
                are outputted by the meters in `meter_list`
            meter_list (list-like): List of meters.meter.Meter instances
                len(meter_list) == num_classes
            input_key (str): input key to use for metric calculation
                specifies our ``y_true``.
            output_key (str): output key to use for metric calculation;
                specifies our ``y_pred``
            class_names (List[str]): class names to display in the logs.
                If None, defaults to indices for each class, starting from 0.
            num_classes (int): Number of classes; must be > 1
            activation (str): An torch.nn activation applied to the logits.
                Must be one of ['none', 'Sigmoid', 'Softmax2d']
        """
        super().__init__(CallbackOrder.Logging)
        self.input_key = input_key
        self.output_key = output_key
        self.class_names = class_names
        self.activation = activation
        self.activation_fn = get_activation_fn(self.activation)


    def on_batch_end(self, state: State):
        """Batch end hook. Computes batch metrics.

        Args:
            state (State): current state
        """
        logits = state.batch_out[self.output_key].detach().float()
        targets = state.input[self.input_key]
        probabilities = self.activation_fn(logits).detach().cpu().numpy().tolist()

        print(targets, probabilities)




__all__ = ["PredictionCallback"]
