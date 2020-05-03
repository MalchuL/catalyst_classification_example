import numpy as np
import os
import re

from typing import List


from catalyst.core import Callback, CallbackOrder, State
from catalyst.dl.utils import get_activation_fn



def tryint(s):
    try:
        return int(s)
    except:
        return s


def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [tryint(c) for c in re.split('([0-9]+)', s)]



class PredictionCallback(Callback):
    """
    A callback that tracks metrics through meters and prints metrics for
    each class on `state.on_loader_end`.

    .. note::
        This callback works for both single metric and multi-metric meters.
    """

    def __init__(
            self,
            path_key: str = "filepath",
            probs_key: str = "logits",
            activation: str = "Softmax",
            out_file: str = 'infer_pred.txt'
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
        self.input_key = path_key
        self.output_key = probs_key
        self.activation = activation
        self.activation_fn = get_activation_fn(self.activation)
        self.preds = []
        self.out_file = out_file

    def on_epoch_start(self, state: "State"):
        self.preds = []

    def on_epoch_end(self, state: "State"):
        print('Start infer prediction')
        self.preds.sort(key=lambda x: alphanum_key(x[0]))

        if self.out_file:
            with open(str(state.logdir / self.out_file),'w') as f:
                for name, class_id in self.preds:
                    f.write(f'{name} {class_id}\n')



    def on_batch_end(self, state: State):
        """Batch end hook. Computes batch metrics.

        Args:
            state (State): current state
        """
        logits = state.batch_out[self.output_key].detach().float()
        paths = state.input[self.input_key]
        probabilities = self.activation_fn(logits).detach().cpu().numpy().tolist()

        for name, probs in zip(paths, probabilities):
            filename = os.path.basename(name)
            class_id = np.argmax(probs)
            self.preds.append([filename, class_id])
            print(filename, class_id)


__all__ = ["PredictionCallback"]
