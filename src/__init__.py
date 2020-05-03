# flake8: noqa
# isort:skip_file

from catalyst.dl import registry, SupervisedRunner as Runner

from .callbacks import PredictionCallback
from .experiment import Experiment
from .model import MultiHeadNet
from efficientnet_pytorch import EfficientNet
from catalyst.contrib.models.cv import ResnetEncoder

registry.Model(MultiHeadNet)
registry.Model(EfficientNet.from_pretrained, name='EfficientNet')
registry.Model(ResnetEncoder)

registry.Callback(PredictionCallback)