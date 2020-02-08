from simple_model import SimpleModel
from simple_model import LSTMModel
from simple_model import S2SModel
from simple_model import S2SAttentionModel
from extended_model import RandRegModel
from extended_model import ContinualRandRegModel
from extended_model import NormalModel
from extended_model import ContinualNormalModel
from extended_model import EWCModel
from extended_model import MASModel


def get_model(name, args):
    if name == 'simple':
        return SimpleModel(args)
    elif name == 'lstm':
        return LSTMModel(args)
    elif name == 's2s':
        return S2SModel(args)
    elif name == 's2s_att':
        return S2SAttentionModel(args)
    elif name == 'rand_reg':
        return RandRegModel(args)
    elif name == 'continual_rand_reg':
        return ContinualRandRegModel(args)
    elif name == 'normal':
        return NormalModel(args)
    elif name == 'continual_normal':
        return ContinualNormalModel(args)
    elif name == 'ewc':
        return EWCModel(args)
    elif name == 'mas':
        return MASModel(args)
    else:
        raise ValueError("Model name is not defined: " + name)
