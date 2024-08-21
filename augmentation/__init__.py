from os import listdir as _listdir
from random import randint as _randint, choice as _choice
from typing import Sequence as _Sequence

from cv2 import imread as _imread
from numpy import ndarray as _ndarray, load as _load
from torch import Tensor as _Tensor, tensor as _tensor

from augmentation.autoaugment import *
from augmentation.transform import *


def select_target(src: str, config: str = "") -> str:
    if not config:
        config = _choice(_listdir(src))
    return f"{src}/{config}/{_choice(_listdir(f'{src}/{config}'))}/regular/left/{_randint(0, 300)}"


def select_targets(src: str, n: int, config: str = "") -> tuple[str, ...]:
    return tuple(select_target(src, config) for _ in range(n))


def load_targets(targets: _Sequence[str]) -> tuple[list[_ndarray], list[_ndarray]]:
    return [_imread(f"{target}.png") for target in targets], [_load(f"{target.replace('regular', 'ground_truth')}.npy")
                                                              for target in targets]


def load_targets_tensor(targets: _Sequence[str]) -> tuple[list[_Tensor], list[_Tensor]]:
    return [_tensor(_imread(f"{target}.png")) for target in targets], [
        _tensor(_load(f"{target.replace('regular', 'ground_truth')}.npy")) for target in targets]
