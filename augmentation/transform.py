from abc import ABCMeta as _ABCMeta, abstractmethod as _abstractmethod
from os.path import abspath as _abspath
from random import randint as _randint

from albumentations import Compose as _Compose, SafeRotate as _SafeRotate, RandomCrop as _RandomCrop, \
    Resize as _Resize, RandomBrightnessContrast as _RandomBrightnessContrast
from cv2 import imread as _imread
from typing_extensions import override as _override, Literal as _Literal

from augmentation.computational import npndarray as _npndarray, zeros as _zeros, linspace as _linspace, full as _full, \
    concatenate as _concatenate, ones as _ones, repeat as _repeat, expand_dims as _expand_dims, \
    array as _array, nparray as _nparray, ndarray as _ndarray, rand as _rand

_ASSETS_PATH: str = f"{_abspath(__file__)[:-12]}assets"
_SMOKE_TEXTURE: _npndarray = _imread(f"{_ASSETS_PATH}/smoke.jpg")


class TransformBase(object, metaclass=_ABCMeta):
    def __init__(self, p: float = 1) -> None:
        self._p: float = p

    @_abstractmethod
    def apply(self, img: _npndarray) -> _npndarray:
        raise NotImplementedError

    def __call__(self, img: _npndarray) -> _npndarray:
        return self.apply(img) if _randint(0, 100) < self._p * 100 else img


class Compose(TransformBase):
    def __init__(self, *transforms: TransformBase) -> None:
        super().__init__()
        self._transforms: tuple[TransformBase, ...] = transforms

    @_override
    def apply(self, img: _npndarray) -> _npndarray:
        for transform in self._transforms:
            img = transform(img)
        return img


class Null(TransformBase):
    def apply(self, img: _npndarray) -> _npndarray:
        return img


class Smoke(TransformBase):
    def __init__(self, attenuation_factor: float = .2, mode: _Literal["linear", "quadratic"] = "linear",
                 smoke_color: tuple[int, int, int] = (200, 200, 200), maximum: float = .7, step: int = 1,
                 p: float = 1) -> None:
        super().__init__(p)
        self._attenuation_factor: float = attenuation_factor
        self._mode: _Literal["linear", "quadratic"] = mode
        self._smoke_color: _ndarray = _array(smoke_color)
        self._maximum: float = maximum
        self._step: int = step

    @_override
    def apply(self, img: _npndarray) -> _npndarray:
        height, width, num_channels = img.shape
        r, h = width * .5, height // 2
        tex = _array(_Compose([
            _RandomCrop(_randint(int(height * .4), int(height * .9)), _randint(int(width * .4), int(width * .9))),
            _SafeRotate(p=1), _Resize(height, width)])(image=_SMOKE_TEXTURE)["image"])
        smoke_mask = _zeros((height, width))
        for i in range(0, h, self._step):
            d = h - i
            total_length = round(2 * (r ** 2 - d ** 2) ** .5)
            total_length += total_length % 2
            num_steps = round(total_length * self._attenuation_factor)
            num_zeros = (width - total_length) // 2
            row = _concatenate((_zeros(num_zeros), _linspace(0, self._maximum, num_steps),
                                _full(total_length - num_steps * 2, self._maximum),
                                _linspace(self._maximum, 0, num_steps), _zeros(num_zeros)))
            if self._mode == "quadratic":
                row **= 2
            smoke_mask[i] = row
        smoke_mask = _repeat(_expand_dims(smoke_mask, -1), 3, 2)
        smoke_mask += smoke_mask[::-1, :, :]
        smoke_mask += smoke_mask * tex / 255
        result = _array(img) * (_ones((height, width, 3)) - smoke_mask) + smoke_mask * self._smoke_color
        return _nparray(result)


class LowBrightness(TransformBase):
    def __init__(self, brightness_range: tuple[float, float] = (-.9, -.1),
                 contrast_range: tuple[float, float] = (-0.2, 0.2), p: float = 1) -> None:
        super().__init__(p)
        self._transform: _RandomBrightnessContrast = _RandomBrightnessContrast(brightness_range, contrast_range, p=1)

    @_override
    def apply(self, img: _npndarray) -> _npndarray:
        return self._transform(image=img)["image"]


class Blood(TransformBase):
    def __init__(self, n: int, root_range: tuple[float, float, float, float] = (0, 0, 1, 1),
                 color: tuple[int, int, int] = (25, 17, 85), opacity: float = .5, infectiousness: float = .4,
                 num_propagation_steps: int = 64, p: float = 1) -> None:
        super().__init__(p)
        self._n: int = n
        self._root_range: tuple[float, float, float, float] = root_range
        self._color: _ndarray = _array(color)
        self._opacity: float = opacity
        self._infectiousness: float = infectiousness
        self._num_propagation_steps: int = num_propagation_steps

    def infect_root(self, mask: _npndarray) -> _npndarray:
        height, width = mask.shape
        root_x = _randint(int(self._root_range[0] * width), int(self._root_range[2] * width))
        root_y = _randint(int(self._root_range[1] * height), int(self._root_range[3] * height))
        mask[root_y, root_x] = True
        return mask

    @_override
    def apply(self, img: _npndarray) -> _npndarray:
        img = _array(img)
        height, width, num_channels = img.shape
        mask = _zeros((height, width), dtype=bool)
        for _ in range(self._n):
            mask = self.infect_root(mask)
        for _ in range(self._num_propagation_steps):
            propagate_up = _rand(height, width) < self._infectiousness
            propagate_down = _rand(height, width) < self._infectiousness
            propagate_left = _rand(height, width) < self._infectiousness
            propagate_right = _rand(height, width) < self._infectiousness
            mask[:-1, :] |= mask[1:, :] & propagate_up[:-1, :]
            mask[1:, :] |= mask[:-1, :] & propagate_down[1:, :]
            mask[:, :-1] |= mask[:, 1:] & propagate_left[:, :-1]
            mask[:, 1:] |= mask[:, :-1] & propagate_right[:, 1:]
        mask_opacity = _array([self._opacity] * 3)
        img_opacity = _array([1 - self._opacity] * 3)
        img[mask == True] = img[mask == True] * img_opacity + self._color * mask_opacity
        return _nparray(img)
