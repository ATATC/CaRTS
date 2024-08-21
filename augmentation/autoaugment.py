from abc import ABCMeta as _ABCMeta, abstractmethod as _abstractmethod
from random import choice as _choice

from torch import tensor as _tensor, Tensor as _Tensor
from torchvision.transforms.functional import rotate as _rotate, adjust_brightness as _adjust_brightness, \
    adjust_saturation as _adjust_saturation, adjust_contrast as _adjust_contrast, \
    adjust_sharpness as _adjust_sharpness, posterize as _posterize, solarize as _solarize, \
    autocontrast as _autocontrast, equalize as _equalize, invert as _invert

from augmentation.computational import npndarray as _npndarray, nparray as _nparray
from augmentation.transform import TransformBase, Compose


class AutoAugmentTransform(TransformBase, metaclass=_ABCMeta):
    @_abstractmethod
    def apply_tensor(self, img: _Tensor) -> _Tensor:
        raise NotImplementedError

    def apply(self, img: _npndarray) -> _npndarray:
        return _nparray(self.apply_tensor(_tensor(img.transpose(2, 0, 1)))).transpose(1, 2, 0)


class Rotate(AutoAugmentTransform):
    def __init__(self, angle: float, p: float = 1) -> None:
        super().__init__(p)
        self._angle: float = angle

    def apply_tensor(self, img: _Tensor) -> _Tensor:
        return _rotate(img, self._angle)


class Brightness(AutoAugmentTransform):
    def __init__(self, brightness_factor: float, p: float = 1) -> None:
        super().__init__(p)
        self._brightness_factor: float = brightness_factor

    def apply_tensor(self, img: _Tensor) -> _Tensor:
        return _adjust_brightness(img, 1 + self._brightness_factor)


class Color(AutoAugmentTransform):
    def __init__(self, saturation_factor: float, p: float = 1) -> None:
        super().__init__(p)
        self._saturation_factor: float = saturation_factor

    def apply_tensor(self, img: _Tensor) -> _Tensor:
        return _adjust_saturation(img, 1 + self._saturation_factor)


class Contrast(AutoAugmentTransform):
    def __init__(self, contrast_factor: float, p: float = 1) -> None:
        super().__init__(p)
        self._contrast_factor: float = contrast_factor

    def apply_tensor(self, img: _Tensor) -> _Tensor:
        return _adjust_contrast(img, 1 + self._contrast_factor)


class Sharpness(AutoAugmentTransform):
    def __init__(self, sharpness_factor: float, p: float = 1) -> None:
        super().__init__(p)
        self._sharpness_factor: float = sharpness_factor

    def apply_tensor(self, img: _Tensor) -> _Tensor:
        return _adjust_sharpness(img, 1 + self._sharpness_factor)


class Posterize(AutoAugmentTransform):
    def __init__(self, bits: int, p: float = 1) -> None:
        super().__init__(p)
        self._bits: int = bits

    def apply_tensor(self, img: _Tensor) -> _Tensor:
        return _posterize(img, self._bits)


class Solarize(AutoAugmentTransform):
    def __init__(self, threshold: float, p: float = 1) -> None:
        super().__init__(p)
        self._threshold: float = threshold

    def apply_tensor(self, img: _Tensor) -> _Tensor:
        return _solarize(img, self._threshold)


class AutoContrast(AutoAugmentTransform):
    def __init__(self, p: float = 1) -> None:
        super().__init__(p)

    def apply_tensor(self, img: _Tensor) -> _Tensor:
        return _autocontrast(img)


class Equalize(AutoAugmentTransform):
    def __init__(self, p: float = 1) -> None:
        super().__init__(p)

    def apply_tensor(self, img: _Tensor) -> _Tensor:
        return _equalize(img)


class Invert(AutoAugmentTransform):
    def __init__(self, p: float = 1) -> None:
        super().__init__(p)

    def apply_tensor(self, img: _Tensor) -> _Tensor:
        return _invert(img)


_IMAGE_NET_POLICY: tuple[Compose, ...] = (
    Compose(Posterize(8, p=0.4), Rotate(9, p=0.6)),
    Compose(Solarize(5, p=0.6), AutoContrast(p=0.6)),
    Compose(Equalize(p=0.8), Equalize(p=0.6)),
    Compose(Posterize(7, p=0.6), Posterize(6, p=0.6)),
    Compose(Equalize(p=0.4), Solarize(4, p=0.2)),
    Compose(Equalize(p=0.4), Rotate(8, p=0.8)),
    Compose(Solarize(3, p=0.6), Equalize(p=0.6)),
    Compose(Posterize(5, p=0.8), Equalize()),
    Compose(Rotate(3, p=0.2), Solarize(8, p=0.6)),
    Compose(Equalize(p=0.6), Posterize(6, p=0.4)),
    Compose(Rotate(8, p=0.8), Color(0, p=0.4)),
    Compose(Rotate(9, p=0.4), Equalize(p=0.6)),
    Compose(Equalize(p=0), Equalize(p=0.8)),
    Compose(Invert(p=0.6), Equalize()),
    Compose(Color(4, p=0.6), Contrast(8)),
    Compose(Rotate(8, p=0.8), Color(2)),
    Compose(Color(8, p=0.8), Solarize(7, p=0.8)),
    Compose(Sharpness(7, p=0.4), Invert(p=0.6)),
    Compose(Sharpness(7, p=0.4), Invert(p=0.6)),
    Compose(Color(0, p=0.4), Equalize(p=0.6)),
    Compose(Equalize(p=0.4), Solarize(4, p=0.2)),
    Compose(Solarize(5, p=0.6), AutoContrast(p=0.6)),
    Compose(Invert(p=0.6), Equalize()),
    Compose(Color(4, p=0.6), Contrast(8)),
    Compose(Equalize(p=0.8), Equalize(p=0.6)),
)


class AutoAugment(TransformBase):
    def __init__(self, p: float = 1) -> None:
        super().__init__(p)

    def apply(self, img: _npndarray) -> _npndarray:
        return _choice(_IMAGE_NET_POLICY)(img)
