from dataclasses import dataclass
from typing import Optional, Tuple, Union
from uuid import UUID


@dataclass
class FloatProperty:
    name: str
    default: float = 0.0
    min: Optional[float] = None
    max: Optional[float] = None
    decimals: Optional[int] = None


@dataclass
class ColorProperty:
    name: str
    default: Tuple[int, int, int] = (255, 255, 255)


@dataclass
class TextureProperty:
    name: str
    default: Optional[UUID] = None


AnyProperty = Union[
    FloatProperty,
    ColorProperty,
    TextureProperty,
]
