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

    @staticmethod
    def map_color(color: Tuple[int, int, int]) -> Tuple[float, float, float]:
        return color[0] / 255, color[1] / 255.0, color[2] / 255.0


@dataclass
class TextureProperty:
    name: str
    default: Optional[UUID] = None


AnyProperty = Union[
    FloatProperty,
    ColorProperty,
    TextureProperty,
]
