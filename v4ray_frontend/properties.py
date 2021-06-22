from dataclasses import dataclass
from typing import Optional, Tuple, List, Any, Union, Callable


@dataclass
class IntProperty:
    name: str
    default: int = 0
    min: Optional[int] = None
    max: Optional[int] = None


@dataclass
class FloatProperty:
    name: str
    default: float = 0.0
    min: Optional[float] = None
    max: Optional[float] = None
    decimals: Optional[int] = None


@dataclass
class BoolProperty:
    name: str
    default: bool = False


@dataclass
class ColorProperty:
    name: str
    default: Tuple[int, int, int] = (255, 255, 255)


@dataclass
class EnumProperty:
    name: str
    options: List[Tuple[str, Any]]
    default: Any
    required: bool = True


AnyProperty = Union[
    IntProperty,
    FloatProperty,
    BoolProperty,
    ColorProperty,
    EnumProperty
]
