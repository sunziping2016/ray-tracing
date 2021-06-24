from abc import ABC, abstractmethod
from typing import List, Any, Protocol, Dict, Tuple
from uuid import UUID

import v4ray
from v4ray_frontend.properties import AnyProperty, ColorProperty


class TextureLike(Protocol):
    ...


class TextureType(ABC):
    @staticmethod
    @abstractmethod
    def kind() -> str:
        pass

    @staticmethod
    @abstractmethod
    def properties() -> List[AnyProperty]:
        pass

    @staticmethod
    @abstractmethod
    def validate(data: List[Any]) -> bool:
        pass

    @staticmethod
    @abstractmethod
    def apply(data: List[Any],
              textures: Dict[UUID, TextureLike]) -> TextureLike:
        pass


class SolidColor(TextureType):
    @staticmethod
    def kind() -> str:
        return 'solid color'

    @staticmethod
    def properties() -> List[AnyProperty]:
        return [
            ColorProperty(name='颜色')
        ]

    @staticmethod
    def validate(data: List[Any]) -> bool:
        return True

    @staticmethod
    def apply(data: List[Any],
              textures: Dict[UUID, TextureLike]) -> TextureLike:
        return v4ray.texture.SolidColor(ColorProperty.map_color(data[0]))
