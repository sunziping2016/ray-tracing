from abc import ABC, abstractmethod
from typing import List, Any, Protocol, Dict
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

    @staticmethod
    @abstractmethod
    def to_json(data: List[Any]) -> Dict[str, Any]:
        pass

    @staticmethod
    @abstractmethod
    def from_json(data: Dict[str, Any]) -> List[Any]:
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

    @staticmethod
    def to_json(data: List[Any]) -> Dict[str, Any]:
        # noinspection PyStringFormat
        return {
            "color": '#%02x%02x%02x' % data[0]
        }

    @staticmethod
    def from_json(data: Dict[str, Any]) -> List[Any]:
        return [
            (int(data['color'][1:3], 16), int(data['color'][3:5], 16),
             int(data['color'][5:7], 16))
        ]
