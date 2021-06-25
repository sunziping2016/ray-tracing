from abc import ABC, abstractmethod
from typing import List, Any, Protocol, Dict, Set
from uuid import UUID

import v4ray
from v4ray_frontend.properties import AnyProperty, ColorProperty, \
    TextureProperty, FloatProperty


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
    def validate(data: List[Any], valid_textures: Set[UUID]) -> bool:
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
    def validate(data: List[Any], valid_textures: Set[UUID]) -> bool:
        return True

    @staticmethod
    def apply(data: List[Any],
              textures: Dict[UUID, TextureLike]) -> TextureLike:
        return v4ray.texture.SolidColor(ColorProperty.map_color(data[0]))

    @staticmethod
    def to_json(data: List[Any]) -> Dict[str, Any]:
        # noinspection PyStringFormat
        return {
            'color': '#%02x%02x%02x' % data[0]
        }

    @staticmethod
    def from_json(data: Dict[str, Any]) -> List[Any]:
        return [
            (int(data['color'][1:3], 16), int(data['color'][3:5], 16),
             int(data['color'][5:7], 16))
        ]


class Checker(TextureType):
    @staticmethod
    def kind() -> str:
        return 'checker'

    @staticmethod
    def properties() -> List[AnyProperty]:
        return [
            TextureProperty(name='纹理1'),
            TextureProperty(name='纹理2'),
            FloatProperty(name='密度', default=1.0)
        ]

    @staticmethod
    def validate(data: List[Any], valid_textures: Set[UUID]) -> bool:
        return data[0] is not None and data[0] in valid_textures and \
               data[1] is not None and data[1] in valid_textures and \
               data[2] > 0.0

    @staticmethod
    def apply(data: List[Any],
              textures: Dict[UUID, TextureLike]) -> TextureLike:
        return v4ray.texture.Checker(
            textures[data[0]], textures[data[1]], data[2])

    @staticmethod
    def to_json(data: List[Any]) -> Dict[str, Any]:
        result = {}
        if data[0] is not None:
            result['texture1'] = str(data[0])
        if data[1] is not None:
            result['texture2'] = str(data[1])
        result['density'] = data[2]
        return result

    @staticmethod
    def from_json(data: Dict[str, Any]) -> List[Any]:
        texture1 = data.get('texture1')
        texture2 = data.get('texture2')
        return [
            UUID(texture1) if texture1 is not None else None,
            UUID(texture2) if texture2 is not None else None,
            data['density'],
        ]
