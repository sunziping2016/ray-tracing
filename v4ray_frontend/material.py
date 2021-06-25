from abc import ABC, abstractmethod
from typing import List, Any, Dict, Protocol, Set
from uuid import UUID

import v4ray
from v4ray_frontend.properties import AnyProperty, TextureProperty, \
    FloatProperty, ColorProperty
from v4ray_frontend.texture import TextureLike


class MaterialLike(Protocol):
    ...


class MaterialType(ABC):
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
    def apply_preview(data: List[Any],
                      textures: Dict[UUID, TextureLike]) -> MaterialLike:
        pass

    @staticmethod
    @abstractmethod
    def apply(data: List[Any],
              textures: Dict[UUID, TextureLike]) -> MaterialLike:
        pass

    @staticmethod
    @abstractmethod
    def to_json(data: List[Any]) -> Dict[str, Any]:
        pass

    @staticmethod
    @abstractmethod
    def from_json(data: Dict[str, Any]) -> List[Any]:
        pass


class Lambertian(MaterialType):
    @staticmethod
    def kind() -> str:
        return 'lambertian'

    @staticmethod
    def properties() -> List[AnyProperty]:
        return [
            TextureProperty(name='纹理')
        ]

    @staticmethod
    def validate(data: List[Any], valid_textures: Set[UUID]) -> bool:
        return data[0] is not None and data[0] in valid_textures

    @staticmethod
    def apply_preview(data: List[Any],
                      textures: Dict[UUID, TextureLike]) -> MaterialLike:
        return v4ray.material.Lambertian(textures[data[0]])

    @staticmethod
    def apply(data: List[Any],
              textures: Dict[UUID, TextureLike]) -> MaterialLike:
        return v4ray.material.Lambertian(textures[data[0]])

    @staticmethod
    def to_json(data: List[Any]) -> Dict[str, Any]:
        if data[0] is None:
            return {}
        return {
            'texture': str(data[0])
        }

    @staticmethod
    def from_json(data: Dict[str, Any]) -> List[Any]:
        texture = data.get('texture')
        return [
            UUID(texture) if texture is not None else None,
        ]


class Dielectric(MaterialType):
    @staticmethod
    def kind() -> str:
        return 'dielectric'

    @staticmethod
    def properties() -> List[AnyProperty]:
        return [
            FloatProperty(name='折射率', default=1.0)
        ]

    @staticmethod
    def validate(data: List[Any], valid_textures: Set[UUID]) -> bool:
        return float(data[0]) >= 1

    @staticmethod
    def apply_preview(data: List[Any],
                      textures: Dict[UUID, TextureLike]) -> MaterialLike:
        return v4ray.material.Lambertian(
            v4ray.texture.SolidColor((0.9, 0.9, 0.9)))

    @staticmethod
    def apply(data: List[Any],
              textures: Dict[UUID, TextureLike]) -> MaterialLike:
        return v4ray.material.Dielectric(data[0])

    @staticmethod
    def to_json(data: List[Any]) -> Dict[str, Any]:
        return {
            'ir': data[0]
        }

    @staticmethod
    def from_json(data: Dict[str, Any]) -> List[Any]:
        return [
            data['ir']
        ]


class Metal(MaterialType):
    @staticmethod
    def kind() -> str:
        return 'metal'

    @staticmethod
    def properties() -> List[AnyProperty]:
        return [
            ColorProperty(name='反射率'),
            FloatProperty(name='模糊度')
        ]

    @staticmethod
    def validate(data: List[Any], valid_textures: Set[UUID]) -> bool:
        return 0 <= float(data[1]) <= 1

    @staticmethod
    def apply_preview(data: List[Any],
                      textures: Dict[UUID, TextureLike]) -> MaterialLike:
        return v4ray.material.Lambertian(
            v4ray.texture.SolidColor(ColorProperty.map_color(data[0])))

    @staticmethod
    def apply(data: List[Any],
              textures: Dict[UUID, TextureLike]) -> MaterialLike:
        return v4ray.material.Metal(ColorProperty.map_color(data[0]), data[1])

    @staticmethod
    def to_json(data: List[Any]) -> Dict[str, Any]:
        # noinspection PyStringFormat
        return {
            'albedo': '#%02x%02x%02x' % data[0],
            'fuzz': data[1],
        }

    @staticmethod
    def from_json(data: Dict[str, Any]) -> List[Any]:
        return [
            (int(data['albedo'][1:3], 16), int(data['albedo'][3:5], 16),
             int(data['albedo'][5:7], 16)),
            data['fuzz']
        ]
