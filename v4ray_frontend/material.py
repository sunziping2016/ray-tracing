from abc import ABC, abstractmethod
from typing import List, Any, Dict, Protocol, Set
from uuid import UUID

import v4ray
from v4ray_frontend.properties import AnyProperty, TextureProperty
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


class Lambertian(MaterialType):
    @staticmethod
    def kind() -> str:
        return 'lambertian'

    @staticmethod
    def properties() -> List[AnyProperty]:
        return [
            TextureProperty(name='质地')
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
