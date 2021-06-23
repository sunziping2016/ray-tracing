from abc import ABC, abstractmethod
from typing import List, Any, Protocol

import numpy as np

import v4ray
from v4ray_frontend.properties import AnyProperty, FloatProperty


class ShapeLike(Protocol):
    def bounding_box(self) -> v4ray.AABB: ...
    def hit(self, ray: v4ray.Ray, t_min: np.ndarray,
            t_max: np.ndarray) -> v4ray.HitRecord: ...


class ShapeType(ABC):
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
    def apply(data: List[Any]) -> List[ShapeLike]:
        pass


class Sphere(ShapeType):
    @staticmethod
    def kind() -> str:
        return 'sphere'

    @staticmethod
    def properties() -> List[AnyProperty]:
        return [
            FloatProperty('坐标 x'),
            FloatProperty('坐标 y'),
            FloatProperty('坐标 z'),
            FloatProperty('半径'),
        ]

    @staticmethod
    def apply(data: List[Any]) -> List[ShapeLike]:
        x = data[0]
        y = data[1]
        z = data[2]
        radius = data[3]
        assert isinstance(x, float) and isinstance(y, float) and \
               isinstance(z, float) and isinstance(radius, float)
        return [v4ray.shape.Sphere((x, y, z), radius)]
