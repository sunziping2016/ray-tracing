from abc import ABC, abstractmethod
from typing import List, Any

import v4ray
from v4ray_frontend.properties import AnyProperty, FloatProperty


class Shape(ABC):
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
    def apply(data: List[Any]) -> Any:
        pass


class Sphere(Shape):
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
    def apply(data: List[Any]) -> Any:
        x = data[0]
        y = data[1]
        z = data[2]
        radius = data[3]
        assert isinstance(x, float) and isinstance(y, float) and \
               isinstance(z, float) and isinstance(radius, float)
        return v4ray.shape.Sphere((x, y, z), radius)
