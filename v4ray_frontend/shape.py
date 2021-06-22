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
        x: float = data[0]
        y: float = data[1]
        z: float = data[2]
        radius: float = data[3]
        return v4ray.shape.Sphere((x, y, z), radius)
