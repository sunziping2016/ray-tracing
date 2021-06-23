from abc import abstractmethod, ABC
from asyncio import Protocol
from typing import List, Any

import v4ray
from v4ray_frontend.properties import AnyProperty, FloatProperty


class CameraLike(Protocol):
    ...


class CameraType(ABC):
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
    def apply(data: List[Any]) -> CameraLike:
        pass


class PerspectiveCamera(CameraType):
    @staticmethod
    def kind() -> str:
        return 'perspective'

    @staticmethod
    def properties() -> List[AnyProperty]:
        return [
            FloatProperty('坐标 x'),  # 0
            FloatProperty('坐标 y'),  # 1
            FloatProperty('坐标 z'),  # 2
            FloatProperty('看向 x'),  # 3
            FloatProperty('看向 y'),  # 4
            FloatProperty('看向 z'),  # 5
            FloatProperty('垂直视场(deg)', default=20),  # 6
            FloatProperty('上方 x', default=0.0),  # 7
            FloatProperty('上方 y', default=1.0),  # 8
            FloatProperty('上方 z', default=0.0),  # 9
            FloatProperty('光圈', default=0.0),  # 10
            FloatProperty('焦距', default=10.0),  # 11
            FloatProperty('快门时间0', default=0.0),  # 12
            FloatProperty('快门时间1', default=0.0),  # 13
        ]

    @staticmethod
    def validate(data: List[Any]) -> bool:
        return 0 < float(data[6]) < 180 and float(data[10]) >= 0 and \
               float(data[11]) >= 0 and float(data[12]) <= float(data[13])

    @staticmethod
    def apply(data: List[Any]) -> CameraLike:
        return v4ray.PerspectiveCameraParam(
            look_from=(data[0], data[1], data[2]),
            look_at=(data[3], data[4], data[5]),
            vfov=data[6],
            up=(data[7], data[8], data[9]),
            aperture=data[10],
            focus_dist=data[11],
            time0=data[12],
            time1=data[13]
        )
