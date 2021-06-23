from abc import ABC, abstractmethod
from typing import List, Any

from v4ray_frontend.properties import AnyProperty


class Material(ABC):
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
