from typing import Tuple


class SolidColor:
    def __new__(cls, color: Tuple[float, float, float]) -> SolidColor: ...
