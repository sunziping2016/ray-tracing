from typing import Tuple

from v4ray_frontend.texture import TextureLike


class SolidColor:
    def __new__(cls, color: Tuple[float, float, float]) -> SolidColor: ...

class Checker:
    def __new__(cls, texture1: TextureLike, texture2: TextureLike,
                density: float) -> Checker: ...
