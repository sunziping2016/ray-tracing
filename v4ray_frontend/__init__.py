from typing import Type, Sequence

from .shape import Sphere, Shape
from .texture import Texture, SolidColor

shapes: Sequence[Type[Shape]] = [Sphere]
textures: Sequence[Type[Texture]] = [SolidColor]
