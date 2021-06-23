from typing import Type, Sequence

from .shape import Sphere, ShapeType
from .texture import TextureType, SolidColor

shapes: Sequence[Type[ShapeType]] = [Sphere]
textures: Sequence[Type[TextureType]] = [SolidColor]
