from typing import Type, List

from .camera import CameraType, PerspectiveCamera
from .material import MaterialType, Lambertian
from .shape import Sphere, ShapeType
from .texture import TextureType, SolidColor

shapes: List[Type[ShapeType]] = [Sphere]
textures: List[Type[TextureType]] = [SolidColor]
materials: List[Type[MaterialType]] = [Lambertian]
cameras: List[Type[CameraType]] = [PerspectiveCamera]
