import json
import math
from random import randint, random
from typing import Dict, Any
from uuid import uuid4


def main() -> None:
    data: Dict[str, Any] = {}
    render = data.setdefault('render', {})
    render['width'] = 1200
    render['height'] = 800
    render['max_depth'] = 50
    render['background'] = '#ffffff'
    camera = data.setdefault('camera', {})
    camera['type'] = 'perspective'
    camera['look_from'] = [13.0, 2.0, -3.0]
    camera['look_at'] = [0.0, 0.0, 0.0]
    camera['vfov'] = 20.0
    camera['up'] = [0.0, 1.0, 0.0]
    camera['aperture'] = 0.1
    camera['focus_dist'] = 10.0
    camera['time0'] = 0.0
    camera['time1'] = 0.0
    data.setdefault('')
    objects = data.setdefault('objects', {})
    materials = data.setdefault('materials', {})
    textures = data.setdefault('textures', {})
    # begin
    texture_id = str(uuid4())
    textures[texture_id] = {
        'name': 'ground color',
        'type': 'solid color',
        'color': '#e6e6e6'
    }
    material_id = str(uuid4())
    materials[material_id] = {
        'name': 'ground lamb',
        'type': 'lambertian',
        'texture': texture_id
    }
    ground_id = str(uuid4())
    objects[ground_id] = {
        'name': 'ground',
        'visible': True,
        'material': material_id,
        'shape': {
            'type': 'sphere',
            'center': [0.0, -1000.0, 0.0],
            'radius': 1000.0,
        }
    }
    small_balls_id = str(uuid4())
    small_balls: Dict[str, Any] = {}
    for a in range(-11, 11):
        for b in range(-11, 11):
            texture_id = str(uuid4())
            textures[texture_id] = {
                'name': f'small ball({a},{b}) color',
                'type': 'solid color',
                'color': '#%02x%02x%02x' % (randint(0, 255),
                                            randint(0, 255), randint(0, 255))
            }
            material_id = str(uuid4())
            materials[material_id] = {
                'name': f'small ball({a},{b}) lamb',
                'type': 'lambertian',
                'texture': texture_id
            }
            center = [a + 0.9 * random(), 0.2, b+ 0.9 * random()]
            if math.sqrt((center[0] - 4.0) ** 2 + (center[1] - 0.2) ** 2 +
                         center[2] ** 2) < 0.9:
                continue
            small_ball_id = str(uuid4())
            small_balls[small_ball_id] = {
                'name': f'small ball({a},{b})',
                'visible': True,
                'material': material_id,
                'shape': {
                    'type': 'sphere',
                    'center': center,
                    'radius': 0.2,
                }
            }

    objects.update(small_balls)
    objects[small_balls_id] = {
        'name': 'small balls',
        'visible': True,
        'children': list(small_balls)
    }
    # end
    data['root_objects'] = [ground_id, small_balls_id]
    with open('scene1.json', 'w') as f:
        json.dump(data, f)


if __name__ == '__main__':
    main()
