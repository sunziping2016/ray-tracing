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
    texture_id1 = str(uuid4())
    textures[texture_id1] = {
        'name': 'ground1',
        'type': 'solid color',
        'color': '#334c1a'
    }
    texture_id2 = str(uuid4())
    textures[texture_id2] = {
        'name': 'ground2',
        'type': 'solid color',
        'color': '#e6e6e6'
    }
    texture_id = str(uuid4())
    textures[texture_id] = {
        'name': 'ground',
        'type': 'checker',
        'texture1': texture_id1,
        'texture2': texture_id2,
        'density': 10.0,
    }
    material_id = str(uuid4())
    materials[material_id] = {
        'name': 'ground',
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
            center = [a + 0.9 * random(), 0.2, b+ 0.9 * random()]
            if math.sqrt((center[0] - 4.0) ** 2 + (center[1] - 0.2) ** 2 +
                         center[2] ** 2) < 0.9:
                continue
            material_id = str(uuid4())
            choose_mat = random()
            if choose_mat < 0.8:
                materials[material_id] = {
                    'name': f'small ball({a},{b})',
                    'type': 'lambertian',
                    'texture': texture_id
                }
            elif choose_mat < 0.95:
                texture_id = str(uuid4())
                textures[texture_id] = {
                    'name': f'small ball({a},{b})',
                    'type': 'solid color',
                    'color': '#%02x%02x%02x' % (
                        randint(0, 255), randint(0, 255), randint(0, 255))
                }
                materials[material_id] = {
                    'name': f'small ball({a},{b})',
                    'type': 'metal',
                    'albedo': '#%02x%02x%02x' % (
                        randint(128, 255), randint(128, 255),
                        randint(128, 255)),
                    'fuzz': 0.5 * random()
                }
            else:
                materials[material_id] = {
                    'name': f'small ball({a},{b})',
                    'type': 'dielectric',
                    'ir': 1.5
                }
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
    big_balls_id = str(uuid4())
    big_balls: Dict[str, Any] = {}
    material_id = str(uuid4())
    materials[material_id] = {
        'name': 'big ball(1)',
        'type': 'dielectric',
        'ir': 1.5
    }
    big_ball_id = str(uuid4())
    big_balls[big_ball_id] = {
        'name': 'big ball(1)',
        'visible': True,
        'material': material_id,
        'shape': {
            'type': 'sphere',
            'center': [0.0, 1.0, 0.0],
            'radius': 1.0,
        }
    }
    texture_id = str(uuid4())
    textures[texture_id] = {
        'name': 'big ball(2)',
        'type': 'solid color',
        'color': '#66331a'
    }
    material_id = str(uuid4())
    materials[material_id] = {
        'name': 'big ball(2)',
        'type': 'lambertian',
        'texture': texture_id
    }
    big_ball_id = str(uuid4())
    big_balls[big_ball_id] = {
        'name': 'big ball(2)',
        'visible': True,
        'material': material_id,
        'shape': {
            'type': 'sphere',
            'center': [-4.0, 1.0, 0.0],
            'radius': 1.0,
        }
    }
    material_id = str(uuid4())
    materials[material_id] = {
        'name': 'big ball(3)',
        'type': 'metal',
        'albedo': '#b29980',
        'fuzz': 0.0
    }
    big_ball_id = str(uuid4())
    big_balls[big_ball_id] = {
        'name': 'big ball(3)',
        'visible': True,
        'material': material_id,
        'shape': {
            'type': 'sphere',
            'center': [4.0, 1.0, 0.0],
            'radius': 1.0,
        }
    }
    objects.update(big_balls)
    objects[big_balls_id] = {
        'name': 'big balls',
        'visible': True,
        'children': list(big_balls)
    }
    # end
    data['root_objects'] = [ground_id, small_balls_id, big_balls_id]
    with open('scene1.json', 'w') as f:
        json.dump(data, f)


if __name__ == '__main__':
    main()
