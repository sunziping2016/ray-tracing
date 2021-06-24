import asyncio
import copy
import json
import os
import pickle
import sys
import typing
from asyncio import AbstractEventLoop
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field
from threading import Thread
from typing import Optional, Union, List, Dict, TypeVar, Any, Tuple, \
    Type, Sequence, Callable, Set
from uuid import UUID, uuid4

import numpy as np
from PyQt5 import QtCore
from PyQt5.QtCore import Qt, QObject, QSize, QTimer, QStandardPaths
from PyQt5.QtGui import QPixmap, QImage, QResizeEvent, QGuiApplication, \
    QDoubleValidator, QColor, QIntValidator
from PyQt5.QtWidgets import QMainWindow, QApplication, QTreeWidgetItem, \
    QLayoutItem, QLabel, QLineEdit, QFormLayout, QWidgetItem, QLayout, \
    QListWidgetItem, QPushButton, QHBoxLayout, QColorDialog, QTabWidget, \
    QComboBox, QFileDialog, QMessageBox, QWidget, QSizePolicy, QAction

import v4ray
import v4ray_frontend
from ui_mainwindow import Ui_MainWindow
from v4ray import RendererParam, Scene
from v4ray_frontend.camera import CameraType, CameraLike
from v4ray_frontend.material import MaterialType, MaterialLike
from v4ray_frontend.texture import TextureType, TextureLike
from v4ray_frontend.properties import AnyProperty, FloatProperty, \
    ColorProperty, TextureProperty
from v4ray_frontend.shape import ShapeType

__version__ = '0.1.0-rc.1'

T = TypeVar('T')


@dataclass
class ObjectData:
    key: UUID = field(default_factory=uuid4, init=False)
    name: str
    shape: Optional[Tuple[str, List[Any]]]
    material: Optional[UUID]
    visible: bool = False


@dataclass
class ObjectListData:
    key: UUID = field(default_factory=uuid4, init=False)
    name: str
    material: Optional[UUID]
    children: List[UUID] = field(default_factory=list)
    visible: bool = False


@dataclass
class TextureData:
    key: UUID = field(default_factory=uuid4, init=False)
    name: str
    texture: Optional[Tuple[str, List[Any]]]


@dataclass
class MaterialData:
    key: UUID = field(default_factory=uuid4, init=False)
    name: str
    material: Optional[Tuple[str, List[Any]]]


@dataclass
class RendererData:
    width: int
    height: int
    max_depth: int
    background: Tuple[int, int, int]


class FormState:
    properties: List[AnyProperty]
    values: List[Any]
    textures: List[Tuple[UUID, str]]

    def on_texture_finished(self, combo: QComboBox, index: int,
                            on_new_state: Callable[[int, Any], None]) -> None:
        text = combo.lineEdit().text()
        try:
            i = [t for _, t in self.textures].index(text)
            on_new_state(index, self.textures[i][0])
        except ValueError:
            on_new_state(index, None)

    def __init__(self, properties: List[AnyProperty],
                 values: Optional[List[Any]],
                 textures: List[Tuple[UUID, str]]):
        self.properties = properties
        self.values = values if values is not None else \
            [p.default for p in properties]
        self.textures = textures

    def data(self) -> List[Any]:
        return self.values

    def apply(
            self,
            on_new_state: Callable[[int, Any], None],
            parent: 'MainWindow'
    ) -> List[Tuple[Any, Any]]:
        # noinspection PyTypeChecker
        widgets: List[Tuple[Any, Any]] = []
        for i, (v, p) in enumerate(zip(self.values, self.properties)):
            label = QLabel(p.name + '：')
            if isinstance(p, FloatProperty):
                line_edit = QLineEdit()
                validator = QDoubleValidator()
                if p.min is not None:
                    validator.setBottom(p.min)
                if p.max is not None:
                    validator.setTop(p.max)
                if p.decimals is not None:
                    validator.setDecimals(p.decimals)
                line_edit.setText(str(v))
                line_edit.setValidator(validator)
                line_edit.editingFinished.connect(
                    lambda i=i, line_edit=line_edit:
                    on_new_state(i, float(line_edit.text())))
                widgets.append((label, line_edit))
            elif isinstance(p, ColorProperty):
                layout = QHBoxLayout()
                button = QPushButton()
                button.setFixedSize(QSize(46, 23))
                color = QColor(v[0], v[1], v[2])
                button.setStyleSheet(f'QPushButton:enabled '
                                     f'{{ background-color: {color.name()}; }}')

                def color_picker(i: int = i, color: QColor = color,
                                 title: str = p.name) -> None:
                    c = QColorDialog.getColor(
                        initial=color, parent=parent, title=title,
                        options=QColorDialog.DontUseNativeDialog)
                    if c.isValid():
                        on_new_state(i, (c.red(), c.green(), c.blue()))

                button.clicked.connect(lambda x: color_picker())
                layout.addWidget(button)
                layout.addStretch(1)
                widgets.append((label, layout))
            elif isinstance(p, TextureProperty):
                layout = QHBoxLayout()
                combo = QComboBox()
                for _, text in self.textures:
                    combo.addItem(text)
                combo.setEditable(True)
                go = QPushButton('Go')
                go.setFixedSize(QSize(46, 23))
                if v is not None:
                    try:
                        index = [t for t, _ in self.textures].index(v)
                        combo.lineEdit().setText(self.textures[index][1])
                        go.clicked.connect(lambda _, v=v:
                                           parent.goto_texture(v))
                        go.setEnabled(True)
                    except ValueError:
                        combo.lineEdit().setText('')
                        go.setEnabled(False)
                else:
                    combo.lineEdit().setText('')
                    go.setEnabled(False)
                combo.lineEdit().editingFinished.connect(
                    lambda combo=combo, go=go, i=i:
                    self.on_texture_finished(combo, i, on_new_state))
                layout.addWidget(combo)
                layout.addWidget(go)
                widgets.append((label, layout))
        return widgets

    def apply_diff(self, _prev: 'FormState',
                   widgets: List[Tuple[QLayoutItem, QLayoutItem]],
                   on_new_state: Callable[[int, Any], None],
                   parent: 'MainWindow') -> None:
        for i, (v, p, (_, f)) in enumerate(zip(
                self.values, self.properties, widgets)):
            if isinstance(p, FloatProperty):
                assert isinstance(f, QWidgetItem)
                line_edit = f.widget()
                assert isinstance(line_edit, QLineEdit)
                line_edit.blockSignals(True)
                line_edit.setText(str(v))
                line_edit.blockSignals(False)
            elif isinstance(p, ColorProperty):
                assert isinstance(f, QHBoxLayout)
                f_item = f.itemAt(0)
                assert isinstance(f_item, QWidgetItem)
                button = f_item.widget()
                color = QColor(v[0], v[1], v[2])
                button.setStyleSheet(f'QPushButton:enabled '
                                     f'{{ background-color: {color.name()}; }}')

                def color_picker(i: int = i, color: QColor = color,
                                 title: str = p.name) -> None:
                    c = QColorDialog.getColor(
                        initial=color, parent=parent, title=title,
                        options=QColorDialog.DontUseNativeDialog)
                    if c.isValid():
                        on_new_state(i, (c.red(), c.green(), c.blue()))

                button.clicked.disconnect()
                button.clicked.connect(lambda x: color_picker())
            elif isinstance(p, TextureProperty):
                assert isinstance(f, QHBoxLayout)
                f_item1, f_item2 = f.itemAt(0), f.itemAt(1)
                assert isinstance(f_item1, QWidgetItem) \
                       and isinstance(f_item2, QWidgetItem)
                combo = f_item1.widget()
                go = f_item2.widget()
                combo.clear()
                for _, text in self.textures:
                    combo.addItem(text)
                combo.lineEdit().editingFinished.disconnect()
                try:
                    go.clicked.disconnect()
                except TypeError:
                    pass
                if v is not None:
                    try:
                        index = [t for t, _ in self.textures].index(v)
                        combo.lineEdit().setText(self.textures[index][1])
                        go.clicked.connect(lambda _, v=v:
                                           parent.goto_texture(v))
                        go.setEnabled(True)
                    except ValueError:
                        combo.lineEdit().setText('')
                        go.setEnabled(False)
                else:
                    combo.lineEdit().setText('')
                    go.setEnabled(False)
                combo.lineEdit().editingFinished.connect(
                    lambda combo=combo, go=go, i=i:
                    self.on_texture_finished(combo, i, on_new_state))


class State:
    render_result: Optional[np.ndarray]

    root_objects: List[UUID]
    objects: Dict[UUID, Union[ObjectData, ObjectListData]]
    current_object: Optional[UUID]
    shape_types: Dict[str, Type[ShapeType]]

    root_textures: List[UUID]
    textures: Dict[UUID, TextureData]
    current_texture: Optional[UUID]
    texture_types: Dict[str, Type[TextureType]]

    root_materials: List[UUID]
    materials: Dict[UUID, MaterialData]
    current_material: Optional[UUID]
    material_types: Dict[str, Type[MaterialType]]

    camera: Optional[Tuple[str, List[Any]]]
    camera_types: Dict[str, Type[CameraType]]

    renderer: RendererData

    # always rebuild
    object_parent: Dict[UUID, Tuple[Optional[UUID], int]]
    object_names: Dict[UUID, str]
    texture_names: Dict[UUID, str]
    material_names: Dict[UUID, str]
    shape_form: Optional[Tuple[str, FormState]]
    texture_form: Optional[Tuple[str, FormState]]
    material_form: Optional[Tuple[str, FormState]]
    camera_form: Optional[Tuple[str, FormState]]
    valid_textures: Set[UUID]
    valid_materials: Set[UUID]
    objects_inherited_materials: Dict[UUID, UUID]
    valid_objects: Set[UUID]
    camera_valid: bool
    visible_objects: Set[UUID]

    rendered_objects: Set[UUID]
    rendered_materials: Set[UUID]
    rendered_textures: Set[UUID]

    def __init__(
            self,
            prev_state: Optional['State'] = None,
    ):
        if prev_state is None:
            self.render_result = None
            self.root_objects = []
            self.objects = {}
            self.current_object = None
            self.shape_types = {}
            self.shape_form = None
            self.root_textures = []
            self.textures = {}
            self.current_texture = None
            self.texture_types = {}
            self.root_materials = []
            self.materials = {}
            self.current_material = None
            self.material_types = {}
            self.camera = None
            self.camera_types = {}
            self.renderer = RendererData(
                width=800, height=600, max_depth=20,
                background=(255, 255, 255))
        else:
            self.render_result = prev_state.render_result
            self.root_objects = prev_state.root_objects
            self.objects = prev_state.objects
            self.current_object = prev_state.current_object
            self.shape_types = prev_state.shape_types
            self.shape_form = prev_state.shape_form
            self.root_textures = prev_state.root_textures
            self.textures = prev_state.textures
            self.current_texture = prev_state.current_texture
            self.texture_types = prev_state.texture_types
            self.root_materials = prev_state.root_materials
            self.materials = prev_state.materials
            self.current_material = prev_state.current_material
            self.material_types = prev_state.material_types
            self.camera = prev_state.camera
            self.camera_types = prev_state.camera_types
            self.renderer = prev_state.renderer
        self.object_parent = {c: (k, i) for k, obj in self.objects.items()
                              if isinstance(obj, ObjectListData)
                              for i, c in enumerate(obj.children)}
        for i, k in enumerate(self.root_objects):
            self.object_parent[k] = None, i
        names = State.calc_unique_name([self.textures[uuid].name
                                        for uuid in self.root_textures])
        self.texture_names = dict(zip(self.root_textures, names))
        names = State.calc_unique_name([self.materials[uuid].name
                                        for uuid in self.root_materials])
        self.material_names = dict(zip(self.root_materials, names))
        names = State.calc_unique_name([o.name for uuid, o
                                        in self.objects.items()])
        self.object_names = dict(zip(self.objects.keys(), names))
        self.shape_form = None
        # noinspection PyTypeChecker
        textures: List[Tuple[UUID, str]] = list(self.texture_names.items())
        if self.current_object is not None:
            obj = self.objects[self.current_object]
            if isinstance(obj, ObjectData) and obj.shape is not None:
                self.shape_form = obj.shape[0], FormState(
                    properties=self.shape_types[obj.shape[0]].properties(),
                    values=obj.shape[1], textures=textures)
        self.texture_form = None
        if self.current_texture is not None:
            texture = self.textures[self.current_texture]
            if texture.texture is not None:
                texture_name = texture.texture[0]
                self.texture_form = texture_name, FormState(
                    properties=self.texture_types[texture_name].properties(),
                    values=texture.texture[1], textures=textures)
        self.material_form = None
        if self.current_material is not None:
            mat = self.materials[self.current_material]
            if mat.material is not None:
                mat_name = mat.material[0]
                self.material_form = mat_name, FormState(
                    properties=self.material_types[mat_name].properties(),
                    values=mat.material[1], textures=textures)
        self.camera_form = None
        if self.camera is not None:
            self.camera_form = self.camera[0], FormState(
                properties=self.camera_types[self.camera[0]].properties(),
                values=self.camera[1], textures=textures)
        self.valid_textures = set()
        for uuid, texture in self.textures.items():
            if texture.name and texture.texture is not None and \
                    self.texture_types[texture.texture[0]].validate(
                        texture.texture[1]):
                self.valid_textures.add(uuid)
        self.valid_materials = set()
        for uuid, mat in self.materials.items():
            if mat.name and mat.material is not None and \
                    self.material_types[mat.material[0]].validate(
                        mat.material[1], self.valid_textures):
                self.valid_materials.add(uuid)
        self.objects_inherited_materials = {}
        self.valid_objects = set()

        def object_traversal(uuids: List[UUID],
                             inherited: Optional[UUID]) -> None:
            for uuid in uuids:
                obj = self.objects[uuid]
                if obj.material is not None and obj.material in self.materials:
                    new_inherited: Optional[UUID] = obj.material
                else:
                    new_inherited = inherited
                if new_inherited:
                    self.objects_inherited_materials[obj.key] = new_inherited
                if isinstance(obj, ObjectListData):
                    object_traversal(obj.children, new_inherited)

        object_traversal(self.root_objects, None)
        for uuid, obj in self.objects.items():
            if isinstance(obj, ObjectData) and \
                    obj.name and obj.shape is not None and \
                    self.shape_types[obj.shape[0]].validate(obj.shape[1]) and \
                    obj.key in self.objects_inherited_materials and \
                    self.objects_inherited_materials[obj.key] \
                    in self.valid_materials:
                self.valid_objects.add(uuid)
        self.camera_valid = False
        if self.camera is not None and self.camera_types[self.camera[0]] \
                .validate(self.camera[1]):
            self.camera_valid = True
        self.visible_objects = set()

        def object_traversal2(uuids: List[UUID]) -> None:
            for uuid in uuids:
                obj = self.objects[uuid]
                if not obj.visible:
                    return
                if isinstance(obj, ObjectListData):
                    object_traversal2(obj.children)
                else:
                    self.visible_objects.add(uuid)

        object_traversal2(self.root_objects)
        self.rendered_objects = self.visible_objects & self.valid_objects
        self.rendered_materials = set()
        for uuid in self.rendered_objects:
            obj = self.objects[uuid]
            assert obj.material is not None
            self.rendered_materials.add(obj.material)
        self.rendered_textures = set()
        for uuid in self.rendered_materials:
            mat = self.materials[uuid]
            assert mat.material is not None
            for i, p in enumerate(
                    self.material_types[mat.material[0]].properties()):
                if isinstance(p, TextureProperty):
                    uuid2 = mat.material[1][i]
                    assert isinstance(uuid2, UUID)
                    self.rendered_textures.add(uuid2)
        stack = list(self.rendered_textures)
        while stack:
            uuid = stack.pop()
            text = self.textures[uuid]
            assert text.texture is not None
            for i, p in enumerate(
                    self.texture_types[text.texture[0]].properties()):
                if isinstance(p, TextureProperty):
                    uuid2 = text.texture[1][i]
                    assert isinstance(uuid2, UUID)
                    if uuid2 not in self.rendered_textures:
                        self.rendered_textures.add(uuid2)
                        stack.append(uuid2)

    def to_json(self, _path: str) -> Dict[str, Any]:
        # noinspection PyDictCreation
        data: Dict[str, Any] = {}
        data['render'] = {
            'width': self.renderer.width,
            'height': self.renderer.height,
            'max_depth': self.renderer.max_depth,
            'background': '#%02x%02x%02x' % self.renderer.background,
        }
        if self.camera is not None:
            # noinspection PyDictCreation
            camera: Dict[str, Any] = {}
            camera['type'] = self.camera[0]
            camera.update(self.camera_types[self.camera[0]]
                          .to_json(self.camera[1]))
            data['camera'] = camera
        data['root_objects'] = [str(o) for o in self.root_objects]
        objects: Dict[str, Any] = {}
        for u, o in self.objects.items():
            # noinspection PyDictCreation
            obj: Dict[str, Any] = {}
            obj['name'] = o.name
            obj['visible'] = o.visible
            if o.material is not None:
                obj['material'] = str(o.material)
            if isinstance(o, ObjectData):
                if o.shape is not None:
                    # noinspection PyDictCreation
                    shape: Dict[str, Any] = {}
                    shape['type'] = o.shape[0]
                    shape.update(
                        self.shape_types[o.shape[0]].to_json(o.shape[1]))
                    obj['shape'] = shape
            else:
                obj['children'] = [str(c) for c in o.children]
            objects[str(u)] = obj
        data['objects'] = objects
        materials: Dict[str, Any] = {}
        for u in self.root_materials:
            m = self.materials[u]
            # noinspection PyDictCreation
            material: Dict[str, Any] = {}
            material['name'] = m.name
            if m.material is not None:
                material['type'] = m.material[0]
                material.update(self.material_types[m.material[0]]
                                .to_json(m.material[1]))
            materials[str(u)] = material
        data['materials'] = materials
        textures: Dict[str, Any] = {}
        for u in self.root_textures:
            text = self.textures[u]
            # noinspection PyDictCreation
            texture: Dict[str, Any] = {}
            texture['name'] = text.name
            if text.texture is not None:
                texture['type'] = text.texture[0]
                texture.update(self.texture_types[text.texture[0]]
                               .to_json(text.texture[1]))
            textures[str(u)] = texture
        data['textures'] = textures
        return data

    def with_from_json(self, data: Dict[str, Any]) -> 'State':
        state: 'State' = copy.deepcopy(self)
        state.render_result = None
        state.root_objects = [UUID(o) for o in data['root_objects']]
        state.current_object = None
        state.objects = {}
        for u, o in data['objects'].items():
            if 'children' in o:
                material = o.get('material')
                state.objects[UUID(u)] = ObjectListData(
                    name=o['name'],
                    material=UUID(material) if material is not None else None,
                    children=[UUID(c) for c in o['children']],
                    visible=o['visible'],
                )
            else:
                material = o.get('material')
                shape = o.get('shape')
                if shape is not None:
                    shape_result: Optional[Tuple[str, List[Any]]] = \
                        shape['type'], \
                        state.shape_types[shape['type']].from_json(shape)
                else:
                    shape_result = None
                state.objects[UUID(u)] = ObjectData(
                    name=o['name'],
                    shape=shape_result,
                    material=UUID(material) if material is not None else None,
                    visible=o['visible'],
                )
            state.objects[UUID(u)].key = UUID(u)
        state.root_textures = [UUID(t) for t in data['textures']]
        state.current_texture = None
        for u, t in data['textures'].items():
            text = t.get('type')
            if text is not None:
                text_result: Optional[Tuple[str, List[Any]]] = \
                    text, state.texture_types[text].from_json(t)
            else:
                text_result = None
            state.textures[UUID(u)] = TextureData(
                name=t['name'], texture=text_result)
            state.textures[UUID(u)].key = UUID(u)
        state.root_materials = [UUID(t) for t in data['materials']]
        state.current_material = None
        for u, m in data['materials'].items():
            mat = m.get('type')
            if mat is not None:
                mat_result: Optional[Tuple[str, List[Any]]] = \
                    mat, state.material_types[mat].from_json(m)
            else:
                mat_result = None
            state.materials[UUID(u)] = MaterialData(
                name=m['name'], material=mat_result)
            state.materials[UUID(u)].key = UUID(u)
        if 'camera' in data:
            camera = data['camera']
            state.camera = \
                camera['type'], \
                state.camera_types[camera['type']].from_json(camera)
        else:
            state.camera = None
        render = data['render']
        background = render['background']
        state.renderer = RendererData(
            width=render['width'],
            height=render['height'],
            max_depth=render['max_depth'],
            background=(int(background[1:3], 16), int(background[3:5], 16),
                        int(background[5:7], 16)),
        )
        return State(state)

    def object_uuid_to_widget(
            self, window: 'MainWindow', uuid: UUID
    ) -> QTreeWidgetItem:
        parents = []
        current: Optional[UUID] = uuid
        while current:
            current, index = self.object_parent[current]
            parents.append(index)
        widget = window.ui.objectTree.topLevelItem(parents.pop())
        while parents:
            widget = widget.child(parents.pop())
        return widget

    def with_more_shapes(self, shapes: Sequence[Type[ShapeType]]) -> 'State':
        state = copy.deepcopy(self)
        for shape in shapes:
            kind = shape.kind()
            assert kind not in self.shape_types
            state.shape_types[kind] = shape
        return State(state)

    def with_render_result(self, image: Optional[np.ndarray]) -> 'State':
        state = copy.deepcopy(self)
        state.render_result = image
        return State(state)

    def with_more_textures(self,
                           textures: Sequence[Type[TextureType]]) -> 'State':
        state = copy.deepcopy(self)
        for texture in textures:
            kind = texture.kind()
            assert kind not in self.texture_types
            state.texture_types[kind] = texture
        return State(state)

    def with_more_materials(self,
                            materials: Sequence[Type[MaterialType]]) -> 'State':
        state = copy.deepcopy(self)
        for material in materials:
            kind = material.kind()
            assert kind not in self.material_types
            state.material_types[kind] = material
        return State(state)

    def with_more_cameras(self, cameras: Sequence[Type[CameraType]]) -> 'State':
        state = copy.deepcopy(self)
        for camera in cameras:
            kind = camera.kind()
            assert kind not in self.camera_types
            state.camera_types[kind] = camera
        return State(state)

    def with_modify_object(
            self, uuid: UUID,
            op: Callable[[Union[ObjectData, ObjectListData]], Any]) -> 'State':
        state = copy.deepcopy(self)
        op(state.objects[uuid])
        return State(state)

    def with_modify_texture(
            self, uuid: UUID, op: Callable[[TextureData], Any]) -> 'State':
        state = copy.deepcopy(self)
        op(state.textures[uuid])
        return State(state)

    def with_modify_material(
            self, uuid: UUID, op: Callable[[MaterialData], Any]) -> 'State':
        state = copy.deepcopy(self)
        op(state.materials[uuid])
        return State(state)

    def with_modify_camera(
            self, op: Callable[[Optional[Tuple[str, List[Any]]]],
                               Optional[Tuple[str, List[Any]]]]) -> 'State':
        state = copy.deepcopy(self)
        state.camera = op(state.camera)
        return State(state)

    def with_modify_renderer(
            self, op: Callable[[RendererData], None]) -> 'State':
        state = copy.deepcopy(self)
        op(state.renderer)
        return State(state)

    def with_current_object(self, uuid: Optional[UUID]) -> 'State':
        state = copy.deepcopy(self)
        state.current_object = uuid
        return State(state)

    def with_current_texture(self, uuid: Optional[UUID]) -> 'State':
        state = copy.deepcopy(self)
        state.current_texture = uuid
        return State(state)

    def with_current_material(self, uuid: Optional[UUID]) -> 'State':
        state = copy.deepcopy(self)
        state.current_material = uuid
        return State(state)

    def with_remove_object(self, uuid: UUID) -> 'State':
        state = copy.deepcopy(self)

        def recursive_remove(uuid: UUID) -> None:
            if state.current_object == uuid:
                state.current_object = None
            obj = state.objects.pop(uuid)
            if isinstance(obj, ObjectListData):
                for child in obj.children:
                    recursive_remove(child)

        parent, index = self.object_parent[uuid]
        if parent is not None and index != 0:
            parent_object = state.objects[parent]
            assert isinstance(parent_object, ObjectListData)
            state.current_object = parent_object.children[index - 1]
        elif parent is None and index != 0:
            state.current_object = self.root_objects[index - 1]
        else:
            state.current_object = parent
        if parent is None:
            state.root_objects.remove(uuid)
        else:
            parent_object = state.objects[parent]
            assert isinstance(parent_object, ObjectListData)
            parent_object.children.remove(uuid)
        recursive_remove(uuid)
        return State(state)

    def with_remove_texture(self, uuid: UUID) -> 'State':
        state = copy.deepcopy(self)
        del state.textures[uuid]
        index = state.root_textures.index(uuid)
        state.root_textures.remove(uuid)
        if state.current_texture == uuid:
            if index == len(state.root_textures):
                index -= 1
            if index >= 0:
                state.current_texture = state.root_textures[index]
            else:
                state.current_texture = None
        return State(state)

    def with_remove_material(self, uuid: UUID) -> 'State':
        state = copy.deepcopy(self)
        del state.materials[uuid]
        index = state.root_materials.index(uuid)
        state.root_materials.remove(uuid)
        if state.current_material == uuid:
            if index == len(state.root_materials):
                index -= 1
            if index >= 0:
                state.current_material = state.root_materials[index]
            else:
                state.current_material = None
        return State(state)

    def with_add_object(self, name: Optional[str] = None,
                        group: bool = True) -> 'State':
        if not self.current_object:
            root: Optional[UUID] = None
            index = len(self.root_objects)
        else:
            obj = self.objects[self.current_object]
            if isinstance(obj, ObjectListData):
                root = self.current_object
                index = len(obj.children)
            else:
                root, index = self.object_parent[self.current_object]
                index += 1
        state = copy.deepcopy(self)
        state.objects = state.objects.copy()
        if group:
            item: Union[ObjectData, ObjectListData] = \
                ObjectListData(name=name or '', material=None,
                               children=[], visible=True)
        else:
            item = ObjectData(name=name or '', shape=None, material=None,
                              visible=True)
        state.objects[item.key] = item
        if root is None:
            state.root_objects.insert(index, item.key)
        else:
            parent = state.objects[root]
            assert isinstance(parent, ObjectListData)
            parent.children.insert(index, item.key)
        state.current_object = item.key
        return State(state)

    def with_add_texture(self, name: Optional[str] = None) -> 'State':
        state = copy.deepcopy(self)
        item = TextureData(name=name or '', texture=None)
        state.textures[item.key] = item
        if state.current_texture is None:
            state.root_textures.append(item.key)
        else:
            state.root_textures.insert(
                state.root_textures.index(state.current_texture) + 1, item.key)
        state.current_texture = item.key
        return State(state)

    def with_add_material(self, name: Optional[str] = None) -> 'State':
        state = copy.deepcopy(self)
        item = MaterialData(name=name or '', material=None)
        state.materials[item.key] = item
        if state.current_material is None:
            state.root_materials.append(item.key)
        else:
            state.root_materials.insert(
                state.root_materials.index(state.current_material) + 1,
                item.key)
        state.current_material = item.key
        return State(state)

    def apply_always(self, window: 'MainWindow') -> None:
        blocks: List[QObject] = [
            window.ui.objectTree, window.ui.objectClearSelection,
            window.ui.objectRemove, window.ui.objectName_,
            window.ui.objectVisible, window.ui.objectMaterial,
            window.ui.objectShape, window.ui.textureType,
            window.ui.objectShape.lineEdit(), window.ui.textureType.lineEdit(),
            window.ui.textureRemove, window.ui.textureList,
            window.ui.textureName, window.ui.materialType,
            window.ui.materialType.lineEdit(), window.ui.materialRemove,
            window.ui.materialList, window.ui.materialName,
            window.ui.objectMaterialGo, window.ui.objectMaterial.lineEdit(),
            window.ui.cameraType, window.ui.cameraType.lineEdit(),
            window.ui.renderWidth, window.ui.renderHeight,
            window.ui.renderMaxDepth, window.ui.renderBackground]
        for o in blocks:
            o.blockSignals(True)
        window.ui.objectClearSelection.setEnabled(bool(self.current_object))
        window.ui.objectRemove.setEnabled(bool(self.current_object))
        window.ui.objectShape.clear()
        for item in self.shape_types:
            window.ui.objectShape.addItem(item)
        window.ui.textureType.clear()
        for item in self.texture_types:
            window.ui.textureType.addItem(item)
        window.ui.materialType.clear()
        for item in self.material_types:
            window.ui.materialType.addItem(item)
        window.ui.objectMaterial.clear()
        for item in self.material_names.values():
            window.ui.objectMaterial.addItem(item)
        window.ui.cameraType.clear()
        for item in self.camera_types:
            window.ui.cameraType.addItem(item)
        if not self.current_object:
            window.ui.objectName_.setEnabled(False)
            window.ui.objectName_.setText('')
            window.ui.objectVisible.setEnabled(False)
            window.ui.objectVisible.setChecked(False)
            window.ui.objectMaterial.setEnabled(False)
            window.ui.objectShape.setEnabled(False)
            window.ui.objectShape.lineEdit().setText('')
            window.ui.objectTree.setCurrentItem(None)  # type: ignore
        else:
            obj = self.objects[self.current_object]
            window.ui.objectName_.setEnabled(True)
            window.ui.objectName_.setText(obj.name)
            window.ui.objectVisible.setEnabled(True)
            window.ui.objectVisible.setChecked(obj.visible)
            window.ui.objectMaterial.setEnabled(True)
            window.ui.objectShape.setEnabled(isinstance(obj, ObjectData))
            window.ui.objectShape.lineEdit().setText(
                '' if isinstance(obj, ObjectListData) or obj.shape is None
                else obj.shape[0])
            window.ui.objectTree.setCurrentItem(
                self.object_uuid_to_widget(window, self.current_object))
        if self.current_object is None or self.objects[self.current_object] \
                .material not in self.materials:
            window.ui.objectMaterial.lineEdit().setText('')
            window.ui.objectMaterialGo.setEnabled(False)
        else:
            window.ui.objectMaterial.setEnabled(True)
            mat_uuid = self.objects[self.current_object].material
            window.ui.objectMaterial.lineEdit().setText(
                self.material_names[mat_uuid] if mat_uuid is not None else '')
            window.ui.objectMaterialGo.setEnabled(True)
        window.ui.textureRemove.setEnabled(bool(self.current_texture))
        if not self.current_texture:
            window.ui.textureType.setEnabled(False)
            window.ui.textureType.lineEdit().setText('')
            window.ui.textureList.setCurrentItem(None)  # type: ignore
            window.ui.textureName.setEnabled(False)
            window.ui.textureName.setText('')
        else:
            text = self.textures[self.current_texture]
            window.ui.textureType.setEnabled(True)
            window.ui.textureType.lineEdit().setText(
                '' if text.texture is None else text.texture[0])
            window.ui.textureList.setCurrentItem(
                window.ui.textureList.item(self.root_textures.index(
                    self.current_texture)))
            window.ui.textureName.setEnabled(True)
            window.ui.textureName.setText(text.name)
        window.ui.materialRemove.setEnabled(bool(self.current_material))
        if not self.current_material:
            window.ui.materialType.setEnabled(False)
            window.ui.materialType.lineEdit().setText('')
            window.ui.materialList.setCurrentItem(None)  # type: ignore
            window.ui.materialName.setEnabled(False)
            window.ui.materialName.setText('')
        else:
            mat = self.materials[self.current_material]
            window.ui.materialType.setEnabled(True)
            window.ui.materialType.lineEdit().setText(
                '' if mat.material is None else mat.material[0])
            window.ui.materialList.setCurrentItem(
                window.ui.materialList.item(self.root_materials.index(
                    self.current_material)))
            window.ui.materialName.setEnabled(True)
            window.ui.materialName.setText(mat.name)
        window.ui.cameraType.lineEdit().setText(
            '' if self.camera is None else self.camera[0])
        window.ui.cameraProperties.setTitle('相机属性 ' +
                                            ('✓' if self.camera_valid else '✗'))
        window.ui.renderWidth.setText(str(self.renderer.width))
        window.ui.renderHeight.setText(str(self.renderer.height))
        window.ui.renderMaxDepth.setText(str(self.renderer.max_depth))
        background_color = QColor(*self.renderer.background)
        window.ui.renderBackground.setStyleSheet(
            f'QPushButton:enabled '
            f'{{ background-color: {background_color.name()}; }}')
        for o in blocks:
            o.blockSignals(False)

    @staticmethod
    def array_to_pixmap(image: np.ndarray) -> QPixmap:
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
        return QPixmap(QImage(
            image.tobytes(), image.shape[1], image.shape[0],
            image.shape[1] * 3, QImage.Format_RGB888))

    def apply_image(self, window: 'MainWindow') -> None:
        if self.render_result is not None:
            pixmap = State.array_to_pixmap(self.render_result)
            window.ui.image.setPixmap(pixmap.scaled(
                window.ui.image.width(),
                window.ui.image.height(),
                Qt.KeepAspectRatio
            ))
        else:
            window.ui.image.clear()

    def apply_object_tree_item_text(
            self,
            item: Union[ObjectData, ObjectListData]) -> str:
        name = self.object_names[item.key]
        if isinstance(item, ObjectData):
            name += ' ' + \
                    ('✓' if item.key in self.valid_objects else '✗')
        else:
            name += ' (组)'
        if item.visible:
            name += ' ' + '👁'
        return name

    def apply_object_tree_item(
            self, item: Union[ObjectData, ObjectListData]) -> QTreeWidgetItem:
        widget = QTreeWidgetItem([self.apply_object_tree_item_text(item)])
        if isinstance(item, ObjectData):
            widget.setData(0, Qt.UserRole, str(item.key))
            return widget
        else:
            for child in item.children:
                widget.addChild(self.apply_object_tree_item(
                    self.objects[child]))
            widget.setData(0, Qt.UserRole, str(item.key))
            if item.children:
                widget.setExpanded(True)
            return widget

    @staticmethod
    def calc_unique_name(items: List[str]) -> List[str]:
        items = [name or '未命名' for name in items]
        repeat = True
        while repeat:
            duplicates = defaultdict(list)
            for i, item in enumerate(items):
                duplicates[item].append(i)
            repeat = False
            for item, indices in duplicates.items():
                if len(indices) > 1:
                    repeat = True
                    for j, index in enumerate(indices):
                        items[index] += '@' + str(j + 1)
        return items

    def apply_texture_item_text(self, item: TextureData) -> str:
        return self.texture_names[item.key] + ' ' + \
               ('✓' if item.key in self.valid_textures else '✗')

    def apply_texture_item(self, item: TextureData) -> QListWidgetItem:
        widget = QListWidgetItem(self.apply_texture_item_text(item))
        widget.setData(Qt.UserRole, str(item.key))
        return widget

    def apply_material_item_text(self, item: MaterialData) -> str:
        return self.material_names[item.key] + ' ' + \
               ('✓' if item.key in self.valid_materials else '✗')

    def apply_material_item(self, item: MaterialData) -> QListWidgetItem:
        widget = QListWidgetItem(self.apply_material_item_text(item))
        widget.setData(Qt.UserRole, str(item.key))
        return widget

    @staticmethod
    def apply_diff_list(
            prev: List[T],
            curr: List[T],
            on_remove: Callable[[int], Any],
            on_add: Callable[[int, T], Any],
            on_update: Callable[[int, T], Any],
            on_nop: Callable[[int, T], Any],
    ) -> None:
        curr_key_set = set(curr)
        prev_key_map = {v: j for j, v in enumerate(prev)}
        prev_keys = prev[:]
        for i, key in reversed(list(enumerate(prev))):
            if key not in curr_key_set:
                on_remove(i)
                del prev_keys[i]
        for i, key in enumerate(curr):
            if key not in prev_key_map:
                on_add(i, key)
                prev_keys.insert(i, key)
            else:
                prev_i = prev_keys.index(key)
                if i != prev_i:
                    on_update(i, key)
                else:
                    on_nop(i, key)

    def apply_diff_object_tree_item(
            self,
            prev_state: 'State',
            key: UUID,
            widget: QTreeWidgetItem
    ) -> Optional[QTreeWidgetItem]:

        def on_update(i: int, child_key: UUID) -> None:
            widget.takeChild(i)
            widget.insertChild(
                i, self.apply_object_tree_item(self.objects[child_key]))

        def on_nop(i: int, child_key: UUID) -> None:
            new_widget = self.apply_diff_object_tree_item(
                prev_state,
                child_key,
                widget.child(i)
            )
            if new_widget is not None:
                widget.takeChild(i)
                widget.insertChild(i, new_widget)

        def on_add(i: int, child_key: UUID) -> None:
            widget.insertChild(
                i, self.apply_object_tree_item(self.objects[child_key]))
            widget.setExpanded(True)

        prev_item = prev_state.objects[key]
        curr_item = self.objects[key]
        if isinstance(prev_item, ObjectData) or \
                isinstance(curr_item, ObjectData):
            return self.apply_object_tree_item(curr_item)
        # prev_item and curr_item is ObjectList
        State.apply_diff_list(
            prev_item.children,
            curr_item.children,
            on_remove=lambda i: widget.takeChild(i),
            on_add=on_add,
            on_update=on_update,
            on_nop=on_nop,
        )
        widget.setText(0, self.apply_object_tree_item_text(curr_item))
        if not prev_item.children:
            widget.setExpanded(True)
        return None

    @staticmethod
    def apply_form(layout: QLayout,
                   start_index: int,
                   current_form: Optional[Tuple[str, FormState]],
                   on_new_state: Callable[[int, Any], None],
                   parent: 'MainWindow') -> None:
        assert isinstance(layout, QFormLayout)
        for i in range(layout.rowCount() - 1, start_index - 1, -1):
            layout.removeRow(i)
        if current_form is not None:
            widgets = current_form[1].apply(on_new_state, parent)
            for label, f in widgets:
                layout.addRow(label, f)

    def apply(self, window: 'MainWindow') -> None:
        blocks: List[QObject] = [
            window.ui.image, window.ui.objectTree,
            window.ui.textureList, window.ui.materialList,
        ]
        for o in blocks:
            o.blockSignals(True)
        # render result
        self.apply_image(window)
        # object tree
        window.ui.objectTree.clear()
        for ob in self.root_objects:
            window.ui.objectTree.addTopLevelItem(
                self.apply_object_tree_item(self.objects[ob]))
        # texture list
        window.ui.textureList.clear()
        for text in self.root_textures:
            window.ui.textureList.addItem(
                self.apply_texture_item(self.textures[text]))
        # material list
        window.ui.materialList.clear()
        for mat in self.root_materials:
            window.ui.materialList.addItem(
                self.apply_material_item(self.materials[mat]))
        # forms
        State.apply_form(
            window.ui.objectProperties.layout(), 4,
            self.shape_form,
            window.shape_form_changed,
            window)
        State.apply_form(
            window.ui.textureProperties.layout(), 2,
            self.texture_form,
            window.texture_form_changed,
            window)
        State.apply_form(
            window.ui.materialProperties.layout(), 2,
            self.material_form,
            window.material_form_changed,
            window)
        State.apply_form(
            window.ui.cameraProperties.layout(), 1,
            self.camera_form,
            window.camera_form_changed,
            window)
        self.apply_always(window)
        if self.camera is not None:
            window.trigger_preview()

    @staticmethod
    def apply_diff_form(layout: QLayout,
                        start_index: int,
                        prev_form: Optional[Tuple[str, FormState]],
                        current_form: Optional[Tuple[str, FormState]],
                        on_new_state: Callable[[int, Any], None],
                        parent: 'MainWindow') -> None:
        assert isinstance(layout, QFormLayout)
        if current_form is None:
            if prev_form is not None:
                for i in range(layout.rowCount() - 1, start_index - 1, -1):
                    layout.removeRow(i)
        elif prev_form is None or current_form[0] != prev_form[0]:
            for i in range(layout.rowCount() - 1, start_index - 1, -1):
                layout.removeRow(i)
            widgets = current_form[1].apply(on_new_state, parent)
            for label, f in widgets:
                layout.addRow(label, f)
        else:
            widgets = [(layout.itemAt(i, QFormLayout.LabelRole),
                        layout.itemAt(i, QFormLayout.FieldRole))
                       for i in range(start_index, layout.rowCount())]
            current_form[1].apply_diff(prev_form[1], widgets,
                                       on_new_state, parent)

    def apply_diff(self, prev_state: 'State', window: 'MainWindow') -> None:

        def on_object_tree_update(i: int, key: UUID) -> None:
            window.ui.objectTree.takeTopLevelItem(i)
            window.ui.objectTree.insertTopLevelItem(
                i, self.apply_object_tree_item(self.objects[key]))

        def on_object_tree_nop(i: int, key: UUID) -> None:
            new_widget = self.apply_diff_object_tree_item(
                prev_state,
                key,
                window.ui.objectTree.topLevelItem(i)
            )
            if new_widget is not None:
                window.ui.objectTree.takeTopLevelItem(i)
                window.ui.objectTree.insertTopLevelItem(i, new_widget)

        def on_texture_list_update(i: int, key: UUID) -> None:
            window.ui.textureList.takeItem(i)
            window.ui.textureList.insertItem(
                i, self.apply_texture_item(self.textures[key]))

        def on_texture_list_nop(i: int, key: UUID) -> None:
            window.ui.textureList.item(i).setText(
                self.apply_texture_item_text(self.textures[key]))

        def on_material_list_update(i: int, key: UUID) -> None:
            window.ui.materialList.takeItem(i)
            window.ui.materialList.insertItem(
                i, self.apply_material_item(self.materials[key]))

        def on_material_list_nop(i: int, key: UUID) -> None:
            window.ui.materialList.item(i).setText(
                self.apply_material_item_text(self.materials[key]))

        blocks: List[QObject] = [
            window.ui.image, window.ui.objectTree,
            window.ui.textureList, window.ui.materialList,
        ]
        for o in blocks:
            o.blockSignals(True)
        # render result
        if id(self.render_result) != id(prev_state.render_result):
            self.apply_image(window)
        # object tree
        State.apply_diff_list(
            prev_state.root_objects,
            self.root_objects,
            on_remove=lambda i: window.ui.objectTree.takeTopLevelItem(i),
            on_add=lambda i, key: window.ui.objectTree.insertTopLevelItem(
                i, self.apply_object_tree_item(self.objects[key])),
            on_update=on_object_tree_update,
            on_nop=on_object_tree_nop,
        )
        State.apply_diff_list(
            prev_state.root_textures,
            self.root_textures,
            on_remove=lambda i: window.ui.textureList.takeItem(i),
            on_add=lambda i, key: window.ui.textureList.insertItem(
                i, self.apply_texture_item(self.textures[key])),
            on_update=on_texture_list_update,
            on_nop=on_texture_list_nop,
        )
        State.apply_diff_list(
            prev_state.root_materials,
            self.root_materials,
            on_remove=lambda i: window.ui.materialList.takeItem(i),
            on_add=lambda i, key: window.ui.materialList.insertItem(
                i, self.apply_material_item(self.materials[key])),
            on_update=on_material_list_update,
            on_nop=on_material_list_nop,
        )
        # forms
        State.apply_diff_form(
            window.ui.objectProperties.layout(), 4,
            prev_state.shape_form, self.shape_form,
            window.shape_form_changed, window)
        State.apply_diff_form(
            window.ui.textureProperties.layout(), 2,
            prev_state.texture_form, self.texture_form,
            window.texture_form_changed, window)
        State.apply_diff_form(
            window.ui.materialProperties.layout(), 2,
            prev_state.material_form, self.material_form,
            window.material_form_changed, window)
        State.apply_diff_form(
            window.ui.cameraProperties.layout(), 1,
            prev_state.camera_form, self.camera_form,
            window.camera_form_changed, window)
        for o in blocks:
            o.blockSignals(False)
        self.apply_always(window)
        if self.camera is not None and self.need_rerender(prev_state):
            window.trigger_preview()

    def need_rerender(self, prev_state: 'State') -> bool:
        if self.camera != prev_state.camera or \
                self.renderer != prev_state.renderer:
            return True
        if self.rendered_objects != prev_state.rendered_objects or \
                self.rendered_materials != prev_state.rendered_materials or \
                self.rendered_textures != prev_state.rendered_textures:
            return True
        for uuid in self.rendered_objects:
            obj1 = self.objects[uuid]
            obj2 = prev_state.objects[uuid]
            assert isinstance(obj1, ObjectData) and isinstance(obj2, ObjectData)
            if obj1.shape != obj2.shape or obj1.material != obj2.material:
                return True
        for uuid in self.rendered_materials:
            mat1 = self.materials[uuid]
            mat2 = prev_state.materials[uuid]
            if mat1.material != mat2.material:
                return True
        for uuid in self.rendered_textures:
            text1 = self.textures[uuid]
            text2 = prev_state.textures[uuid]
            if text1.texture != text2.texture:
                return True
        return False

    def generate(self,
                 preview: bool) -> Tuple[RendererParam, CameraLike, Scene]:
        textures: Dict[UUID, TextureLike] = {}

        def generate_texture(uuid: UUID) -> None:
            if uuid in textures:
                return
            text = self.textures[uuid]
            assert text.texture is not None
            texture_type = self.texture_types[text.texture[0]]
            for i, p in enumerate(texture_type.properties()):
                if isinstance(p, TextureProperty):
                    generate_texture(text.texture[1][i])
            textures[uuid] = texture_type.apply(text.texture[1], textures)

        for uuid in self.rendered_textures:
            generate_texture(uuid)
        materials: Dict[UUID, MaterialLike] = {}
        for uuid in self.rendered_materials:
            mat = self.materials[uuid]
            assert mat.material is not None
            mat_type = self.material_types[mat.material[0]]
            if preview:
                materials[uuid] = mat_type.apply_preview(mat.material[1],
                                                         textures)
            else:
                materials[uuid] = mat_type.apply(mat.material[1], textures)
        scene = Scene(
            background=ColorProperty.map_color(self.renderer.background),
            environment=(1.0, 1.0, 1.0) if preview else (0.0, 0.0, 0.0))
        for uuid in self.rendered_objects:
            obj = self.objects[uuid]
            assert isinstance(obj, ObjectData)
            assert obj.shape is not None and obj.material is not None
            for s in self.shape_types[obj.shape[0]].apply(obj.shape[1]):
                scene.add(s, materials[obj.material])
        assert self.camera is not None
        camera = self.camera_types[self.camera[0]].apply(self.camera[1])
        renderer = RendererParam(self.renderer.width, self.renderer.height,
                                 1 if preview else self.renderer.max_depth,
                                 not preview)
        return renderer, camera, scene

    def to_pickle(self) -> Dict[str, Any]:
        return {
            'root_objects': self.root_objects,
            'objects': self.objects,
            'shape_types': self.shape_types,
            'root_textures': self.root_textures,
            'textures': self.textures,
            'texture_types': self.texture_types,
            'root_materials': self.root_materials,
            'materials': self.materials,
            'material_types': self.material_types,
            'camera': self.camera,
            'camera_types': self.camera_types,
            'renderer': self.renderer,
        }

    @staticmethod
    def from_pickle(data: Dict[str, Any]) -> 'State':
        state = State()
        state.root_objects = data['root_objects']
        state.objects = data['objects']
        state.shape_types = data['shape_types']
        state.root_textures = data['root_textures']
        state.textures = data['textures']
        state.texture_types = data['texture_types']
        state.root_materials = data['root_materials']
        state.materials = data['materials']
        state.material_types = data['material_types']
        state.camera = data['camera']
        state.camera_types = data['camera_types']
        state.renderer = data['renderer']
        return State(state)


@dataclass
class HistoryItem:
    state: 'State'
    name: str
    parent: int
    child: int

    def to_pickle(self) -> List[Any]:
        return [self.state.to_pickle(), self.name, self.parent, self.child]

    @staticmethod
    def from_pickle(data: List[Any]) -> 'HistoryItem':
        return HistoryItem(
            state=State.from_pickle(data[0]),
            name=data[1], parent=data[2], child=data[3])


class MainWindow(QMainWindow):
    render_result = QtCore.pyqtSignal(np.ndarray)
    ui: Ui_MainWindow
    loop: AbstractEventLoop

    state: State
    filename: Optional[str]
    history: typing.OrderedDict[int, HistoryItem]
    current_history: int

    def __init__(self, loop: AbstractEventLoop) -> None:
        super(MainWindow, self).__init__()
        self.loop = loop
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        # state
        folder = QStandardPaths.writableLocation(QStandardPaths.AppDataLocation)
        os.makedirs(folder, exist_ok=True)
        self.workspace_path = os.path.join(folder, 'workspace.pkl')
        self.load_workspace()
        # undo redo

        def move_history_parent() -> None:
            parent = self.history[self.current_history].parent
            if parent >= 0:
                self.history[parent].child = self.current_history
                self.move_history(parent)
        self.ui.undo.triggered.connect(lambda _: move_history_parent())
        self.addAction(self.ui.undo)
        self.ui.redo.triggered.connect(
            lambda x: self.move_history(
                self.history[self.current_history].child))
        self.addAction(self.ui.redo)
        clear_unreachable = QAction('删除不可达记录', self)
        clear_unreachable.triggered.connect(
            lambda x: self.clear_unreachable_history())
        self.ui.history.addAction(clear_unreachable)
        clear_other = QAction('删除其他记录', self)
        clear_other.triggered.connect(lambda x: self.clear_other())
        self.ui.history.addAction(clear_other)
        # signals
        old_handle_resize_event = self.ui.image.resizeEvent

        def handle_resize_event(event: QResizeEvent) -> None:
            old_handle_resize_event(event)
            pixmap = self.ui.image.pixmap()
            if pixmap is not None:
                self.ui.image.setPixmap(pixmap.scaled(
                    self.ui.image.width(),
                    self.ui.image.height(),
                    Qt.KeepAspectRatio
                ))

        self.ui.renderWidth.setValidator(QIntValidator(1, 3840))
        self.ui.renderHeight.setValidator(QIntValidator(1, 2160))
        self.ui.renderMaxDepth.setValidator(QIntValidator(1, 1000))
        self.ui.image.resizeEvent = handle_resize_event  # type: ignore
        self.ui.objectAddGroup.clicked.connect(
            lambda _: self.object_add(True))
        self.ui.objectAdd.clicked.connect(
            lambda _: self.object_add(False))
        self.ui.objectTree.currentItemChanged.connect(
            lambda x, _: self.object_current_changed(x))
        self.ui.objectClearSelection.clicked.connect(
            lambda _: self.object_current_changed(None))
        self.ui.objectRemove.clicked.connect(lambda _: self.object_remove())
        self.ui.objectName_.editingFinished.connect(self.object_name_changed)
        self.ui.objectVisible.stateChanged.connect(
            lambda state: self.object_visible_changed(state == Qt.Checked))
        self.ui.objectShape.lineEdit().editingFinished.connect(
            self.object_shape_changed)
        self.ui.objectMaterial.lineEdit().editingFinished.connect(
            self.object_material_change)
        self.ui.objectMaterialGo.clicked.connect(
            lambda _: self.goto_current_material())
        self.ui.textureList.currentItemChanged.connect(
            lambda x, _: self.texture_current_changed(x))
        self.ui.textureAdd.clicked.connect(lambda _: self.texture_add())
        self.ui.textureRemove.clicked.connect(lambda _: self.texture_remove())
        self.ui.textureType.lineEdit().editingFinished.connect(
            self.texture_type_changed)
        self.ui.textureName.editingFinished.connect(self.texture_name_changed)
        self.ui.materialList.currentItemChanged.connect(
            lambda x, _: self.material_current_changed(x))
        self.ui.materialAdd.clicked.connect(lambda _: self.material_add())
        self.ui.materialRemove.clicked.connect(lambda _: self.material_remove())
        self.ui.materialType.lineEdit().editingFinished.connect(
            self.material_type_changed)
        self.ui.materialName.editingFinished.connect(self.material_name_changed)
        self.ui.cameraType.lineEdit().editingFinished.connect(
            self.camera_type_changed)
        self.ui.renderWidth.editingFinished.connect(self.renderer_width_changed)
        self.ui.renderHeight.editingFinished.connect(
            self.renderer_height_changed)
        self.ui.renderMaxDepth.editingFinished.connect(
            self.renderer_max_depth_changed)
        self.ui.renderBackground.clicked.connect(
            lambda _: self.render_background_set())
        self.render_result.connect(self.render_result_available)
        self.ui.about.triggered.connect(lambda x: self.about())
        self.ui.open.triggered.connect(lambda x: self.open())
        self.ui.save.triggered.connect(lambda x: self.save())
        self.ui.saveAs.triggered.connect(lambda x: self.save_as())
        for dock in [self.ui.dockScene, self.ui.dockTexture,
                     self.ui.dockMaterial, self.ui.dockCamera,
                     self.ui.dockOperation]:
            self.ui.viewMenu.addAction(dock.toggleViewAction())
        self.ui.history.currentRowChanged.connect(
            lambda i: self.move_history(list(self.history.keys())[i]))
        # resize
        self.setTabPosition(Qt.AllDockWidgetAreas, QTabWidget.North)
        # self.tabifyDockWidget(self.ui.dockScene, self.ui.dockMaterial)
        # self.tabifyDockWidget(self.ui.dockScene, self.ui.dockTexture)
        self.ui.dockScene.raise_()
        size = QGuiApplication.primaryScreen().size()
        self.resize(QSize(int(0.8 * size.width()), int(0.8 * size.height())))

    def load_workspace(self) -> None:
        try:
            with open(self.workspace_path, 'rb') as f:
                data = pickle.load(f)
            self.filename = data['filename']
            self.history = OrderedDict([(k, HistoryItem.from_pickle(v))
                                        for k, v in data['history'].items()])
            self.current_history = data['current_history']
            self.ui.history.blockSignals(True)
            self.ui.history.clear()
            for index in self.history.keys():
                item, widget = self.history_widget(index)
                self.ui.history.addItem(item)
                self.ui.history.setItemWidget(item, widget)
            self.ui.history.setCurrentRow(
                list(self.history.keys()).index(self.current_history))
            self.ui.history.blockSignals(False)
            self.state = self.history[self.current_history].state
            self.state.apply(self)
        except IOError:
            self.filename = None
            self.history = OrderedDict()
            self.current_history = -1
            self.state = State() \
                .with_more_shapes(v4ray_frontend.shapes) \
                .with_more_textures(v4ray_frontend.textures) \
                .with_more_materials(v4ray_frontend.materials) \
                .with_more_cameras(v4ray_frontend.cameras)
            self.set_state(state=self.state, action='初始状态')

    def save_workspace(self) -> None:
        with open(self.workspace_path, 'wb') as f:
            data = {
                'history': OrderedDict([(k, v.to_pickle())
                                        for k, v in self.history.items()]),
                'current_history': self.current_history,
                'filename': self.filename,
            }
            pickle.dump(data, f)

    def history_widget(self, index: int) -> Tuple[QListWidgetItem, QWidget]:
        widget = QWidget()
        layout = QHBoxLayout()
        history = self.history[index]
        old_index = history.parent
        name = history.name
        if old_index >= 0:
            label = QLabel(f'#{index + 1} ← #{old_index + 1}: {name}')
        else:
            label = QLabel(f'#{index + 1}: {name}')
        label.setSizePolicy(QSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed))
        layout.addWidget(label)
        remove = QPushButton('×')
        remove.setFixedSize(QSize(23, 23))
        remove.setStyleSheet('QPushButton:enabled { color: red; }')

        def on_remove() -> None:
            self.ui.history.blockSignals(True)
            self.ui.history.takeItem(list(self.history.keys()).index(index))
            data = self.history.pop(index)
            for i, (key, value) in enumerate(self.history.items()):
                if value.parent == index:
                    value.parent = data.parent
                    label2 = self.ui.history.itemWidget(
                        self.ui.history.item(i)).layout().itemAt(0).widget()
                    label2.setText(f'#{key + 1} ← #{value.parent + 1}: '
                                   f'{value.name}')
                if value.child == index:
                    value.child = data.child
            self.ui.history.blockSignals(False)
            if self.current_history == index:
                self.move_history(data.parent)
            self.save_workspace()
        remove.setEnabled(index != 0)
        remove.clicked.connect(lambda _: on_remove())
        layout.addWidget(remove)
        layout.setContentsMargins(2, 2, 2, 2)
        widget.setLayout(layout)
        item = QListWidgetItem()
        item.setSizeHint(widget.sizeHint())
        return item, widget

    def move_history(self, index: int) -> None:
        if index < 0:
            return
        self.current_history = index
        self.ui.history.blockSignals(True)
        self.ui.history.setCurrentRow(list(self.history.keys()).index(index))
        self.ui.history.blockSignals(False)
        self.set_state(self.history[index].state)

    def insert_history(self, state: State, name: str) -> None:
        index = 0 if not self.history else list(self.history.keys())[-1] + 1
        self.history[index] = HistoryItem(state, name, self.current_history, -1)
        old_index = self.current_history
        if old_index >= 0:
            self.history[old_index].child = index
        self.current_history = index
        self.ui.history.blockSignals(True)
        item, widget = self.history_widget(index)
        self.ui.history.addItem(item)
        self.ui.history.setItemWidget(item, widget)
        self.ui.history.setCurrentRow(len(self.history) - 1)
        self.ui.history.blockSignals(False)

    def clear_unreachable_history(self) -> None:
        assert self.current_history >= 0
        if QMessageBox.question(
                self, '确认', '删除不可达历史记录是不可以恢复的，确认继续么？'
        ) != QMessageBox.Yes:
            return
        reachable = {self.current_history}
        history = self.current_history
        while history >= 0:
            history = self.history[history].parent
            reachable.add(history)
        history = self.current_history
        while history >= 0:
            history = self.history[history].child
            reachable.add(history)
        self.ui.history.blockSignals(True)
        for row, index in reversed(list(enumerate(self.history.keys()))):
            if index not in reachable:
                self.ui.history.takeItem(row)
        self.history = OrderedDict([(k, v) for k, v in self.history.items()
                                    if k in reachable])
        self.ui.history.setCurrentRow(list(self.history.keys())
                                      .index(self.current_history))
        self.ui.history.blockSignals(False)
        self.save_workspace()

    def clear_other(self) -> None:
        if QMessageBox.question(
                self, '确认', '删除其他历史记录是不可以恢复的，确认继续么？'
        ) != QMessageBox.Yes:
            return
        assert self.current_history >= 0
        self.ui.history.blockSignals(True)
        for row, index in reversed(list(enumerate(self.history.keys()))):
            if index != self.current_history and index != 0:
                self.ui.history.takeItem(row)
        history = self.history[self.current_history]
        history.child = -1
        if self.current_history > 0:
            history.parent = 0
            self.current_history = 1
            self.history = OrderedDict([(0, self.history[0]), (1, history)])
            label = self.ui.history.itemWidget(self.ui.history.item(1)) \
                .layout().itemAt(0).widget()
            label.setText(f'#2 ← #1: {history.name}')
            self.ui.history.setCurrentRow(1)
        else:
            history.parent = -1
            self.history = OrderedDict([(0, self.history[0])])
            self.ui.history.setCurrentRow(0)
        self.ui.history.blockSignals(False)
        self.save_workspace()

    def update_window_title(self) -> None:
        if self.filename is not None:
            self.setWindowTitle('渲染大作业 By Sun - ' + self.filename)
        else:
            self.setWindowTitle('渲染大作业 By Sun')

    def about(self) -> None:
        QMessageBox.about(self, self.windowTitle(),
                          f'软件版本：v{__version__}\n开发者：me@szp.io')

    def open(self) -> None:
        filename = QFileDialog.getOpenFileName(
            self, caption='保存项目', filter="v4ray 工程文件 (*.json)",
            options=QFileDialog.DontUseNativeDialog)[0]
        if filename:
            self.filename = filename
            self.update_window_title()
            with open(self.filename) as f:
                self.set_state(
                    self.state.with_from_json(json.load(f)),
                    f'打开 {os.path.basename(filename)}')

    def save(self) -> None:
        if self.filename is not None:
            with open(self.filename, 'w') as f:
                json.dump(self.state.to_json(self.filename), f)
        else:
            self.save_as()

    def save_as(self) -> None:
        filename = QFileDialog.getSaveFileName(
            self, caption='保存项目', filter="v4ray 工程文件 (*.json)",
            options=QFileDialog.DontUseNativeDialog)[0]
        if filename is not None:
            self.filename = filename
            self.update_window_title()
            with open(self.filename, 'w') as f:
                json.dump(self.state.to_json(self.filename), f)

    def render_background_set(self) -> None:
        initial = QColor(*self.state.renderer.background)
        color = QColorDialog.getColor(
            initial=initial, parent=self, title='渲染背景色',
            options=QColorDialog.DontUseNativeDialog)

        def modify(renderer: RendererData) -> None:
            renderer.background = color.red(), color.green(), color.blue()

        if color.isValid() and color != initial:
            self.set_state(
                self.state.with_modify_renderer(modify),
                f'设置渲染背景色为 {color.name()}')

    @QtCore.pyqtSlot(np.ndarray)
    def render_result_available(self, data: np.ndarray) -> None:
        self.set_state(self.state.with_render_result(data))

    def trigger_preview(self) -> None:
        def trigger() -> None:
            param, camera, scene = self.state.generate(True)
            renderer = v4ray.Renderer(param, camera, scene)
            asyncio.run_coroutine_threadsafe(
                render(renderer, self.render_result),
                self.loop)

        QTimer.singleShot(0, trigger)

    def renderer_width_changed(self) -> None:
        width = int(self.ui.renderWidth.text())
        def modify(renderer: RendererData) -> None:
            renderer.width = width
        if self.state.renderer.width != width:
            self.set_state(self.state.with_modify_renderer(modify),
                           f'设置图像宽度为 {self.ui.renderWidth.text()}')

    def renderer_height_changed(self) -> None:
        height = int(self.ui.renderHeight.text())
        def modify(renderer: RendererData) -> None:
            renderer.height = height
        if self.state.renderer.height != height:
            self.set_state(self.state.with_modify_renderer(modify),
                           f'设置图像高度为 {self.ui.renderHeight.text()}')

    def renderer_max_depth_changed(self) -> None:
        max_depth = int(self.ui.renderMaxDepth.text())
        def modify(renderer: RendererData) -> None:
            renderer.max_depth = max_depth
        if self.state.renderer.max_depth != max_depth:
            self.set_state(self.state.with_modify_renderer(modify),
                           f'设置渲染最大深度为 {self.ui.renderMaxDepth.text()}')

    def goto_texture(self, uuid: UUID) -> None:
        self.ui.dockTexture.raise_()
        self.set_state(self.state.with_current_texture(uuid))

    def goto_current_material(self) -> None:
        assert self.state.current_object
        material = self.state.objects[self.state.current_object].material
        assert material is not None
        self.ui.dockMaterial.raise_()
        self.set_state(self.state.with_current_material(material))

    def shape_form_changed(self, index: int, data: Any) -> None:
        def modify(obj: Union[ObjectData, ObjectListData]) -> None:
            assert isinstance(obj, ObjectData) and obj.shape is not None
            shape_data = obj.shape[1]
            shape_data[index] = data
            obj.shape = obj.shape[0], shape_data
        assert self.state.current_object
        name = self.state.object_names[self.state.current_object]
        obj = self.state.objects[self.state.current_object]
        assert isinstance(obj, ObjectData) and obj.shape is not None
        field_name = self.state.shape_types[obj.shape[0]].properties()[index]
        if obj.shape[1][index] != data:
            self.set_state(
                self.state.with_modify_object(
                    self.state.current_object, modify),
                f'修改对象 {name} 的形状 {field_name}')

    def texture_form_changed(self, index: int, data: Any) -> None:
        def modify(text: TextureData) -> None:
            assert text.texture is not None
            texture_data = text.texture[1]
            texture_data[index] = data
            text.texture = text.texture[0], texture_data
        assert self.state.current_texture
        name = self.state.texture_names[self.state.current_texture]
        t = self.state.textures[self.state.current_texture]
        assert t.texture is not None
        field_name = self.state.texture_types[t.texture[0]].properties()[index]
        if t.texture[1][index] != data:
            self.set_state(
                self.state.with_modify_texture(
                    self.state.current_texture, modify),
                f'修改质地 {name} 的 {field_name}')

    def material_form_changed(self, index: int, data: Any) -> None:
        def modify(mat: MaterialData) -> None:
            assert mat.material is not None
            material_data = mat.material[1]
            material_data[index] = data
            mat.material = mat.material[0], material_data
        assert self.state.current_material
        name = self.state.material_names[self.state.current_material]
        m = self.state.materials[self.state.current_material]
        assert m.material is not None
        field_name = self.state.material_types[m.material[0]] \
            .properties()[index]
        if m.material[1][index] != data:
            self.set_state(
                self.state.with_modify_material(
                    self.state.current_material, modify),
                f'修改材料 {name} 的 {field_name}')

    def camera_form_changed(self, index: int, data: Any) -> None:
        def modify(
                camera: Optional[Tuple[str, List[Any]]]
        ) -> Optional[Tuple[str, List[Any]]]:
            assert camera is not None
            camera[1][index] = data
            return camera
        camera = self.state.camera
        assert camera is not None
        field_name = self.state.camera_types[camera[0]].properties()[index]
        if camera[1][index] != data:
            self.set_state(
                self.state.with_modify_camera(modify),
                f'修改相机的 {field_name}')

    def object_add(self, group: bool) -> None:
        self.set_state(
            self.state.with_add_object(group=group),
            '添加对象组' if group else '添加对象')

    def texture_add(self) -> None:
        self.set_state(self.state.with_add_texture(), '添加质地')

    def material_add(self) -> None:
        self.set_state(self.state.with_add_material(), '添加材料')

    def object_remove(self) -> None:
        widget = self.ui.objectTree.currentItem()
        assert widget
        uuid = UUID(widget.data(0, Qt.UserRole))
        name = self.state.object_names[uuid]
        self.set_state(
            self.state.with_remove_object(uuid),
            f'删除对象(组) {name}')

    def texture_remove(self) -> None:
        widget = self.ui.textureList.currentItem()
        assert widget
        uuid = UUID(widget.data(Qt.UserRole))
        name = self.state.texture_names[uuid]
        self.set_state(
            self.state.with_remove_texture(uuid),
            f'删除质地 {name}')

    def material_remove(self) -> None:
        widget = self.ui.materialList.currentItem()
        assert widget
        uuid = UUID(widget.data(Qt.UserRole))
        name = self.state.material_names[uuid]
        self.set_state(
            self.state.with_remove_material(uuid),
            f'删除材料 {name}')

    def object_shape_changed(self) -> None:
        text = self.ui.objectShape.lineEdit().text()
        assert self.state.current_object
        shape = None if not text or text not in self.state.shape_types else text
        obj = self.state.objects[self.state.current_object]
        assert isinstance(obj, ObjectData)
        current_shape = '' if obj.shape is None else obj.shape[0]
        if current_shape == text:
            return

        def modify(obj: Union[ObjectData, ObjectListData]) -> None:
            assert isinstance(obj, ObjectData)
            obj.shape = None if shape is None else \
                (shape, [p.default for p in
                         self.state.shape_types[shape].properties()])

        def update_state() -> None:
            assert self.state.current_object
            name = self.state.object_names[self.state.current_object]
            self.set_state(self.state.with_modify_object(
                self.state.current_object, modify), f'修改对象 {name} 的形状类型')

        QTimer.singleShot(0, update_state)

    def object_material_change(self) -> None:
        text = self.ui.objectMaterial.lineEdit().text()
        try:
            uuid: Optional[UUID] = list(self.state.material_names.keys())[
                list(self.state.material_names.values()).index(text)]
        except ValueError:
            uuid = None
        assert self.state.current_object
        if self.state.objects[self.state.current_object].material == uuid:
            return

        def modify(obj: Union[ObjectData, ObjectListData]) -> None:
            assert isinstance(obj, ObjectData)
            obj.material = uuid

        def update_state() -> None:
            assert self.state.current_object
            name = self.state.object_names[self.state.current_object]
            self.set_state(self.state.with_modify_object(
                self.state.current_object, modify), f'修改对象 {name} 的材料')

        QTimer.singleShot(0, update_state)

    def texture_type_changed(self) -> None:
        text = self.ui.textureType.lineEdit().text()
        assert self.state.current_texture
        texture = None if not text or text not in self.state.texture_types \
            else text
        t = self.state.textures[self.state.current_texture]
        current_text = '' if t.texture is None else t.texture[0]
        if current_text == text:
            return

        def modify(text: TextureData) -> None:
            text.texture = None if texture is None else \
                (texture, [p.default for p in
                           self.state.texture_types[texture].properties()])

        def update_state() -> None:
            assert self.state.current_texture
            name = self.state.texture_names[self.state.current_texture]
            self.set_state(self.state.with_modify_texture(
                self.state.current_texture, modify), f'修改质地 {name} 的类型')

        QTimer.singleShot(0, update_state)

    def material_type_changed(self) -> None:
        text = self.ui.materialType.lineEdit().text()
        assert self.state.current_material
        material = None if not text or text not in self.state.material_types \
            else text
        t = self.state.materials[self.state.current_material]
        current_mat = '' if t.material is None else t.material[0]
        if current_mat == text:
            return

        def modify(mat: MaterialData) -> None:
            mat.material = None if material is None else \
                (material, [p.default for p in
                            self.state.material_types[material].properties()])

        def update_state() -> None:
            assert self.state.current_material
            name = self.state.material_names[self.state.current_material]
            self.set_state(self.state.with_modify_material(
                self.state.current_material, modify), f'修改材料 {name} 的类型')

        QTimer.singleShot(0, update_state)

    def camera_type_changed(self) -> None:
        text = self.ui.cameraType.lineEdit().text()
        current_camera = '' if self.state.camera is None else \
            self.state.camera[0]
        if current_camera == text:
            return
        camera = None if not text or text not in self.state.camera_types \
            else text

        def modify(
                _camera: Optional[Tuple[str, List[Any]]]
        ) -> Optional[Tuple[str, List[Any]]]:
            return None if camera is None else \
                (camera, [p.default for p in
                          self.state.camera_types[camera].properties()])

        def update_state() -> None:
            self.set_state(self.state.with_modify_camera(modify),
                           '修改相机的类型')

        QTimer.singleShot(0, update_state)

    def object_name_changed(self) -> None:
        def modify(obj: Union[ObjectData, ObjectListData]) -> None:
            obj.name = self.ui.objectName_.text()

        assert self.state.current_object
        name = self.state.object_names[self.state.current_object]
        if name == self.ui.objectName_.text():
            return
        self.set_state(self.state.with_modify_object(
            self.state.current_object, modify), f'修改对象 {name} 的名字')

    def texture_name_changed(self) -> None:
        def modify(text: TextureData) -> None:
            text.name = self.ui.textureName.text()

        assert self.state.current_texture
        name = self.state.texture_names[self.state.current_texture]
        if name == self.ui.textureName.text():
            return
        self.set_state(self.state.with_modify_texture(
            self.state.current_texture, modify), f'修改质地 {name} 的名字')

    def material_name_changed(self) -> None:
        def modify(mat: MaterialData) -> None:
            mat.name = self.ui.materialName.text()

        assert self.state.current_material
        name = self.state.material_names[self.state.current_material]
        if name == self.ui.materialName.text():
            return
        self.set_state(self.state.with_modify_material(
            self.state.current_material, modify), f'修改材料 {name} 的名字')

    def object_visible_changed(self, visible: bool) -> None:
        def modify(obj: Union[ObjectData, ObjectListData]) -> None:
            obj.visible = visible

        assert self.state.current_object
        name = self.state.object_names[self.state.current_object]
        if self.state.objects[self.state.current_object].visible == visible:
            return
        self.set_state(self.state.with_modify_object(
            self.state.current_object, modify), f'修改对象 {name} 的可见性')

    def object_current_changed(self,
                               current: Optional[QTreeWidgetItem]) -> None:
        self.set_state(self.state.with_current_object(
            UUID(current.data(0, Qt.UserRole))
            if current is not None else None))

    def texture_current_changed(self,
                                current: Optional[QListWidgetItem]) -> None:
        self.set_state(self.state.with_current_texture(
            UUID(current.data(Qt.UserRole))
            if current is not None else None))

    def material_current_changed(self,
                                 current: Optional[QListWidgetItem]) -> None:
        self.set_state(self.state.with_current_material(
            UUID(current.data(Qt.UserRole))
            if current is not None else None))

    def set_state(self, state: State, action: Optional[str] = None) -> None:
        if action is not None:
            self.insert_history(state, action)
        state.apply_diff(self.state, self)
        self.state = state
        self.save_workspace()


async def render(renderer: v4ray.Renderer,
                 signal: QtCore.pyqtBoundSignal) -> None:
    data = await renderer.render()
    signal.emit(data)


def async_loop(loop: AbstractEventLoop) -> None:
    asyncio.set_event_loop(loop)
    loop.run_forever()


def gui_loop(loop: AbstractEventLoop) -> None:
    app = QApplication(sys.argv)
    app.setApplicationName('v4ray')

    window = MainWindow(loop)
    window.show()

    sys.exit(app.exec_())


def main() -> None:
    loop = asyncio.get_event_loop()
    t = Thread(target=async_loop, args=(loop,), daemon=True)
    t.start()
    gui_loop(loop)


if __name__ == '__main__':
    main()
