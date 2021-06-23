import copy
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, TypeVar, Any, Tuple, \
    Type, Sequence, Callable, Set
from uuid import UUID, uuid4

import numpy as np
from PyQt5.QtCore import Qt, QObject, QSize, QTimer
from PyQt5.QtGui import QPixmap, QImage, QResizeEvent, QGuiApplication, \
    QDoubleValidator, QColor
from PyQt5.QtWidgets import QMainWindow, QApplication, QTreeWidgetItem, \
    QLayoutItem, QLabel, QLineEdit, QFormLayout, QWidgetItem, QLayout, \
    QListWidgetItem, QPushButton, QHBoxLayout, QColorDialog, QTabWidget, \
    QComboBox

import v4ray_frontend
from ui_mainwindow import Ui_MainWindow
from v4ray_frontend.material import MaterialType
from v4ray_frontend.texture import TextureType
from v4ray_frontend.properties import AnyProperty, FloatProperty, \
    ColorProperty, TextureProperty
from v4ray_frontend.shape import ShapeType

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
                    c = QColorDialog.getColor(initial=color, parent=parent,
                                              title=title)
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
                    c = QColorDialog.getColor(initial=color, parent=parent,
                                              title=title)
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

    # always rebuild
    object_parent: Dict[UUID, Tuple[Optional[UUID], int]]
    object_names: Dict[UUID, str]
    texture_names: Dict[UUID, str]
    material_names: Dict[UUID, str]
    shape_form: Optional[Tuple[str, FormState]]
    texture_form: Optional[Tuple[str, FormState]]
    material_form: Optional[Tuple[str, FormState]]
    valid_textures: Set[UUID]
    valid_materials: Set[UUID]

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
        names = State.calc_unique_name([self.objects[uuid].name
                                        for uuid in self.root_objects])
        self.object_names = dict(zip(self.root_objects, names))
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
                               children=[], visible=False)
        else:
            item = ObjectData(name=name or '', shape=None, material=None,
                              visible=False)
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
            window.ui.materialList, window.ui.materialName]
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
        if not self.current_object:
            window.ui.objectName_.setEnabled(False)
            window.ui.objectName_.setText('')
            window.ui.objectVisible.setEnabled(False)
            window.ui.objectVisible.setChecked(False)
            window.ui.objectMaterial.setEnabled(False)
            window.ui.objectMaterial.lineEdit().setText('')
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
            window.ui.objectMaterial.lineEdit().setText('')  # TODO
            window.ui.objectShape.setEnabled(isinstance(obj, ObjectData))
            window.ui.objectShape.lineEdit().setText(
                '' if isinstance(obj, ObjectListData) or obj.shape is None
                else obj.shape[0])
            window.ui.objectTree.setCurrentItem(
                self.object_uuid_to_widget(window, self.current_object))
        window.ui.textureRemove.setEnabled(bool(self.current_texture))
        if not self.current_texture:
            window.ui.textureType.setEnabled(False)
            window.ui.textureType.lineEdit().setText('')
            window.ui.textureList.setCurrentItem(None)   # type: ignore
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
            window.ui.materialList.setCurrentItem(None)   # type: ignore
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
        for o in blocks:
            o.blockSignals(False)

    @staticmethod
    def array_to_pixmap(image: np.ndarray) -> QPixmap:
        return QPixmap(QImage(
            image.tobytes(), image.shape[1], image.shape[0],
            image.shape[1] * 3, QImage.Format_RGB888))

    def apply_image(self, window: 'MainWindow') -> None:
        if self.render_result:
            pixmap = State.array_to_pixmap(self.render_result)
            window.ui.image.setPixmap(pixmap)
        else:
            window.ui.image.clear()

    def apply_object_tree_item_text(
            self,
            item: Union[ObjectData, ObjectListData]) -> str:
        name = self.object_names[item.key]
        if isinstance(item, ObjectData):
            return name + ' ' + ('✓' if item.visible else '✗')
        return name + '  (组)  ' + ('✓' if item.visible else '✗')

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
        prev_keys = prev
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
                shape_layout.addRow(label, f)  # type: ignore

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
        self.apply_always(window)

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
        for o in blocks:
            o.blockSignals(False)
        self.apply_always(window)


class MainWindow(QMainWindow):
    ui: Ui_MainWindow
    state: State

    def __init__(self) -> None:
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        # state
        self.state = State() \
            .with_more_shapes(v4ray_frontend.shapes) \
            .with_more_textures(v4ray_frontend.textures) \
            .with_more_materials(v4ray_frontend.materials)

        self.state.apply(self)
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
        # resize
        self.setTabPosition(Qt.AllDockWidgetAreas, QTabWidget.North)
        self.tabifyDockWidget(self.ui.dockScene, self.ui.dockMaterial)
        self.tabifyDockWidget(self.ui.dockScene, self.ui.dockTexture)
        self.ui.dockScene.raise_()
        size = QGuiApplication.primaryScreen().size()
        self.resize(QSize(int(0.8 * size.width()), int(0.8 * size.height())))

    def goto_texture(self, uuid: UUID):
        self.ui.dockTexture.raise_()
        self.set_state(self.state.with_current_texture(uuid))

    def shape_form_changed(self, index: int, data: Any) -> None:
        def modify(obj: Union[ObjectData, ObjectListData]) -> None:
            assert isinstance(obj, ObjectData) and obj.shape is not None
            shape_data = obj.shape[1]
            shape_data[index] = data
            obj.shape = obj.shape[0], shape_data
        assert self.state.current_object
        self.set_state(self.state.with_modify_object(
            self.state.current_object, modify))

    def texture_form_changed(self, index: int, data: Any) -> None:
        def modify(text: TextureData) -> None:
            assert text.texture is not None
            texture_data = text.texture[1]
            texture_data[index] = data
            text.texture = text.texture[0], texture_data
        assert self.state.current_texture
        self.set_state(self.state.with_modify_texture(
            self.state.current_texture, modify))

    def material_form_changed(self, index: int, data: Any) -> None:
        def modify(mat: MaterialData) -> None:
            assert mat.material is not None
            material_data = mat.material[1]
            material_data[index] = data
            mat.material = mat.material[0], material_data
        assert self.state.current_material
        self.set_state(self.state.with_modify_material(
            self.state.current_material, modify))

    def object_add(self, group: bool) -> None:
        self.set_state(self.state.with_add_object(group=group))

    def texture_add(self) -> None:
        self.set_state(self.state.with_add_texture())

    def material_add(self) -> None:
        self.set_state(self.state.with_add_material())

    def object_remove(self) -> None:
        widget = self.ui.objectTree.currentItem()
        assert widget
        self.set_state(self.state.with_remove_object(
            UUID(widget.data(0, Qt.UserRole))))

    def texture_remove(self) -> None:
        widget = self.ui.textureList.currentItem()
        assert widget
        self.set_state(self.state.with_remove_texture(
            UUID(widget.data(Qt.UserRole))))

    def material_remove(self) -> None:
        widget = self.ui.materialList.currentItem()
        assert widget
        self.set_state(self.state.with_remove_material(
            UUID(widget.data(Qt.UserRole))))

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
            self.set_state(self.state.with_modify_object(
                self.state.current_object, modify))
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
            self.set_state(self.state.with_modify_texture(
                self.state.current_texture, modify))
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
            self.set_state(self.state.with_modify_material(
                self.state.current_material, modify))
        QTimer.singleShot(0, update_state)

    def object_name_changed(self) -> None:
        def modify(obj: Union[ObjectData, ObjectListData]) -> None:
            obj.name = self.ui.objectName_.text()
        assert self.state.current_object
        self.set_state(self.state.with_modify_object(
            self.state.current_object, modify))

    def texture_name_changed(self) -> None:
        def modify(text: TextureData) -> None:
            text.name = self.ui.textureName.text()
        assert self.state.current_texture
        self.set_state(self.state.with_modify_texture(
            self.state.current_texture, modify))

    def material_name_changed(self) -> None:
        def modify(mat: MaterialData) -> None:
            mat.name = self.ui.materialName.text()
        assert self.state.current_material
        self.set_state(self.state.with_modify_material(
            self.state.current_material, modify))

    def object_visible_changed(self, visible: bool) -> None:
        def modify(obj: Union[ObjectData, ObjectListData]) -> None:
            obj.visible = visible
        assert self.state.current_object
        self.set_state(self.state.with_modify_object(
            self.state.current_object, modify))

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

    def set_state(self, state: State) -> None:
        state.apply_diff(self.state, self)
        self.state = state


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName('dashboard')

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
