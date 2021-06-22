import copy
import sys
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, TypeVar, Any, Tuple, \
    Type, Sequence, Callable
from uuid import UUID, uuid4

import numpy as np
from PyQt5.QtCore import Qt, QObject, QSize, QTimer
from PyQt5.QtGui import QPixmap, QImage, QResizeEvent, QGuiApplication, \
    QDoubleValidator
from PyQt5.QtWidgets import QMainWindow, QApplication, QTreeWidgetItem, \
    QLayoutItem, QLabel, QLineEdit, QFormLayout, QWidgetItem

import v4ray_frontend
from ui_mainwindow import Ui_MainWindow
from v4ray_frontend.properties import AnyProperty, FloatProperty
from v4ray_frontend.shape import Shape

T = TypeVar('T')


@dataclass
class Object:
    key: UUID = field(default_factory=uuid4, init=False)
    name: str
    shape: Optional[Tuple[str, List[Any]]]
    material: Optional[UUID]
    visible: bool = False


@dataclass
class ObjectList:
    key: UUID = field(default_factory=uuid4, init=False)
    name: str
    material: Optional[UUID]
    children: List[UUID] = field(default_factory=list)
    visible: bool = False


class FormState:
    properties: List[AnyProperty]
    values: List[Any]

    def __init__(self, properties: List[AnyProperty],
                 values: Optional[List[Any]] = None):
        self.properties = properties
        self.values = values if values is not None else \
            [p.default for p in properties]

    def data(self) -> List[Any]:
        return self.values

    def apply(
            self, on_new_state: Callable[[int, Any], None]
    ) -> List[Tuple[QLayoutItem, QLayoutItem]]:
        # noinspection PyTypeChecker
        widgets: List[Tuple[QLayoutItem, QLayoutItem]] = []
        for i, (v, p) in enumerate(zip(self.values, self.properties)):
            label = QLabel(p.name)
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
                widgets.append((label, line_edit))  # type: ignore
            else:
                assert False
        return widgets

    def apply_diff(self, prev: 'FormState',
                   widgets: List[Tuple[QLayoutItem, QLayoutItem]]) -> None:
        for i, (v, p, (_, f)) in enumerate(zip(
                self.values, self.properties, widgets)):
            if isinstance(p, FloatProperty):
                assert isinstance(f, QWidgetItem)
                line_edit = f.widget()
                assert isinstance(line_edit, QLineEdit)
                line_edit.blockSignals(True)
                line_edit.setText(str(v))
                line_edit.blockSignals(False)


class State:
    render_result: Optional[np.ndarray]
    root_objects: List[UUID]
    objects: Dict[UUID, Union[Object, ObjectList]]
    current_object: Optional[UUID]
    shapes: Dict[str, Type[Shape]]
    # always rebuild
    object_parent: Dict[UUID, Tuple[Optional[UUID], int]]
    shape_form: Optional[Tuple[str, FormState]]

    def __init__(
            self,
            prev_state: Optional['State'] = None,
    ):
        self.render_result = prev_state.render_result \
            if prev_state is not None else None
        self.root_objects = prev_state.root_objects \
            if prev_state is not None else []
        self.objects = prev_state.objects if prev_state is not None else {}
        self.current_object = prev_state.current_object \
            if prev_state is not None else None
        self.shapes = prev_state.shapes if prev_state is not None else {}
        self.shape_form = prev_state.shape_form \
            if prev_state is not None else None
        self.object_parent = {c: (k, i) for k, obj in self.objects.items()
                              if isinstance(obj, ObjectList)
                              for i, c in enumerate(obj.children)}
        for i, k in enumerate(self.root_objects):
            self.object_parent[k] = None, i
        self.shape_form = None
        if self.current_object is not None:
            obj = self.objects[self.current_object]
            if isinstance(obj, Object) and obj.shape is not None:
                self.shape_form = obj.shape[0], FormState(
                    properties=self.shapes[obj.shape[0]].properties(),
                    values=obj.shape[1])

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

    def with_more_shapes(self, shapes: Sequence[Type[Shape]]) -> 'State':
        state = copy.deepcopy(self)
        for shape in shapes:
            kind = shape.kind()
            assert kind not in self.shapes
            state.shapes[kind] = shape
        return State(state)

    def with_modify_object(
            self, uuid: UUID,
            op: Callable[[Union[Object, ObjectList]], Any]) -> 'State':
        state = copy.deepcopy(self)
        op(state.objects[uuid])
        return State(state)

    def with_current_object(self, uuid: Optional[UUID]) -> 'State':
        state = copy.deepcopy(self)
        state.current_object = uuid
        return State(state)

    def with_remove_object(self, uuid: UUID) -> 'State':
        state = copy.deepcopy(self)

        def recursive_remove(uuid2: UUID) -> None:
            if state.current_object == uuid2:
                state.current_object = None
            obj = state.objects.pop(uuid2)
            if isinstance(obj, ObjectList):
                for child in obj.children:
                    recursive_remove(child)
        parent, index = self.object_parent[uuid]
        if parent is not None and index != 0:
            parent_object = state.objects[parent]
            assert isinstance(parent_object, ObjectList)
            state.current_object = parent_object.children[index - 1]
        elif parent is None and index != 0:
            state.current_object = self.root_objects[index - 1]
        else:
            state.current_object = parent
        if parent is None:
            state.root_objects.remove(uuid)
        else:
            parent_object = state.objects[parent]
            assert isinstance(parent_object, ObjectList)
            parent_object.children.remove(uuid)
        recursive_remove(uuid)
        return State(state)

    def with_add_object(self, name: Optional[str] = None,
                        group: bool = True) -> 'State':
        if not self.current_object:
            root: Optional[UUID] = None
            index = len(self.root_objects)
        else:
            obj = self.objects[self.current_object]
            if isinstance(obj, ObjectList):
                root = self.current_object
                index = len(obj.children)
            else:
                root, index = self.object_parent[self.current_object]
                index += 1
        state = copy.deepcopy(self)
        state.objects = state.objects.copy()
        if group:
            item: Union[Object, ObjectList] = \
                ObjectList(name=name or '未命名', material=None,
                           children=[], visible=False)
        else:
            item = Object(name=name or '未命名', shape=None, material=None,
                          visible=False)
        state.objects[item.key] = item
        if root is None:
            state.root_objects.insert(index, item.key)
        else:
            parent = state.objects[root]
            assert isinstance(parent, ObjectList)
            parent.children.insert(index, item.key)
        state.current_object = item.key
        return State(state)

    def apply_always(self, window: 'MainWindow') -> None:
        blocks: List[QObject] = [
            window.ui.objectTree, window.ui.objectClearSelection,
            window.ui.objectRemove, window.ui.objectName_,
            window.ui.objectVisible, window.ui.objectMaterial,
            window.ui.shapeType]
        for o in blocks:
            o.blockSignals(True)
        if self.current_object:
            window.ui.objectTree.setCurrentItem(
                self.object_uuid_to_widget(window, self.current_object))
        else:
            window.ui.objectTree.setCurrentItem(None)  # type: ignore
        window.ui.objectClearSelection.setEnabled(bool(self.current_object))
        window.ui.objectRemove.setEnabled(bool(self.current_object))
        if not self.current_object:
            window.ui.objectName_.setEnabled(False)
            window.ui.objectName_.setText('')
            window.ui.objectVisible.setEnabled(False)
            window.ui.objectVisible.setChecked(False)
            window.ui.objectMaterial.setEnabled(False)
            window.ui.objectMaterial.lineEdit().setText('')
            window.ui.shapeType.setEnabled(False)
            window.ui.shapeType.lineEdit().setText('')  # TODO
        else:
            obj = self.objects[self.current_object]
            window.ui.objectName_.setEnabled(True)
            window.ui.objectName_.setText(obj.name)
            window.ui.objectVisible.setEnabled(True)
            window.ui.objectVisible.setChecked(obj.visible)
            window.ui.objectMaterial.setEnabled(True)
            window.ui.objectMaterial.lineEdit().setText('')  # TODO
            window.ui.shapeType.setEnabled(isinstance(obj, Object))
            window.ui.shapeType.lineEdit().setText(
                '' if isinstance(obj, ObjectList) or obj.shape is None
                else obj.shape[0])
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

    @staticmethod
    def apply_object_tree_item_text(item: Union[Object, ObjectList]) -> str:
        if isinstance(item, Object):
            return item.name + ' ' + ('✓' if item.visible else '✗')
        return item.name + '  (组)  ' + ('✓' if item.visible else '✗')

    def apply_object_tree_item(
            self, item: Union[Object, ObjectList]) -> QTreeWidgetItem:
        widget = QTreeWidgetItem([State.apply_object_tree_item_text(item)])
        if isinstance(item, Object):
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
        if isinstance(prev_item, Object) or isinstance(curr_item, Object):
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
        widget.setText(0, State.apply_object_tree_item_text(curr_item))
        if not prev_item.children:
            widget.setExpanded(True)
        return None

    def apply(self, window: 'MainWindow') -> None:
        blocks: List[QObject] = [
            window.ui.image, window.ui.objectTree, window.ui.shapeType
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
        for o in blocks:
            o.blockSignals(False)
        # shape type
        window.ui.shapeType.clear()
        for shape in self.shapes:
            window.ui.shapeType.addItem(shape)
        # TODO: shape form
        self.apply_always(window)

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

        blocks: List[QObject] = [
            window.ui.image, window.ui.objectTree, window.ui.shapeType
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
        # shape type
        for shape in set(prev_state.shapes) - set(self.shapes):
            window.ui.shapeType.removeItem(window.ui.shapeType.findText(shape))
        for shape in set(self.shapes) - set(prev_state.shapes):
            window.ui.shapeType.addItem(shape)
        # shape form
        shape_layout: QFormLayout = \
            window.ui.shapeProperties.layout()  # type: ignore
        if self.shape_form is None:
            if prev_state.shape_form is not None:
                for i in range(shape_layout.rowCount() - 1, 0, -1):
                    shape_layout.removeRow(i)
        elif prev_state.shape_form is None or \
                self.shape_form[0] != prev_state.shape_form[0]:
            for i in range(shape_layout.rowCount() - 1, 0, -1):
                shape_layout.removeRow(i)
            widgets = self.shape_form[1].apply(window.shape_form_changed)
            for label, f in widgets:
                shape_layout.addRow(label, f)  # type: ignore
        else:
            widgets = [(shape_layout.itemAt(i, QFormLayout.LabelRole),
                        shape_layout.itemAt(i, QFormLayout.FieldRole))
                       for i in range(1, shape_layout.rowCount())]
            self.shape_form[1].apply_diff(prev_state.shape_form[1], widgets)
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
            .with_more_shapes(v4ray_frontend.shapes)

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
        self.ui.shapeType.lineEdit().editingFinished.connect(
            self.object_shape_changed)
        # resize
        size = QGuiApplication.primaryScreen().size()
        self.resize(QSize(int(0.8 * size.width()), int(0.8 * size.height())))

    def shape_form_changed(self, index: int, data: Any) -> None:
        def modify(obj: Union[Object, ObjectList]) -> None:
            assert isinstance(obj, Object) and obj.shape is not None
            shape_data = obj.shape[1][:]
            shape_data[index] = data
            obj.shape = obj.shape[0], shape_data
        assert self.state.current_object
        state = self.state.with_modify_object(self.state.current_object, modify)
        state.apply_diff(self.state, self)
        self.state = state

    def object_add(self, group: bool) -> None:
        state = self.state.with_add_object(group=group)
        state.apply_diff(self.state, self)
        self.state = state

    def object_remove(self) -> None:
        widget = self.ui.objectTree.currentItem()
        assert widget
        state = self.state.with_remove_object(UUID(widget.data(0, Qt.UserRole)))
        state.apply_diff(self.state, self)
        self.state = state

    def object_shape_changed(self) -> None:
        text = self.ui.shapeType.lineEdit().text()
        assert self.state.current_object
        shape = None if not text or text not in self.state.shapes else text
        obj = self.state.objects[self.state.current_object]
        assert isinstance(obj, Object)
        current_shape = '' if obj.shape is None else obj.shape[0]
        if current_shape == text:
            return

        def modify(obj: Union[Object, ObjectList]) -> None:
            assert isinstance(obj, Object)
            obj.shape = None if shape is None else \
                (shape, [p.default for p in
                         self.state.shapes[shape].properties()])

        def update_state() -> None:
            assert self.state.current_object
            state = self.state.with_modify_object(self.state.current_object,
                                                  modify)
            state.apply_diff(self.state, self)
            self.state = state
        QTimer.singleShot(0, update_state)

    def object_name_changed(self) -> None:
        def modify(obj: Union[Object, ObjectList]) -> None:
            obj.name = self.ui.objectName_.text()
        assert self.state.current_object
        state = self.state.with_modify_object(self.state.current_object, modify)
        state.apply_diff(self.state, self)
        self.state = state

    def object_visible_changed(self, visible: bool) -> None:
        def modify(obj: Union[Object, ObjectList]) -> None:
            obj.visible = visible
        assert self.state.current_object
        state = self.state.with_modify_object(self.state.current_object, modify)
        state.apply_diff(self.state, self)
        self.state = state

    def object_current_changed(self,
                               current: Optional[QTreeWidgetItem]) -> None:
        state = self.state.with_current_object(
            UUID(current.data(0, Qt.UserRole))
            if current is not None else None)
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
