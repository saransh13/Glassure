# -*- coding: utf-8 -*-

from ...qt import QtCore, QtGui, QtWidgets, Signal
from .box import ExpandableBox
from .lines import HorizontalLine
from .pattern import PatternWidget


def VerticalSpacerItem():
    return QtWidgets.QSpacerItem(0, 0, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)


def HorizontalSpacerItem():
    return QtWidgets.QSpacerItem(0, 0, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.MinimumExpanding)


class NumberTextField(QtWidgets.QLineEdit):
    def __init__(self, *args, **kwargs):
        super(NumberTextField, self).__init__(*args, **kwargs)
        self.setValidator(QtGui.QDoubleValidator())
        self.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)


class LabelAlignRight(QtWidgets.QLabel):
    def __init__(self, *args, **kwargs):
        super(LabelAlignRight, self).__init__(*args, **kwargs)
        self.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)


class FlatButton(QtWidgets.QPushButton):
    def __init__(self, *args):
        super(FlatButton, self).__init__(*args)
        self.setFlat(True)


class CheckableFlatButton(QtWidgets.QPushButton):
    def __init__(self, *args):
        super(CheckableFlatButton, self).__init__(*args)
        self.setFlat(True)
        self.setCheckable(True)


class ListTableWidget(QtWidgets.QTableWidget):
    def __init__(self, columns=3):
        super(ListTableWidget, self).__init__()

        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.setColumnCount(columns)
        self.horizontalHeader().setVisible(False)
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().setStretchLastSection(True)
        self.setShowGrid(False)


class ValueLabelTxtPair(QtWidgets.QWidget):
    editingFinished = Signal()

    def __init__(self, label_str, value_init, unit_str, layout, layout_row=0, layout_col=0, parent=None):
        super(ValueLabelTxtPair, self).__init__(parent)

        self.desc_lbl = LabelAlignRight(label_str)
        self.value_txt = NumberTextField(str(value_init))
        self.unit_lbl = LabelAlignRight(unit_str)

        self.layout = layout

        self.layout.addWidget(self.desc_lbl, layout_row, layout_col)
        self.layout.addWidget(self.value_txt, layout_row, layout_col + 1)
        self.layout.addWidget(self.unit_lbl, layout_row, layout_col + 2)

        self.value_txt.editingFinished.connect(self.editingFinished.emit)

    def get_value(self):
        return float(str(self.value_txt.text()))

    def set_value(self, value):
        self.value_txt.setText(str(value))

    def setText(self, new_str):
        self.value_txt.setText(new_str)


