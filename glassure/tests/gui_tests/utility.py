# -*- coding: utf8 -*-
import numpy as np

from gui.qt import QtGui, QtCore, QTest


def set_widget_text(widget, txt):
    widget.setText('')
    txt = str(txt)
    QTest.keyClicks(widget, txt)
    QTest.keyClick(widget, QtCore.Qt.Key_Enter)
    QtGui.QApplication.processEvents()


def click_checkbox(checkbox_widget):
    QTest.mouseClick(checkbox_widget, QtCore.Qt.LeftButton, pos=QtCore.QPoint(2, checkbox_widget.height() / 2))


def click_button(widget):
    QTest.mouseClick(widget, QtCore.Qt.LeftButton)


def array_almost_equal(array1, array2, places=3):
    return np.sum(array1 - array2)/len(array1) < 1/(places*10.)