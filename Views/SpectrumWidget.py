# -*- coding: utf8 -*-
__author__ = 'Clemens Prescher'

import pyqtgraph as pg
import numpy as np
from PyQt4 import QtCore

# TODO refactoring of the 3 lists: overlays, overlay_names, overlay_show,
# should probably a class, making it more readable


class SpectrumWidget(object):

    def __init__(self, pg_layout_widget):
        self.pg_layout_widget = pg_layout_widget
        self.create_plots()
        self.create_items()

    def create_plots(self):
        self.pg_layout = pg.GraphicsLayout()
        self.pg_layout.setContentsMargins(0, 0, 0, 0)
        self.pg_layout_widget.setContentsMargins(0, 0, 0, 0)

        self.spectrum_plot = ModifiedPlotItem()
        self.sq_plot = ModifiedPlotItem()
        self.pdf_plot = ModifiedPlotItem()

        self.pg_layout.addItem(self.spectrum_plot, 0, 0)
        self.pg_layout.addItem(self.sq_plot, 1, 0)
        self.pg_layout.addItem(self.pdf_plot, 2, 0)

        self.pg_layout_widget.addItem(self.pg_layout)

    def create_items(self):
        self.spectrum_item = pg.PlotDataItem(pen=pg.mkPen('w', width=1.5))
        self.bkg_item = pg.PlotDataItem(pen=pg.mkPen('w', width=1.5, style=QtCore.Qt.DashLine))
        self.sq_item = pg.PlotDataItem(pen=pg.mkPen('w', width=1.5))
        self.pdf_item = pg.PlotDataItem(pen=pg.mkPen('w', width=1.5))

        self.spectrum_plot.addItem(self.spectrum_item)
        self.spectrum_plot.addItem(self.bkg_item)
        self.sq_plot.addItem(self.sq_item)
        self.pdf_plot.addItem(self.pdf_item)

    def plot_spectrum(self, spec):
        x, y = spec.data
        self.spectrum_item.setData(x=x, y=y)

    def plot_bkg(self, spectrum):
        x, y = spectrum.data
        self.bkg_item.setData(x=x, y=y)

    def plot_sq(self, spectrum):
        x, y = spectrum.data
        self.sq_item.setData(x=x, y=y)

    def plot_pdf(self, spectrum):
        x, y = spectrum.data
        self.pdf_item.setData(x=x, y=y)


class ModifiedPlotItem(pg.PlotItem):
    mouse_moved = QtCore.pyqtSignal(float, float)
    mouse_left_clicked = QtCore.pyqtSignal(float, float)
    range_changed = QtCore.pyqtSignal(list)

    def __init__(self, *args, **kwargs):
        super(ModifiedPlotItem, self).__init__(*args, **kwargs)

        self.modify_mouse_behavior()

    def modify_mouse_behavior(self):
        self.vb.mouseClickEvent = self.mouse_click_event
        self.vb.mouseDragEvent = self.mouse_drag_event
        self.vb.mouseDoubleClickEvent = self.mouse_double_click_event
        self.vb.wheelEvent = self.wheel_event
        self.range_changed_timer = QtCore.QTimer()
        self.range_changed_timer.timeout.connect(self.emit_sig_range_changed)
        self.range_changed_timer.setInterval(30)
        self.last_view_range = np.array(self.vb.viewRange())

    def connect_mouse_move_event(self):
        self.scene().sigMouseMoved.connect(self.mouse_move_event)

    def mouse_move_event(self, pos):
        if self.sceneBoundingRect().contains(pos):
            pos = self.vb.mapSceneToView(pos)
            self.mouse_moved.emit(pos.x(), pos.y())

    def mouse_click_event(self, ev):
        if ev.button() == QtCore.Qt.RightButton or \
                (ev.button() == QtCore.Qt.LeftButton and
                         ev.modifiers() & QtCore.Qt.ControlModifier):
            self.vb.scaleBy(2)
            self.vb.sigRangeChangedManually.emit(self.vb.state['mouseEnabled'])
        elif ev.button() == QtCore.Qt.LeftButton:
            if self.sceneBoundingRect().contains(ev.pos()):
                pos = self.vb.mapToView(ev.pos())
                x = pos.x()
                y = pos.y()
                self.mouse_left_clicked.emit(x, y)

    def mouse_double_click_event(self, ev):
        if (ev.button() == QtCore.Qt.RightButton) or (ev.button() == QtCore.Qt.LeftButton and
                                                              ev.modifiers() & QtCore.Qt.ControlModifier):
            self.vb.autoRange()
            self.vb.enableAutoRange()
            self._auto_range = True
            self.vb.sigRangeChangedManually.emit(self.vb.state['mouseEnabled'])

    def mouse_drag_event(self, ev, axis=None):
        # most of this code is copied behavior mouse drag from the original code
        ev.accept()
        pos = ev.pos()
        last_pos = ev.lastPos()
        dif = pos - last_pos
        dif *= -1

        if ev.button() == QtCore.Qt.RightButton or \
                (ev.button() == QtCore.Qt.LeftButton and ev.modifiers() & QtCore.Qt.ControlModifier):
            # determine the amount of translation
            tr = dif
            tr = self.vb.mapToView(tr) - self.vb.mapToView(pg.Point(0, 0))
            x = tr.x()
            y = tr.y()
            self.vb.translateBy(x=x, y=y)
            if ev.start:
                self.range_changed_timer.start()
            if ev.isFinish():
                self.range_changed_timer.stop()
                self.emit_sig_range_changed()
        else:
            if ev.isFinish():  # This is the final move in the drag; change the view scale now
                self._auto_range = False
                self.vb.enableAutoRange(enable=False)
                self.vb.rbScaleBox.hide()
                ax = QtCore.QRectF(pg.Point(ev.buttonDownPos(ev.button())), pg.Point(pos))
                ax = self.vb.childGroup.mapRectFromParent(ax)
                self.vb.showAxRect(ax)
                self.vb.axHistoryPointer += 1
                self.vb.axHistory = self.vb.axHistory[:self.vb.axHistoryPointer] + [ax]
                self.vb.sigRangeChangedManually.emit(self.vb.state['mouseEnabled'])
            else:
                # update shape of scale box
                self.vb.updateScaleBox(ev.buttonDownPos(), ev.pos())

    def emit_sig_range_changed(self):
        new_view_range = np.array(self.vb.viewRange())
        if not np.array_equal(self.last_view_range, new_view_range):
            self.vb.sigRangeChangedManually.emit(self.vb.state['mouseEnabled'])
            self.last_view_range = new_view_range

    def wheel_event(self, ev, axis=None, *args):
        pg.ViewBox.wheelEvent(self.vb, ev, axis)
        self.vb.sigRangeChangedManually.emit(self.vb.state['mouseEnabled'])