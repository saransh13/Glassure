# -*- coding: utf-8 -*-

from ...qt import QtWidgets, QtCore
import pyqtgraph as pg

from ..custom import FlatButton, HorizontalLine, LabelAlignRight
from ..custom.pattern import ModifiedPlotItem

class PinkBeamWidget(QtWidgets.QWidget):

    def __init__(self, *args):
        super(PinkBeamWidget, self).__init__(*args)

        self.create_widgets()
        self.create_layout()
        self.style_widgets()
        self.cretate_item()
        self.create_signals()

        self.load_xray_spectrum.setVisible(False)
        self.spec_widget.setVisible(False)
        self.activate_cb.setChecked(False)

    def create_widgets(self):
        self.load_xray_spectrum = QtWidgets.QPushButton("Load spectrum")
        self.spectrum_filename_lbl = LabelAlignRight("")

        self.spec_layout = pg.GraphicsLayout()

        self.spec_widget = pg.GraphicsLayoutWidget()
        self.spec_widget.setContentsMargins(0, 0, 0, 0)
        self.spec_widget.setWindowTitle("xray energy spectrum for pink beam")
        self.spec_widget.resize(600, 300)

        self.spec_plot = ModifiedPlotItem()
        self.spec_widget.addItem(self.spec_plot, 1, 1)
        self.spec_widget.addItem(self.spec_layout)

    def create_layout(self):

        self.pinkbeam_layout = QtWidgets.QHBoxLayout()
        self.pinkbeam_layout.setContentsMargins(0, 0, 0, 0)
        self.activate_cb = QtWidgets.QCheckBox("activate")
        self.pinkbeam_layout.addWidget(self.activate_cb)
        self.pinkbeam_layout.addWidget(self.load_xray_spectrum)
        self.pinkbeam_layout.addWidget(self.spectrum_filename_lbl)

        self.setLayout(self.pinkbeam_layout)


    def style_widgets(self):
        self.load_xray_spectrum.setFlat(True)
        self.spec_plot.setLabel('bottom', text='Energy (keV)')
        self.spec_plot.setLabel('left', text='flux (a.u.)')

    def create_signals(self):
        self.activate_cb.stateChanged.connect(self.load_xray_spectrum.setVisible)
        self.spec_plot.connect_mouse_move_event()
        # self.activate_cb.stateChanged.connect(self.spec_widget.setVisible)

    def cretate_item(self):
        self.spec_item = pg.PlotDataItem(pen=pg.mkPen('w', width=1.5))
        self.spec_plot.addItem(self.spec_item)

    def plot_spec(self, spectrum):
        x, y = spectrum.data
        self.spec_item.setData(x=x, y=y)