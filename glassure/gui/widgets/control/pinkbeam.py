# -*- coding: utf-8 -*-

from ...qt import QtWidgets, QtCore

from ..custom import FlatButton, HorizontalLine, LabelAlignRight


class PinkBeamWidget(QtWidgets.QWidget):

    def __init__(self, *args):
        super(PinkBeamWidget, self).__init__(*args)

        self.create_widgets()
        self.create_layout()
        self.style_widgets()
        self.create_signals()

        self.load_xray_spectrum.setVisible(False)
        self.activate_cb.setChecked(False)

    def create_widgets(self):
        self.load_xray_spectrum = QtWidgets.QPushButton("Load xray Spectrum")

    def create_layout(self):

        self.pinkbeam_layout = QtWidgets.QHBoxLayout()
        self.activate_cb = QtWidgets.QCheckBox("activate")
        self.pinkbeam_layout.addWidget(self.activate_cb)
        self.pinkbeam_layout.addWidget(self.load_xray_spectrum)

        self.setLayout(self.pinkbeam_layout)

    def style_widgets(self):
        self.load_xray_spectrum.setFlat(True)

    def create_signals(self):
        self.activate_cb.stateChanged.connect(self.load_xray_spectrum.setVisible)
