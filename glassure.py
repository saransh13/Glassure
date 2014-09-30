# -*- coding: utf8 -*-
__author__ = 'Clemens Prescher'

import sys
from PyQt4 import QtGui

from Controller.MainController import MainController

if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    from sys import platform as _platform

    if _platform == "linux" or _platform == "linux2":
        app.setStyle('plastique')
    elif _platform == "win32" or _platform == 'cygwin':
        app.setStyle('plastique')
        # possible values:
        # "windows", "motif", "cde", "plastique", "windowsxp", or "macintosh"
    controller = MainController()
    app.exec_()
    del app