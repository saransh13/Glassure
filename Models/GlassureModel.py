# -*- coding: utf8 -*-
__author__ = 'Clemens Prescher'

import numpy as np

from .Spectrum import Spectrum
from .HelperModule import Observable
from GlassCalculations import calc_transforms


class GlassureModel(Observable):
    def __init__(self):
        super(GlassureModel, self).__init__()
        self.original_spectrum = Spectrum()
        self.background_spectrum = Spectrum()
        self.background_scaling = 1.0
        self.subtracted_spectrum = Spectrum()
        self.sq_spectrum = Spectrum()
        self.pdf_spectrum = Spectrum()

        self.composition = {}
        self.density = 2.2
        self.q_min = 0.0
        self.q_max = 10.0
        self.r_cutoff = 1.0

    def load_data(self, filename):
        self.original_spectrum.load(filename)
        self.subtracted_spectrum.load(filename)
        self.calculate_spectra()

    def load_bkg(self, filename):
        self.background_spectrum.load(filename)
        self.subtracted_spectrum.set_background(self.background_spectrum)
        self.calculate_spectra()

    def set_bkg_scale(self, scaling):
        self.background_spectrum.scaling = scaling
        self.background_scaling = scaling
        self.calculate_spectra()

    def set_bkg_offset(self, offset):
        self.background_spectrum.offset = offset
        self.calculate_spectra()

    def set_smooth(self, value):
        self.original_spectrum.set_smoothing(value)
        self.subtracted_spectrum.set_smoothing(value)
        self.background_spectrum.set_smoothing(value)
        self.calculate_spectra()

    def update_parameter(self, composition, density, q_min, q_max, r_cutoff):
        print density
        self.composition = composition
        self.density = density
        self.q_min = q_min
        self.q_max = q_max
        self.r_cutoff = r_cutoff
        self.calculate_spectra()

    def calculate_spectra(self):
        if len(self.composition)!=0:
            print 'calculating'
            self.sq_spectrum, _, self.gr_spectrum = calc_transforms(
                self.original_spectrum,
                self.background_spectrum,
                self.background_scaling,
                self.composition,
                self.density,
                np.linspace(0, 10, 1000)
            )
        self.notify()


