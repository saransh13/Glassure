# -*- coding: utf8 -*-
__author__ = 'Clemens Prescher'
import unittest

import numpy as np

from core import Spectrum


class SpectrumTest(unittest.TestCase):

    def test_plus_and_minus_operators(self):
        x = np.linspace(0, 10, 100)
        spectrum1 = Spectrum(x, np.sin(x))
        spectrum2 = Spectrum(x, np.sin(x))

        spectrum3 = spectrum1+spectrum2
        self.assertTrue(np.array_equal(spectrum3._y, np.sin(x)*2))
        self.assertTrue(np.array_equal(spectrum2._y, np.sin(x)*1))
        self.assertTrue(np.array_equal(spectrum1._y, np.sin(x)*1))

        spectrum3 = spectrum1+spectrum1
        self.assertTrue(np.array_equal(spectrum3._y, np.sin(x)*2))
        self.assertTrue(np.array_equal(spectrum1._y, np.sin(x)*1))
        self.assertTrue(np.array_equal(spectrum1._y, np.sin(x)*1))

        spectrum3 = spectrum2-spectrum1
        self.assertTrue(np.array_equal(spectrum3._y, np.sin(x)*0))
        self.assertTrue(np.array_equal(spectrum2._y, np.sin(x)*1))
        self.assertTrue(np.array_equal(spectrum1._y, np.sin(x)*1))

        spectrum3 = spectrum1-spectrum1
        self.assertTrue(np.array_equal(spectrum3._y, np.sin(x)*0))
        self.assertTrue(np.array_equal(spectrum1._y, np.sin(x)*1))
        self.assertTrue(np.array_equal(spectrum1._y, np.sin(x)*1))

    def test_multiply_operator(self):
        x = np.linspace(0, 10, 100)
        spectrum = 2*Spectrum(x, np.sin(x))

        self.assertTrue(np.array_equal(spectrum._y, np.sin(x)*2))

    def test_equality_operator(self):
        x = np.linspace(0, 10, 100)
        spectrum1 = Spectrum(x, np.sin(x))
        spectrum2 = Spectrum(x, np.sin(2*x))

        self.assertTrue(spectrum1 == spectrum1)
        self.assertFalse(spectrum1 == spectrum2)

    def test_binning(self):
        x = np.linspace(2.8, 10.8, 100)

        spectrum = Spectrum(x, np.sin(x))

        binned_spectrum = spectrum.rebin(1)

        self.assertTrue(np.sum(binned_spectrum.y), np.sum(spectrum.y))
        # self.assertLessEqual(np.min(binned_spectrum.x), np.min(x))
        # self.assertEqual(np.min(np.min(binned_)))
        print binned_spectrum.x
        print binned_spectrum.y





