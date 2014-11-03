# -*- coding: utf8 -*-
__author__ = 'Clemens Prescher'

import unittest
import numpy as np
import os
import time

import matplotlib.pyplot as plt

from GlassCalculations import normalize_elemental_abundances, calculate_f_mean_squared, \
    calculate_f_squared_mean, calculate_incoherent_scattering, convert_density_to_atoms_per_cubic_angstrom,\
    calc_fr_from_sq, calc_fr_from_sq_matrix

from Spectrum import Spectrum


class GlassCalculationsTest(unittest.TestCase):
    def setUp(self):
        self.my_abundances = {
            'Fe': 1,
            'Si': 1,
            'O': 3,
        }

    def tearDown(self):
        pass

    def save_graph(self, x, y, name):
        plt.figure()
        plt.plot(x, y)
        plt.tight_layout()
        plt.savefig(os.path.join("output", name))

    def plot_spectrum(self, spectrum):
        x, y = spectrum.data
        plt.plot(x,y)

    # @unittest.skip("blabla")
    def test_normalize_elemental_abundances(self):
        res = normalize_elemental_abundances(self.my_abundances)

        self.assertEqual(res['Si'], 0.2)
        self.assertEqual(res['O'], 0.6)
        self.assertEqual(res['Fe'], 0.2)

        self.assertEqual(self.my_abundances['Si'], 1)

    def test_calculate_f_mean_squared(self):
        q = np.linspace(0, 10, 1000)
        res = calculate_f_mean_squared(self.my_abundances, q)

        self.save_graph(q, res, "f_mean_squared_FeSiO3.png")

    def test_calculate_f_squared_mean(self):
        q = np.linspace(0, 10, 1000)
        res = calculate_f_squared_mean(self.my_abundances, q)

        self.save_graph(q, res, "f_squared_mean_FeSiO3.png")

    def test_calculate_incoherent_scattering(self):
        q = np.linspace(0, 10, 1000)
        res = calculate_incoherent_scattering(self.my_abundances, q)

        self.save_graph(q, res, "incoherent_scattering_FeSiO3")

    def test_convert_density_to_atoms_per_cubic_angstrom(self):
        composition = {
            'Si': 1,
            'O': 2
        }
        density_au = convert_density_to_atoms_per_cubic_angstrom(composition, 2.2)
        self.assertAlmostEqual(density_au, 0.06615204386)

    def test_calc_fr_from_sq(self):
        q = np.linspace(0, 12, 100)
        r = np.linspace(0, 10, 1000)

        sq = np.sin(q)

        fr = calc_fr_from_sq(Spectrum(q, sq), r)

        fr_x, fr_y = fr.data

        self.assertEqual(r.shape, fr_y.shape)
        self.save_graph(fr_x, fr_y, 'f_r_test.png')

    def test_calc_fr_from_sq_matrix(self):
        q = np.linspace(0, 12, 100)
        r = np.linspace(0, 10, 1000)

        sq = np.sin(q)

        fr1 = calc_fr_from_sq(Spectrum(q, sq), r)
        fr2 = calc_fr_from_sq_matrix(Spectrum(q, sq), r)

        fr1_x, fr1_y = fr1.data
        fr2_x, fr2_y = fr2.data

        self.assertEqual(fr1_y.shape, fr2_y.shape)
        self.assertAlmostEqual(np.sum(np.abs(fr2_y-fr1_y)), 0)

        #benchmark
        n = 100

        t1 = time.time()
        for ind in range(n):
            calc_fr_from_sq(Spectrum(q, sq), r)

        print("Normal iterative procedure takes: {}".format(time.time()-t1))

        t1 = time.time()
        for ind in range(n):
            calc_fr_from_sq_matrix(Spectrum(q, sq), r)

        print("Matrix procedure takes: {}".format(time.time()-t1))



