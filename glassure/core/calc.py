# -*- coding: utf-8 -*-

import numpy as np
import lmfit

from . import Pattern
from .utility import calculate_incoherent_scattering, calculate_f_squared_mean, \
                     calculate_f_mean_squared, convert_density_to_atoms_per_cubic_angstrom,\
                     normalize_composition

from .scattering_factors import calculate_coherent_scattering_factor_derivative, \
                                calculate_coherent_scattering_factor, \
                                get_atomic_number

__all__ = ['calculate_normalization_factor_raw', 'calculate_normalization_factor', 'fit_normalization_factor',
           'calculate_sq', 'calculate_sq_raw', 'calculate_sq_from_fr', 'calculate_sq_from_gr',
           'calculate_fr', 'calculate_gr_raw', 'calculate_gr']


def calculate_normalization_factor_raw(sample_pattern, atomic_density, f_squared_mean, f_mean_squared,
                                       incoherent_scattering=None, attenuation_factor=0.001):
    """
    Calculates the normalization factor for a sample pattern given all the parameters. If you do not have them
    already calculated please consider using calculate_normalization_factor, which has an easier interface since it
    just requires density and composition as parameters.

    :param sample_pattern:     background subtracted sample pattern
    :param atomic_density:      density in atoms per cubic Angstrom
    :param f_squared_mean:      <f^2>
    :param f_mean_squared:      <f>^2
    :param incoherent_scattering: compton scattering from sample, if set to None, it will not be used
    :param attenuation_factor:  attenuation factor used in the exponential, in order to correct for the q cutoff

    :return:                    normalization factor
    """
    q, intensity = sample_pattern.data
    # calculate values for integrals
    if incoherent_scattering is None:
        incoherent_scattering = np.zeros_like(q)
    n1 = q ** 2 * ((f_squared_mean + incoherent_scattering) * np.exp(-attenuation_factor * q ** 2)) / \
         f_mean_squared
    n2 = q ** 2 * intensity * np.exp(-attenuation_factor * q ** 2) / f_mean_squared

    n = ((-2 * np.pi ** 2 * atomic_density + np.trapz(q, n1)) / np.trapz(q, n2))

    return n


def calculate_normalization_factor(sample_pattern, density, composition, attenuation_factor=0.001,
                                   use_incoherent_scattering=True):
    """
    Calculates the normalization factor for a background subtracted sample pattern based on density and composition.

    :param sample_pattern:     background subtracted sample pattern with A-1 as x unit
    :param density:             density in g/cm^3
    :param composition:         composition as a dictionary with the elements as keys and the abundances as values
    :param attenuation_factor:  attenuation factor used in the exponential, in order to correct for the q cutoff
    :use_incoherent_scattering: whether to use incoherent scattering, in some cases it is already subtracted

    :return: normalization factor
    """
    q, intensity = sample_pattern.data

    f_squared_mean = calculate_f_squared_mean(composition, q)
    f_mean_squared = calculate_f_mean_squared(composition, q)
    if use_incoherent_scattering:
        incoherent_scattering = calculate_incoherent_scattering(composition, q)
    else:
        incoherent_scattering = None
    atomic_density = convert_density_to_atoms_per_cubic_angstrom(composition, density)

    return calculate_normalization_factor_raw(sample_pattern, atomic_density, f_squared_mean, f_mean_squared,
                                              incoherent_scattering, attenuation_factor)


def fit_normalization_factor(sample_pattern, composition, q_cutoff=3, method="squared", use_incoherent_scattering=True):
    """
    Estimates the normalization factor n for calculating S(Q) by fitting

        (Intensity*n-Multiple Scattering) * Q^2
    to
        (Incoherent Scattering + Self Scattering) * Q^2

    where n and Multiple Scattering are free parameters

    :param sample_pattern:      background subtracted sample pattern with A^-1 as x unit
    :param composition:         composition as a dictionary with the elements as keys and the abundances as values
    :param q_cutoff:            q value above which the fitting will be performed, default = 3
    :param method:              specifies whether q^2 ("squared") or q (linear) should be used
    :use_incoherent_scattering: whether to use incoherent scattering, in some cases it is already subtracted

    :return: normalization factor
    """
    q, intensity = sample_pattern.limit(q_cutoff, 100000).data

    if method == "squared":
        x = q ** 2
    elif method == "linear":
        x = q
    else:
        raise NotImplementedError("{} is not an allowed method for fit_normalization_factor".format(method))

    theory = calculate_f_squared_mean(composition, q) * x
    if use_incoherent_scattering:
        theory += x * calculate_incoherent_scattering(composition, q)

    params = lmfit.Parameters()
    params.add("n", value=1, min=0)
    params.add("multiple", value=1, min=0)

    def optimization_fcn(params, q, sample_intensity, theory_intensity):
        n = params['n'].value
        multiple = params['multiple'].value
        return ((sample_intensity * n - multiple) * x - theory_intensity) ** 2

    out = lmfit.minimize(optimization_fcn, params, args=(q, intensity, theory))
    return out.params['n'].value


def calculate_sq_raw(sample_pattern, f_squared_mean, f_mean_squared, 
                    incoherent_scattering=None, normalization_factor=1, 
                    method='FZ'):
    """
    Calculates the structure factor of a material with the given parameters. Using the equation:

    S(Q) = (n * Intensity - incoherent_scattering - <f>^2-)/<f^2> + 1

    where n is the normalization factor and f are the scattering factors.

    :param sample_pattern:       background subtracted sample pattern with A^-1 as x unit
    :param f_squared_mean:        <f^2>
    :param f_mean_squared:        <f>^2
    :param incoherent_scattering: compton scattering from sample
    :param normalization_factor:  previously calculated normalization factor, if None, it will not be subtracted
    :param method:                describing the method to calculate the structure factor, possible values are
                                    - 'AL' - Ashcroft-Langreth
                                    - 'FZ' - Faber-Ziman
    :return: S(Q) pattern
    """
    q, intensity = sample_pattern.data
    if incoherent_scattering is None:
        incoherent_scattering = np.zeros_like(q)

    if method == 'FZ':
        sq = (normalization_factor * intensity - incoherent_scattering - f_squared_mean + f_mean_squared) / \
             f_mean_squared
    elif method == 'AL':
        sq = (normalization_factor * intensity - incoherent_scattering) / f_squared_mean
    else:
        raise NotImplementedError('{} method is not implemented'.format(method))
    return Pattern(q, sq)


def calculate_sq(sample_pattern, density, composition, attenuation_factor=0.001, method='FZ',
                 normalization_method='int', use_incoherent_scattering=True):
    """
    Calculates the structure factor of a material with the given parameters. Using the equation:

    S(Q) = (n * Intensity - incoherent_scattering - <f>^2-)/<f^2> + 1

    where n is the normalization factor and f are the scattering factors. All parameters from the equation are
    calculated from the density, composition and the sample pattern

    :param sample_pattern:     background subtracted sample pattern with A^-1 as x unit
    :param density:             density of the sample in g/cm^3
    :param composition:         composition as a dictionary with the elements as keys and the abundances as values
    :param attenuation_factor:  attenuation factor used in the exponential for the calculation of the normalization
                                factor
    :param method:              describing the method to calculate the structure factor, possible values are
                                    - 'AL' - Ashcroft-Langreth
                                    - 'FZ' - Faber-Ziman

    :param normalization_method: determines the method used for estimating the normalization method. possible values are
                                 'int' for an integral or 'fit' for fitting the high q region form factors.

    :use_incoherent_scattering: whether to use incoherent scattering, in some cases it is already subtracted

    :return: S(Q) pattern
    """
    q, intensity = sample_pattern.data
    f_squared_mean = calculate_f_squared_mean(composition, q)
    f_mean_squared = calculate_f_mean_squared(composition, q)
    if use_incoherent_scattering:
        incoherent_scattering = calculate_incoherent_scattering(composition, q)
    else:
        incoherent_scattering = None

    atomic_density = convert_density_to_atoms_per_cubic_angstrom(composition, density)
    if normalization_method == 'fit':
        normalization_factor = fit_normalization_factor(sample_pattern, composition, use_incoherent_scattering)
    else:
        normalization_factor = calculate_normalization_factor_raw(sample_pattern,
                                                                  atomic_density,
                                                                  f_squared_mean,
                                                                  f_mean_squared,
                                                                  incoherent_scattering,
                                                                  attenuation_factor)
    return calculate_sq_raw(sample_pattern,
                            f_squared_mean,
                            f_mean_squared,
                            incoherent_scattering,
                            normalization_factor,
                            method)


def calculate_fr(sq_pattern, r=None, use_modification_fcn=False, method='integral'):
    """
    Calculates F(r) from a given S(Q) pattern for r values. If r is none a range from 0 to 10 with step 0.01 is used.
    A Lorch modification function of the form:

        m = sin(q*pi/q_max)/(q*pi/q_max)

    can be used to address issues with a low q_max. This will broaden the sharp peaks in g(r)

    :param sq_pattern:              Structure factor S(Q) with lim_inf S(Q) = 1 and unit(q)=A^-1
    :param r:                       numpy array giving the r-values for which F(r) will be calculated,
                                    default is 0 to 10 with 0.01 as a step. units should be in Angstrom.
    :param use_modification_fcn:    boolean flag whether to use the Lorch modification function
    :param method:                  determines the method used for calculating fr, possible values are:
                                            - 'integral' solves the Fourier integral, by calculating the integral
                                            - 'fft' solves the Fourier integral by using fast fourier transformation

    :return: F(r) pattern
    """
    if r is None:
        r = np.linspace(0, 10, 1001)

    q, sq = sq_pattern.data
    if use_modification_fcn:
        modification = np.sin(q * np.pi / np.max(q)) / (q * np.pi / np.max(q))
    else:
        modification = 1

    if method == 'integral':
        fr = 2.0 / np.pi * np.trapz(modification * q * (sq - 1) * \
                                    np.array(np.sin(np.outer(q.T, r))).T, q)
    elif method == 'fft':
        q_step = q[1] - q[0]
        r_step = r[1] - r[0]

        n_out = np.max([len(q), int(np.pi / (r_step * q_step))])
        q_max_for_ifft = 2 * n_out * q_step
        y_for_ifft = np.concatenate((modification * q * (sq - 1), np.zeros(2 * n_out - len(q))))

        ifft_result = np.fft.ifft(y_for_ifft) * 2 / np.pi * q_max_for_ifft
        ifft_imag = np.imag(ifft_result)[:n_out]
        ifft_x_step = 2 * np.pi / q_max_for_ifft
        ifft_x = np.arange(n_out) * ifft_x_step

        fr = np.interp(r, ifft_x, ifft_imag)
    else:
        raise NotImplementedError("{} is not an allowed method for calculate_fr".format(method))
    return Pattern(r, fr)


def calculate_sq_from_fr(fr_pattern, q, method='integral'):
    """
    Calculates S(Q) from a F(r) pattern for given q values.

    :param fr_pattern:              input F(r) pattern
    :param q:                       numpy array giving the q-values for which S(q) will be calculated,
    :param method:                  determines the method use for calculating fr, possible values are:
                                            - 'integral' solves the Fourier integral, by calculating the integral
                                            - 'fft' solves the Fourier integral by using fast fourier transformation

    :return: F(r) pattern
    """
    r, fr = fr_pattern.data

    if method == 'integral':
        sq = np.trapz(fr * np.array(np.sin(np.outer(r.T, q))).T, r) / q + 1

    elif method == 'fft':
        q_step = q[1] - q[0]
        r_step = r[1] - r[0]

        n_out = int(np.pi / (r_step * q_step))

        r_max_for_ifft = 2 * n_out * r_step
        ifft_x_step = 2 * np.pi / r_max_for_ifft
        ifft_x = np.arange(n_out) * ifft_x_step

        y_for_ifft = np.concatenate((fr, np.zeros(2 * n_out - len(r))))
        ifft_result = np.fft.ifft(y_for_ifft) * r_max_for_ifft
        ifft_imag = np.imag(ifft_result)[:n_out]

        sq = np.interp(q, ifft_x, ifft_imag) / q + 1
    else:
        raise NotImplementedError("{} is not an allowed method for calculate_sq_from_fr".format(method))

    return Pattern(q, sq)


def calculate_sq_from_gr(gr_pattern, q, density, composition, method='integral'):
    """
    Performs a back Fourier transform from the pair distribution function g(r)

    :param gr_pattern:      g(r) pattern
    :param q:               numpy array of q values for which S(Q) should be calculated
    :param density:         density of the sample in g/cm^3
    :param composition:     composition as a dictionary with the elements as keys and the abundances as values

    :return: S(Q) pattern
    """
    atomic_density = convert_density_to_atoms_per_cubic_angstrom(composition, density)
    r, gr = gr_pattern.data

    # removing the nan value at the first index, which is caused by the division by zero when r started from zero
    if np.isnan(gr[0]):
        gr[0] = 0
    fr_pattern = Pattern(r, (gr - 1) * (4.0 * np.pi * r * atomic_density))
    return calculate_sq_from_fr(fr_pattern, q, method)


def calculate_gr_raw(fr_pattern, atomic_density):
    """
    Calculates a g(r) pattern from a given F(r) pattern and the atomic density

    :param fr_pattern:     F(r) pattern
    :param atomic_density:  atomic density in atoms/A^3

    :return: g(r) pattern
    """
    r, f_r = fr_pattern.data
    g_r = 1 + f_r / (4.0 * np.pi * r * atomic_density)
    return Pattern(r, g_r)


def calculate_gr(fr_pattern, density, composition):
    """
    Calculates a g(r) pattern from a given F(r) pattern, the material density and composition.

    :param fr_pattern:     F(r) pattern
    :param density:         density in g/cm^3
    :param composition:     composition as a dictionary with the elements as keys and the abundances as values

    :return: g(r) pattern
    """
    return calculate_gr_raw(fr_pattern, convert_density_to_atoms_per_cubic_angstrom(composition, density))

def calc_Z_total(composition):
    norm_elemental_abundances = normalize_composition(composition)

    res = 0
    for key, value in norm_elemental_abundances.items():
        res += value * get_atomic_number(key)
    return res

def calc_df_dE(composition, theta, E):
    """
    =======================================================================
    >> @DATE:   02/06/2020  SS 1.0 original
    >> @AUTHOR: Saransh Singh Lawrence Livermore national Lab
    >> @DETAIL: calculates derivative of the form factor with energy

    >> @param   composition
    >> @param   theta scattering anle
    >> @param   E assumed energy of xray beam

    >> @return  theta array which converts q to scattering angle
    =======================================================================
    """
    norm_elemental_abundances = normalize_composition(composition)

    # plancks constant times speed of light in kev A
    q   = calc_q_from_theta(theta, E)
    res = 0
    for key, value in norm_elemental_abundances.items():
        res += value * calculate_coherent_scattering_factor_derivative(key, q, E)
    return res

def calc_sum_of_product_scattering_factors_derivatives(composition, q):
    """
    =======================================================================
    >> @DATE:   02/07/2020  SS 1.0 original
    >> @AUTHOR: Saransh Singh Lawrence Livermore national Lab
    >> @DETAIL: calculates sum of product for form factors and derivative
                i.e. sum (f_p * df_p / dq)

    >> @param   composition
    >> @param   q scattering parameter

    >> @return  sum (f_p * df_p / dq)
    =======================================================================
    """
    norm_elemental_abundances = normalize_composition(composition)
    res = 0
    for key, value in norm_elemental_abundances.items():
        res += value * calculate_coherent_scattering_factor_derivative(key, q) * \
               calculate_coherent_scattering_factor(key, q)
    return res

def calc_product_of_sum_scattering_factors_derivatives(composition, q):
    """
    =======================================================================
    >> @DATE:   02/07/2020  SS 1.0 original
    >> @AUTHOR: Saransh Singh Lawrence Livermore national Lab
    >> @DETAIL: calculates product of sums for form factors and derivative
                i.e. sum (f_p) * sum (df_p / dq)

    >> @param   composition
    >> @param   q scattering parameter

    >> @return  sum (f_p) * sum (df_p / dq)
    =======================================================================
    """
    norm_elemental_abundances = normalize_composition(composition)
    res_f  = 0
    res_df = 0
    for key, value in norm_elemental_abundances.items():
        res_df += calculate_coherent_scattering_factor_derivative(key, q) 
        res_f  += value * calculate_coherent_scattering_factor(key, q)
    return res_f * res_df

def calc_theta_from_q(q, E):
    """
    =======================================================================
    >> @DATE:   02/06/2020  SS 1.0 original
    >> @AUTHOR: Saransh Singh Lawrence Livermore national Lab
    >> @DETAIL: converts q to theta given an energy

    >> @param   q scattering parameter = 4 * pi * sin(theta) / lambda (in A^-1)
    >> @param   E  assumed energy of xray beam in keV

    >> @return  theta array which converts q to scattering angle
                (could have NaNs if the scattering parameter is not allowed!)
    =======================================================================
    """
    # plancks constant times speed of light in kev A
    hc      = 12.39841984 
    theta   = np.arcsin(q * hc / 4.0 / np.pi / E)

def calc_q_from_theta(theta, E):
    """
    =======================================================================
    >> @DATE:   02/06/2020  SS 1.0 original
    >> @AUTHOR: Saransh Singh Lawrence Livermore national Lab
    >> @DETAIL: converts q to theta given an energy

    >> @param   theta scattering angle (in radians)
    >> @param   E  assumed energy of xray beam in keV

    >> @return  q array which converts scattering angle to scattering 
                parameter (in A^-1)
                q = 4 * pi * sin(theta) / lambda = 4 * pi * sin(theta) * E / hc
    =======================================================================
    """
    # plancks constant times speed of light in kev A
    hc      = 12.39841984 
    q       = 4.0 * np.pi * np.sin(theta) * E / hc
    return q


def calc_weighted_delE(pinkbeam_spectrum):
    """
    =======================================================================
    >> @DATE:   02/06/2020  SS 1.0 original
    >> @AUTHOR: Saransh Singh Lawrence Livermore national Lab
    >> @DETAIL: calculates the weighted mean of the deviation from peak energy
                delE = sum [ w(E) * (E - Emax) ] such that sum [ w(E) ] = 1.0

    >> @param   pinkbeam_spec, pectrum of pink beam if beam is not mono

    >> @return  delE weighted mean of the deviation from peak energy
    >> @return  Emax maximum energy in the spectrum
    =======================================================================
    """
    ''' 
        get data and normalize
    '''
    E, flux = pinkbeam_spectrum.data
    ww      = flux / np.trapz(flux)

    '''
        extract energy with peak flux and calculate the averages
    '''
    amax = np.argmax(flux)
    Emax = E[amax]

    dE   = E - Emax
    delE = np.average(dE, weights=ww) 

    return delE, Emax

def calc_sinqroq(r, q):
    """
    =======================================================================
    >> @DATE:   02/07/2020  SS 1.0 original
    >> @AUTHOR: Saransh Singh Lawrence Livermore national Lab
    >> @DETAIL: this function evaluates the sin(qr)/q function
                encodes analytical result

    >> @param   r distance in real space
    >> @param   q distance in reciprocal space 

    >> @return  sin(qr)/q
    =======================================================================
    """
    eps = 1e-8
    if(np.abs(q) < eps):
        return r

    return np.sin(q*r) / q

def calc_dsincqr_dq(r, q):
    """
    =======================================================================
    >> @DATE:   02/06/2020  SS 1.0 original
    >> @AUTHOR: Saransh Singh Lawrence Livermore national Lab
    >> @DETAIL: this function evaluates the derivative of the sinc function
                encodes analytical result

    >> @param   r distance in real space
    >> @param   q distance in reciprocal space 

    >> @return  d(sin(r*q) / q) / dq = [qr*cos(qr) - sin(qr)]/q^2
    =======================================================================
    """
    eps = 1e-8
    if(np.abs(q) < eps):
        return 0.0

    if(np.abs(q) < eps):
        return 0.0

    x = q * r
    return ( x * np.cos(x) - np.sin(x) ) / (q *q )


def calc_dIcoh_dE(composition, Fr, r, q, E):
    """
    =======================================================================
    >> @DATE:   02/06/2020  SS 1.0 original
    >> @AUTHOR: Saransh Singh Lawrence Livermore national Lab
    >> @DETAIL: calculates derivative of the coherent intensity with energy

    >> @param   composition
    >> @param   Fr = 4 * pi * r * (rho - rho_0)

    >> @return  theta array which converts q to scattering angle
    =======================================================================
    """
    pre = 2.0 * q / E

    t1  = calc_sum_of_product_scattering_factors_derivatives(composition, q) 
    t2  = calc_product_of_sum_scattering_factors_derivatives(composition, q)
    f_mean_sq = calculate_f_mean_squared(composition, q)

    integral1 = np.array([np.trapz(Fr * calc_sinqroq(r, qq), r) for qq in q])
    integral2 = np.array([np.trapz(Fr * calc_dsincqr_dq(r, qq), r) for qq in q])

    res = pre * (t1 + t2 * integral1 + f_mean_sq * integral2)

    return res

def calc_pbcorrection(sample_pattern, pinkbeam_spectrum, composition,
                      density, r=None, use_modification_fcn=False):
    """
    =======================================================================
    >> @DATE:   02/06/2020  SS 1.0 original
    >> @AUTHOR: Saransh Singh Lawrence Livermore national Lab
    >> @DETAIL: calcultes the correction due to pink beam, given the spectrum 
                of the incident beam. effectively transforms the coherent 
                intensity to an "equivalent" monochromatic intensity for the 
                peak energy in the spectrum

    >> @param   sample_pattern experimentally measured spectra
    >> @param   pinkbeam_spec, pectrum of pink beam if beam is not mono
    >> @param   composition

    >> @return  sample_pattern (input) is updated with corrected data
    =======================================================================
    """
    q, intensity = sample_pattern.data
    delE, Emax = calc_weighted_delE(pinkbeam_spectrum)

    attenuation_factor = 0.001

    sq_pattern = calculate_sq(sample_pattern, 
                            density, 
                            composition, 
                            attenuation_factor=attenuation_factor, 
                            method='FZ',
                            normalization_method='int',
                            use_incoherent_scattering=True)

    f_squared_mean        = calculate_f_squared_mean(composition, q)
    f_mean_squared        = calculate_f_mean_squared(composition, q)
    atomic_density        = convert_density_to_atoms_per_cubic_angstrom(composition, density)
    incoherent_scattering = calculate_incoherent_scattering(composition, q)

    normalization_factor  = calculate_normalization_factor_raw(sample_pattern,
                                                              atomic_density,
                                                              f_squared_mean,
                                                              f_mean_squared,
                                                              incoherent_scattering,
                                                              attenuation_factor)

    Fr = calculate_fr(sq_pattern, r=r, use_modification_fcn=use_modification_fcn, method='integral')
    r, fr = Fr.data
    
    intensity_coherent_correction = calc_dIcoh_dE(composition, fr, r, q, Emax) * normalization_factor
    intensity_corrected  = intensity - delE * intensity_coherent_correction

    sample_pattern = Pattern(q, intensity_corrected)