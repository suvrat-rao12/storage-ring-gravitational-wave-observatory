""" Simulating the detection of millihertz (mHz) gravitational waves (GWs)
    from astrophysical sources by a Storage Ring Gravitational-wave Observatory (SRGO).
    Authors: Suvrat Rao, Hamburg Observatory, University of Hamburg 
             Julia Baumgarten, Physics department, Jacobs University Bremen """

import numpy as np
from matplotlib import ticker as mpl_ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from scipy.spatial.transform import Rotation as R
from scipy.signal import spectrogram
# from scipy.signal import wiener
from scipy.integrate import simps
from scipy.interpolate import interp1d
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import pymc3 as pm
import arviz as az
import theano.tensor as tt
# from theano import config
# import os

pi = np.pi
exp = np.e
sqrt = np.sqrt
sin = np.sin
cos = np.cos
array = np.array
arange = np.arange
ones = np.ones
fft = np.fft.fft
fftfreq = np.fft.fftfreq
concatenate = np.concatenate
floatX = 'float32'  # config.floatX
# os.environ["MKL_NUM_THREADS"] = "5"

# Constants in SI units:
G = 6.67430 * 10 ** (-11)  # gravitational constant
c = 299792458.0  # speed of light
pc = 3.085677581 * 10 ** 16  # parsec
au = 1.495978707 * 10 ** 11  # astronomical unit
msol = 1.98847 * 10 ** 30  # solar mass
day = 86164.0905  # sidereal Earth day
radian = pi / 180.0  # degree to radian
we = 7.2921150 * 10 ** (-5)  # angular speed of Earth's rotation in radian/s.
H0 = 73.0*(1000/(10**6 * pc))  # Hubble constant
W0 = 0.6911  # LCDM universe dark energy density parameter
cosmo = FlatLambdaCDM(H0=73 * u.km / u.s / u.Mpc,
                      Tcmb0=2.725 * u.K, Om0=0.3089)


# ---------------------------------------------------------------------------------------------------------
# "DASHBOARD" for user input parameters:
"""Cannot use input() function since it gives EOFError when MCMC scans the code for global variables."""
# GW SOURCE PARAMETERS:
M1 = 10.0  # mass of object #1 in solar masses
M2 = 10.0  # mass of object #2 in solar masses
r = 0.007  # initial separation between bodies in au
# GW SOURCE-OBSERVER PARAMETERS:
inc = 0.0  # inclination angle of observer w.r.t binary system in degrees
Z = 4*10**-7  # redshift of the binary system
phase = 0.0  # GW signal initial phase in degrees
# local sidereal time of SRGO in hh:mm:ss at the start of the observation run
LST = (00.0, 00.0, 00.0)
RA = (00.0, 00.0, 00.0)  # GW source right ascension in hh:mm:ss
dec = 0.0  # GW source declination in degrees
# GW polarization angle (in degrees) as measured in equatorial celestial coordinates
psi_eq = 0.0
# STORAGE RING PARAMETERS:
v0 = 0.7*c  # 0.999999991*c  # SRGO ion bunch speed
L = 100.0  # 26659.0  # SRGO circumference in meters
n_p = 1  # 2*2808  # SRGO number of bunches
# MCMC PARAMETERS AND FLAGS:
# input("Compute sky localization with MCMC using synthetic data? ([y]/n)\n")
flag_mcmc = "y"
T_obs = 1.0 * day  # duration of SRGO observation run
N = 16  # no. of SRGO data points acquired during T_obs
psnr = 100  # peak signal to noise ratio
# "ligo" (Whittle likelihood in Fourier space) or "custom" likelihood functions for MCMC
likelihood_type = "ligo"
flag_show_pts = "y"  # show synthetic noisy data-points in signal plot ([y]/n)
# input("Plot signal spectrogram & evolution of orientation angles? ([y]/n)\n")
flag_extras = "y"
flag_sensi = "n"  # find SRGO sensitivity curve ([y]/n)
flag_err = "n"  # find numerical integration error ([y]/n)
flag_range = "y"  # find observational range of SRGO ([y]/n)
# ---------------------------------------------------------------------------------------------------------


# GW waveform:
""" Dominant harmonic of inspiral phase GWs from non-spinning binaries, modelled via post-Newtonian theory.
    Approximation of ISCO (Innermost Stable Circular Orbit) used for the end of the inspiral phase. """
m1 = M1*msol
m2 = M2*msol
M = (1 + Z) * ((m1 * m2) ** (3.0 / 5.0)) / \
    ((m1 + m2) ** (1.0 / 5.0))  # chirp mass
# final separation between bodies @ ISCO
Rf = 6.0 * G * max(m1, m2) / c ** 2.0
f0 = 2.0*sqrt(G * (m1 + m2)) / (2.0 * pi * (r*au) **
                                (3.0 / 2.0))  # initial GW frequency
# final GW frequency at the end of inspiral phase
f1 = 2.0*sqrt(G * (m1 + m2)) / (2.0 * pi * Rf ** (3.0 / 2.0))
k = (96.0 / 5.0) * (pi ** (8.0 / 3.0)) * \
    ((G * M / c ** 3.0) ** (5.0 / 3.0))
t_ins = (
    (3.0 / 8.0) * (1.0 / k) * (f0 ** (-8.0 / 3.0) - f1 ** (-8.0 / 3.0))
)   # inspiral time
# t_mrg = ( 5./256. )*( c**5/G**3 )*( r**4/((m1*m2)*(m1+m2)) )  #time to merger
t_ins_plot = t_ins
# GW source luminosity distance approximation
# https://iopscience.iop.org/article/10.1086/313167
#d = (c/H0)*(Z + (1.0 - 3.0*W0/4.0)*Z**2.0 + (9.0*W0-10.0)*(W0/8.0)*Z**3.0)
d = 10**6 * pc * cosmo.luminosity_distance(Z).value


def f(t, f0, k, Z):
    """GW frequency."""
    # Theano tensor variable and numpy object compatibility:
    if flag_mcmc == "y" and flag_sensi != "y" and flag_err != "y" and flag_range != "y" and str(type(t)) == "<class 'numpy.ndarray'>":
        return (array([
            (1.0 / (1.0 + Z)) * (f0 ** (-8.0 / 3.0) -
                                 (8.0 / 3.0) * k * x) ** (-3.0 / 8.0)
            for x in t])).astype(floatX)
    else:
        return (
            (1.0 / (1.0 + Z)) * (f0 ** (-8.0 / 3.0) -
                                 (8.0 / 3.0) * k * t) ** (-3.0 / 8.0)
        ).astype(floatX)
    # return ((1./(1.+Z))* f0).astype(floatX)  #constant GW frequency => continuous wave case


def h_plus(t, inc, phase, d, f0, k, Z):
    """GW plus-polarization."""
    inc *= radian
    phase *= radian
    # Theano tensor variable and numpy object compatibility:
    if flag_mcmc == "y" and flag_sensi != "y" and flag_err != "y" and flag_range != "y" and str(type(t)) == "<class 'numpy.ndarray'>":
        return (array([
            4.0 / d
            * (G * M / c ** 2) ** (5.0 / 3.0)
            * ((pi * f(x, f0, k, Z) / c) ** (2.0 / 3.0))
            * ((1.0 + cos(inc) ** 2.0) / 2.0)
            * cos(2 * pi * f(x, f0, k, Z) * x + phase)
            for x in t]))
    else:
        return (
            4.0 / d
            * (G * M / c ** 2) ** (5.0 / 3.0)
            * ((pi * f(t, f0, k, Z) / c) ** (2.0 / 3.0)).astype(floatX)
            * ((1.0 + cos(inc) ** 2.0) / 2.0).astype(floatX)
            * cos(2 * pi * f(t, f0, k, Z) * t + phase).astype(floatX)
        ).astype(floatX)


def h_cross(t, inc, phase, d, f0, k, Z):
    """GW cross-polarization."""
    inc *= radian
    phase *= radian
    # Theano tensor variable and numpy object compatibility:
    if flag_mcmc == "y" and flag_sensi != "y" and flag_err != "y" and flag_range != "y" and str(type(t)) == "<class 'numpy.ndarray'>":
        return (array([
            4.0 / d
            * (G * M / c ** 2) ** (5.0 / 3.0)
            * ((pi * f(x, f0, k, Z) / c) ** (2.0 / 3.0))
            * cos(inc)
            * sin(2 * pi * f(x, f0, k, Z) * x + phase)
            for x in t]))
    else:
        return (
            4.0 / d
            * (G * M / c ** 2) ** (5.0 / 3.0)
            * ((pi * f(t, f0, k, Z) / c) ** (2.0 / 3.0)).astype(floatX)
            * cos(inc).astype(floatX)
            * sin(2 * pi * f(t, f0, k, Z) * t + phase).astype(floatX)
        ).astype(floatX)


def findPrevPowerOf2(n):
    """"Round off to nearest power of 2, smaller than n."""
    k = 1
    while k < n:
        k = k << 1
    if k == n:
        return k
    else:
        return int(k/2)


# SRGO bunch clock least count (time between two consecutive ion arrivals)
T_sample = (L / v0) * (1.0 / n_p)


def time(n, t_obs, t_ins, f0, k, Z):
    """"Return the corrected timing data points allowed by the experiment measurement."""
    # observation period cannot exceed inspiral period in our code
    t_obs = min(t_obs, t_ins)
    n = min(n, int(t_obs / T_sample))  # allowed N
    # Nyquist–Shannon sampling theorem
    n = max(n, int(2.0*f(t_obs, f0, k, Z)*t_obs)+1)
    if flag_mcmc == "y" and flag_sensi != "y" and flag_err != "y" and flag_range != "y":
        # to be able to use FFT trick for computing speed
        n = findPrevPowerOf2(n)
        if n < int(2.0*f(t_obs, f0, k, Z)*t_obs)+1:
            n = 2*n
    # allowed timing interval
    T_timing = int(t_obs / (n * T_sample)) * T_sample
    t = arange(0.0, t_obs, T_timing, dtype=floatX)  # timing data points
    if flag_mcmc == "y" and flag_sensi != "y" and flag_err != "y" and flag_range != "y" and len(t) > n:
        # to be able to use FFT trick for computing speed
        pts = len(t) - n
        t = t[:-pts]
    return t, T_timing


# Effect of Earth's rotation:
# angular position of the detector on the SRGO ring, in degrees converted to radians
# (value 0 => detector is placed along the longitude passing via the center of SRGO, south of the center)
phi0 = 0.0 * radian
# SRGO latitude in degrees converted to radians
# if theta_lat == dec, sometimes there is a numerical error in calculating the angle from the rotation matrix, so we define theta_lat += 0.001
theta_lat = 51.001 * radian
ra = (
    (RA[0] * 3600.0 + RA[1] * 60.0 + RA[2]) * 360.0/(24.0*60.0*60.0)
)  # GW source right ascension in seconds converted to degrees
# initial local sidereal time in seconds converted to degrees
# (lst==ra => observation run begins when the GW source is on the celestial meridian of SRGO)
lst = (LST[0] * 3600.0 + LST[1] * 60.0 + LST[2]) * 360.0/(24.0*60.0*60.0)
dec = dec  # GW source declination in degrees
# GW polarization angle (in degrees) in equatorial celestial coordinates.
psi_eq = psi_eq

matrix_phi0 = array(
    [
        [cos(phi0), -sin(phi0), 0.0],
        [sin(phi0), cos(phi0), 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=floatX,
)

matrix_theta_lat = array(
    [
        [sin(theta_lat), 0.0, -cos(theta_lat)],
        [0.0, 1.0, 0.0],
        [cos(theta_lat), 0.0, sin(theta_lat)],
    ],
    dtype=floatX,
)

matmul = np.matmul
matmul_matrix_theta_lat_matrix_phi0 = matmul(
    matrix_theta_lat, matrix_phi0, dtype=floatX
)
tt_arctan2 = tt.arctan2
tt_arccos = tt.arccos
np_arctan2 = np.arctan2
np_arccos = np.arccos


def earth_rot(t, ra, lst, dec, psi_eq, flag):
    """Time-evolution of Euler angles due to effect of Earth's rotation."""
    ra *= radian
    lst *= radian
    dec *= radian
    psi_eq *= radian

    matrix_dec = array(
        [
            [-sin(dec), 0.0, cos(dec)],
            [0.0, 1.0, 0.0],
            [-cos(dec), 0.0, -sin(dec)],
        ]
    )

    matrix_psi_eq = array(
        [
            [cos(psi_eq), -sin(psi_eq), 0.0],
            [sin(psi_eq), cos(psi_eq), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    def mp_func():  # generator function, which can also be made into a multiprocessing function
        for i in range(len(t)):
            # print(i)
            matrix_ra_time = array(
                [
                    [
                        -cos(ra - lst - we * t[i]),
                        sin(ra - lst - we * t[i]),
                        0.0,
                    ],
                    [
                        -sin(ra - lst - we * t[i]),
                        -cos(ra - lst - we * t[i]),
                        0.0,
                    ],
                    [0.0, 0.0, 1.0],
                ]
            )

            rhs = matmul(matrix_psi_eq,
                         matmul(matrix_dec,
                                matmul(matrix_ra_time,
                                       matmul_matrix_theta_lat_matrix_phi0
                                       ),
                                ),
                         )

            if flag == "mcmc":
                yield (
                    tt_arctan2(rhs[2][1], -rhs[2][0]),
                    tt_arccos(rhs[2][2]),
                    tt_arctan2(rhs[1][2], rhs[0][2]),
                )
            else:
                yield (
                    np_arctan2(rhs[2][1], -rhs[2][0], dtype=floatX),
                    np_arccos(rhs[2][2], dtype=floatX),
                    np_arctan2(rhs[1][2], rhs[0][2], dtype=floatX),
                )

    return array(list(zip(*mp_func())))


# SRGO response (without noise):
def F_plus(phi, theta, psi):
    """SRGO antenna pattern for GW plus-polarization."""
    return cos(2.0 * psi) * sin(theta) ** 2.0


def F_cross(phi, theta, psi):
    """SRGO antenna pattern for GW cross-polarization."""
    return sin(2.0 * psi) * sin(theta) ** 2.0


def h_eff(t, phi, theta, psi, inc, phase, d, f0, k, Z):
    """Effective longitudinal GW strain for SRGO test masses
    that are circulating rapidly compared to the GW frequency."""
    return (
        -(1.0 / 2.0)
        * (
            F_plus(phi, theta, psi) * h_plus(t, inc, phase, d, f0, k, Z)
            + F_cross(phi, theta, psi) * h_cross(t, inc, phase, d, f0, k, Z)
        )
    )


def boole_quad(y, x, yofx, a, b, c, d, e, g, k, l, p):
    """Custom implementation of Boole's rule quadrature."""
    h = (x[1] - x[0]) / 4.0
    a1 = a[0] + h * (a[1] - a[0]) / (4.0 * h)
    a2 = a[0] + 2.0 * h * (a[1] - a[0]) / (4.0 * h)
    a3 = a[0] + 3.0 * h * (a[1] - a[0]) / (4.0 * h)
    b1 = b[0] + h * (b[1] - b[0]) / (4.0 * h)
    b2 = b[0] + 2.0 * h * (b[1] - b[0]) / (4.0 * h)
    b3 = b[0] + 3.0 * h * (b[1] - b[0]) / (4.0 * h)
    c1 = c[0] + h * (c[1] - c[0]) / (4.0 * h)
    c2 = c[0] + 2.0 * h * (c[1] - c[0]) / (4.0 * h)
    c3 = c[0] + 3.0 * h * (c[1] - c[0]) / (4.0 * h)

    return (2.0 * h / 45.0) * (
        7.0 * y[0]
        + 32.0 * yofx(x[0] + h, a1, b1, c1, d, e, g, k, l, p)
        + 12.0 * yofx(x[0] + 2.0 * h, a2, b2, c2, d, e, g, k, l, p)
        + 32.0 * yofx(x[0] + 3.0 * h, a3, b3, c3, d, e, g, k, l, p)
        + 7.0 * y[1]
    )


def SRGO_signal(t, phi, theta, psi, inc, phase, d, f0, k, Z):
    """SRGO signal from response to GWs."""
    integrand = h_eff(t, phi, theta, psi, inc, phase, d, f0, k, Z)

    signal = [0.0]
    append = signal.append  # functional programming optimization

    gen = (                 # generator comprehension
        append(
            signal[-1]
            + (1.0 - v0 ** 2.0 / (2.0 * c ** 2.0))
            * boole_quad(
                integrand[i: i+2],
                t[i: i+2],
                h_eff,
                phi[i: i+2],
                theta[i: i+2],
                psi[i: i+2],
                inc,
                phase,
                d,
                f0,
                k,
                Z
            )
        )
        for i in range(len(t) - 1)
    )

    dummy = list(gen)

    return array(signal)


# Some quantities for plotting:
t_plt, dt_plt = time(20000, 3.0*day, t_ins, f0, k, Z)
phi_plt, theta_plt, psi_plt = earth_rot(t_plt, ra, lst, dec, psi_eq, "regular")
signal_plt = SRGO_signal(t_plt, phi_plt, theta_plt,
                         psi_plt, inc, phase, d, f0, k, Z)


def error(p):
    """For obtaining the numerical error of integration given the no. of data pts, 'p'."""
    t_err, dt_err = time(p, 1.0*day, t_ins, f0, k, Z)
    phi_err, theta_err, psi_err = earth_rot(
        t_err, ra, lst, dec, psi_eq, "regular")
    signal_err = SRGO_signal(t_err, phi_err, theta_err,
                             psi_err, inc, phase, d, f0, k, Z)

    err = max(abs(signal_err - signal_ref(t_err)))
    print(p, err)
    return(err)


if flag_err == "y":
    t_ref, dt_ref = time(10**5, 3.0*day, t_ins, f0, k, Z)  # increase to 10**7
    phi_ref, theta_ref, psi_ref = earth_rot(
        t_ref, ra, lst, dec, psi_eq, "regular")
    signal_ref = SRGO_signal(t_ref, phi_ref, theta_ref,
                             psi_ref, inc, phase, d, f0, k, Z)
    signal_ref = interp1d(t_ref, signal_ref)
    pts = 2.**arange(1, 15, 1)  # increase to 20
    errs = [error(p) for p in pts]
    sns.reset_orig()
    fig2, axes1 = plt.subplots(1, 1)
    f_samprates = pts/day
    axes1 = plt.gca()
    axes2 = axes1.twiny()
    axes1.loglog(pts, errs, color='blue', linewidth=1.25)
    axes1.set_xlabel("No. of Data Points", fontweight='bold')
    axes1.set_ylabel(
        "Max. Numerical Integration Error [$\mathbf{s}$]", fontweight='bold')
    axes1.legend(["$T_{obs} = 1.0$ day(s)"])
    axes2.loglog(f_samprates, errs, color='blue', linewidth=0.0)
    axes2.set_xlabel(
        "Sampling Frequency [$\mathbf{s^{-1}}$]", fontweight='bold')
    fig2.savefig("saved_plots/numerical_error.png", format="png", dpi=fig2.dpi)
    # plt.close(fig2)


def srgo_range(dee, zed):
    """For numerically obtaining the SRGO observational range."""
    M = (1 + zed) * ((m1 * m2) ** (3.0 / 5.0)) / \
        ((m1 + m2) ** (1.0 / 5.0))  # chirp mass
    # final separation between bodies @ ISCO
    Rf = 6.0 * G * max(m1, m2) / c ** 2.0
    f0 = 0.00001  # initial GW frequency
    # final GW frequency at the end of inspiral phase
    f1 = 2.0*sqrt(G * (m1 + m2)) / (2.0 * pi * Rf ** (3.0 / 2.0))
    k = (96.0 / 5.0) * (pi ** (8.0 / 3.0)) * \
        ((G * M / c ** 3.0) ** (5.0 / 3.0))
    t_ins = (
        (3.0 / 8.0) * (1.0 / k) * (f0 ** (-8.0 / 3.0) - f1 ** (-8.0 / 3.0))
    )  # inspiral time
    inc = 0.0  # inclination angle
    phase = 0.0  # initial GW phase

    t_range, dt_range = time(20000, 3.0*day, t_ins, f0, k, zed)
    phi_range, theta_range, psi_range = earth_rot(
        t_range, ra, lst, dec, psi_eq, "regular")
    signal_range = SRGO_signal(
        t_range, phi_range, theta_range, psi_range, inc, phase, dee, f0, k, zed)

    peak_signal = max(abs(max(signal_range)), abs(min(signal_range)))
    print(zed, peak_signal)
    return(peak_signal)


if flag_range == "y":
    ra_vals = np.linspace(-45.0, 90.0, 4)
    dec_vals = np.linspace(-45.0, 90.0, 4)
    psi_eq_vals = np.linspace(-45.0, 90.0, 4)

    l = 150
    # up to redshift of first stars (Cosmic Dawn)
    z_array1 = np.linspace(0.0, 15.0, 100000)
    d_array1 = 10**6 * pc * cosmo.luminosity_distance(z_array1).value
    z_of_d = interp1d(d_array1, z_array1)
    # from closest mHz spectroscopic binaries up to Cosmic Dawn distance
    d_array = np.linspace(10*pc, d_array1[-1], l)
    z_array = z_of_d(d_array)
    sig_array = np.zeros(l)

    for aaa in ra_vals:
        ra = aaa
        for bbb in dec_vals:
            dec = bbb
            for ccc in psi_eq_vals:
                psi_eq = ccc
                for i in range(l):
                    if z_array[i] > 10:  # first mHz SMBBHs due to galaxy mergers
                        # https://arxiv.org/abs/1903.06867
                        m1 = 10**5 * msol
                        m2 = m1  # equal mass binaries
                    elif z_array[i] <= 10 and d_array[i] >= 27.4*pc*10**6:
                        # closest SMBBH (NGC 7727)
                        # https://doi.org/10.1051/0004-6361/202140827
                        m1 = 10**7 * msol
                        m2 = m1
                    elif d_array[i] < 27.4*pc*10**6 and d_array[i] >= 777847.7361*pc:
                        # closest galaxy (Andromeda)
                        # https://iopscience.iop.org/article/10.3847/1538-4357/ac339f
                        m1 = 10**5 * msol
                        m2 = m1
                    elif d_array[i] < 777847.7361*pc and d_array[i] >= 7665.0348*pc:
                        # closest dwarf galaxy (Canis Major)
                        # https://academic.oup.com/mnras/article/473/1/1186/4060726
                        m1 = 10**2 * msol
                        m2 = m1
                    elif d_array[i] < 7665.0348*pc and d_array[i] >= 1580*pc:
                        # closest black hole within our galaxy (Milky Way)
                        # https://arxiv.org/abs/2201.13296
                        m1 = 10 * msol
                        m2 = m1
                    elif d_array[i] < 1580*pc and d_array[i] >= 267*pc:
                        # closest mHz binary neutron star within our galaxy
                        # https://iopscience.iop.org/article/10.1088/0004-637X/789/2/119
                        m1 = 1 * msol
                        m2 = m1
                    else:
                        # closest mHz spectroscopic binaries within Milky Way:
                        # https://en.wikipedia.org/wiki/Gliese_829
                        # https://en.wikipedia.org/wiki/GJ_3991
                        # https://en.wikipedia.org/wiki/Delta_Trianguli
                        # https://en.wikipedia.org/wiki/HR_5553
                        # https://iopscience.iop.org/article/10.1088/0004-6256/147/6/129
                        m1 = 0.5 * msol
                        m2 = m1

                    sig_array[i] = max(
                        sig_array[i], srgo_range(d_array[i], z_array[i]))

    sns.reset_orig()
    fig3, axes3 = plt.subplots(1, 1)
    axes3 = plt.gca()
    axes4 = axes3.twiny()  # the spines of the second axes are overdrawn on the first
    facecolour = "black"
    axes3.set_facecolor(facecolour)
    fig3.set_facecolor(facecolour)
    textcolour = "white"
    axes4.spines['top'].set_color(textcolour)
    axes4.spines['bottom'].set_color(textcolour)
    axes4.spines['left'].set_color(textcolour)
    axes4.spines['right'].set_color(textcolour)
    axes3.xaxis.label.set_color(textcolour)
    axes3.yaxis.label.set_color(textcolour)
    axes4.xaxis.label.set_color(textcolour)
    axes3.tick_params(axis='x', colors=textcolour)
    axes3.tick_params(axis='y', colors=textcolour)
    axes4.tick_params(axis='x', colors=textcolour)
    # axes3.plot(d_array1/pc, d_array1*0.0 + 10.0,
    #           linewidth=0.95, linestyle='dotted', color="darkgrey")
    axes3.plot(d_array1/pc, d_array1*0.0 + 1.0,
               linewidth=0.95, linestyle='dotted', color="darkgrey")
    # axes3.plot(d_array1/pc, d_array1*0.0 + 0.1,
    #           linewidth=0.95, linestyle='dotted', color="darkgrey")
    txtsize = axes3.xaxis.label.get_size()
    axes3.plot(d_array/pc, sig_array/(10**-16), linewidth=1.6, color="orange")
    axes3.text(10**4, 6*10**6,
               "$\sigma_{noise} = 0.1fs$", size=4*txtsize/5, color=textcolour)
    axes3.plot(d_array/pc, sig_array/(10**-15), linewidth=1.6, color="orange")
    axes3.text(10**4, 10**4, "$\sigma_{noise} = 1fs$",
               size=4*txtsize/5, color=textcolour)
    axes3.plot(d_array/pc, sig_array/(10**-12), linewidth=1.6, color="orange")
    axes3.text(10**4, 10**1, "$\sigma_{noise} = 1ps$",
               size=4*txtsize/5, color=textcolour)
    axes3.plot(d_array/pc, sig_array/(10**-9), linewidth=1.6, color="orange")
    axes3.text(10**4, 10**-2,
               "$\sigma_{noise} = 1ns$", size=4*txtsize/5, color=textcolour)
    axes3.axvline(x=10, linewidth=0.95, linestyle='dashdot', color="blueviolet",
                  label="nearest ~$0.5M_{\odot}$ WD binaries")
    axes3.axvline(x=267, linewidth=0.95, linestyle='dashdot', color="steelblue",
                  label="nearest ~$1M_{\odot}$ NS binaries")
    axes3.axvline(x=1580, linewidth=0.95, linestyle='dashdot', color="cadetblue",
                  label="nearest ~$10M_{\odot}$ BH binaries")
    axes3.axvline(x=7665.0348, linewidth=0.95, linestyle='dashdot', color="seagreen",
                  label="nearest ~$10^2M_{\odot}$ IMBH binaries")
    axes3.axvline(x=777847.7361, linewidth=0.95, linestyle='dashdot', color="salmon",
                  label="nearest ~$10^5M_{\odot}$ SMBH binaries")
    axes3.axvline(x=27.4*10**6, linewidth=0.95, linestyle='dashdot', color="deeppink",
                  label="nearest ~$10^7M_{\odot}$ SMBH binaries")
    axes3.axvline(x=cosmo.luminosity_distance(10.0).value*10**6, linewidth=0.95, color="red",
                  linestyle='dashdot', label="first ~$10^7M_{\odot}$ SMBH binaries")
    axes3.set_xlabel("GW source distance [pc]")
    axes3.set_ylabel("Max. PSNR (peak signal-to-noise ratio)")
    axes3.set_xscale('log')
    axes3.set_yscale('log')
    axes3.set_ylim(0.5*min(sig_array)/(10**-9), 2.0*max(sig_array)/(10**-16))
    axes3.set_xticks([1, 10, 10**2, 10**3, 10**4, 10**5, 10 **
                     6, 10**7, 10**8, 10**9, 10**10, 10**11])
    axes3.set_yticks([10**-8, 10**-7, 10**-5, 10**-3, 10**-1,
                     10**0, 10**1, 10**3, 10**5, 10**7, 10**9, 10**10])
    leg = axes3.legend(loc='upper right', bbox_to_anchor=(
        0.935, 1.0), facecolor=facecolour, framealpha=1, prop={'size': 5.75})
    for text in leg.get_texts():
        text.set_color(textcolour)
    axes4.plot(z_array, sig_array, color='orange', linewidth=0.0)
    axes4.set_xlabel("GW source redshift (z)")
    axes4.set_xscale('log')
    axes4.set_xticks([10**-10, 10**-9, 10**-8, 10**-7, 10**-6,
                     10**-5, 10**-4, 10**-3, 10**-2, 10**-1, 10**0, 10**1])
    fig3.canvas.manager.full_screen_toggle()
    fig3.tight_layout()
    fig3.savefig("saved_plots/srgo_range.png", format="png",
                 dpi=1200, facecolor=fig3.get_facecolor())
    # plt.close(fig3)


def sensi_curve(b):
    """For numerically obtaining the SRGO sensitivity curve."""
    sigma_noise = 10**(-12)

    M = (1 + Z) * ((m1 * m2) ** (3.0 / 5.0)) / \
        ((m1 + m2) ** (1.0 / 5.0))  # chirp mass
    # final separation between bodies @ ISCO
    Rf = 6.0 * G * max(m1, m2) / c ** 2.0
    f0 = 2.0*sqrt(G * (m1 + m2)) / (2.0 * pi * (r*au) **
                                    (3.0 / 2.0))  # initial GW frequency
    # final GW frequency at the end of inspiral phase
    f1 = 2.0*sqrt(G * (m1 + m2)) / (2.0 * pi * Rf ** (3.0 / 2.0))
    k = (96.0 / 5.0) * (pi ** (8.0 / 3.0)) * \
        ((G * M / c ** 3.0) ** (5.0 / 3.0))
    t_ins = (
        (3.0 / 8.0) * (1.0 / k) * (f0 ** (-8.0 / 3.0) - f1 ** (-8.0 / 3.0))
    )   # inspiral time

    t_sensi, dt_sensi = time(20000, 3.0*day, t_ins, f0, k, Z)
    phi_sensi, theta_sensi, psi_sensi = earth_rot(
        t_sensi, ra, lst, dec, psi_eq, "regular")
    signal_sensi = SRGO_signal(
        t_sensi, phi_sensi, theta_sensi, psi_sensi, inc, phase, b, f0, k, Z)

    val_lhs = (simps(signal_sensi**2.0, x=t_sensi) / t_sensi[-1])**0.5
    val_rhs = 2.0 * sigma_noise / ((1.0/T_sample)*t_sensi[-1])**0.5
    #global d
    d = b*val_lhs/val_rhs
    return(f(0.0, f0, k, Z), abs(h_plus(0.0, inc, phase, d, f0, k, Z)))


if flag_sensi == "y":
    M1_vals = np.logspace(1.0, 6.0, 3)
    #M2_vals = np.logspace(1.0, 6.0, 6)
    ra_vals = np.linspace(-45.0, 90.0, 4)
    dec_vals = np.linspace(-45.0, 90.0, 4)
    psi_eq_vals = np.linspace(-45.0, 90.0, 4)
    inc_vals = np.linspace(0.0, 180.0, 1)
    phase_vals = np.linspace(0.0, 360.0, 1)

    l = 20
    f0_array = np.zeros(l)
    h0_array = np.zeros(l)
    h0_temp = np.zeros(l)

    for aaa in ra_vals:
        ra = aaa
        for bbb in dec_vals:
            dec = bbb
            for ccc in psi_eq_vals:
                psi_eq = ccc
                for ddd in inc_vals:
                    inc = ddd
                    for eee in phase_vals:
                        phase = eee
                        for fff in M1_vals:
                            m1 = fff * msol
                            m2 = m1  # equal mass binaries
                            separation = (1.0/au)*(2.0*sqrt(G * (m1 + m2)) / (2.0 * pi * (1.0+Z) *
                                                                              np.logspace(-5.0, -1.0, l)))**(2.0/3.0)
                            for i in range(l):
                                r = separation[i]
                                f_0, h_0 = sensi_curve(d)
                                print(f_0, h_0)
                                if f0_array[i] == 0.0:
                                    f0_array[i] = f_0
                                if np.isnan(h_0):
                                    h_0 = 0.0
                                h0_temp[i] = h_0

                            h0_array += h0_temp

    h0_array *= 1.0/(len(ra_vals)*len(dec_vals) *
                     len(psi_eq_vals)*len(inc_vals)*len(phase_vals)*len(M1_vals))

    # To simulate graphic
    fq = 10**np.linspace(-11, 6, 1000000)
    i = int(np.where(np.abs(fq-10**(-5.0)) ==
            np.min(np.abs(fq-10**(-5.0))))[0])
    j = int(np.where(np.abs(fq-10**(-1.0)) ==
            np.min(np.abs(fq-10**(-1.0))))[0])
    i1 = int(np.where(np.abs(fq-10**(-7.0)) ==
             np.min(np.abs(fq-10**(-7.0))))[0])
    j1 = int(np.where(np.abs(fq-10**(1.0)) == np.min(np.abs(fq-10**(1.0))))[0])

    t_noise = 10.0**(-12)
    f_sampling = 2.0*2808*11245

    h = (16.0*np.pi/np.sqrt(30.0))*(t_noise/np.sqrt(f_sampling))*fq**(3.0/2.0)

    for k in range(len(fq)):
        if (k < i and k != i1) or (k > j and k != j1):
            h[k] = np.nan

    fm = np.array([3.33*10**(-4), 4.66*10**(-1)])
    hm = np.array([3.0*10**(-18), 2.0*10**(-20)])

    fe = np.array([10**(-4), 10**(-1)])
    he = np.array([8.0*10**(-19), 8.0*10**(-21)])

    fr = np.array([5.33*10**(-4), 10**(-2)])
    hr = np.array([10**(-19), 10**(-19)])

    fu = np.array([8.66*10**(-5), 2.0*10**(-3)])
    hu = np.array([3.0*10**(-20), 8.0*10**(-21)])

    sns.reset_orig()
    plt.figure()
    plt.fill_between(fm, hm, facecolor='steelblue',
                     alpha=0.6, label="Massive binaries")
    plt.fill_between(fe, he, facecolor='darkcyan', alpha=0.6, label="EMRIs")
    plt.fill_between(fr, hr, facecolor='lightskyblue', alpha=0.6,
                     label="Resolvable galactic binaries")
    plt.fill_between(fu, hu, facecolor='lime', alpha=0.6,
                     label="Unresolvable galactic binaries")
    plt.loglog(fq, h, basex=10, basey=10, color='black', linewidth=1.5,
               label="LHC-GW sensitivity curve estimation from paper-I")
    plt.loglog(f0_array, h0_array, basex=10, basey=10, color='black', linestyle='dashed',
               linewidth=1.5, label="Numerically obtained sensitivity curve")
    plt.legend(loc=2, borderpad=0.2, labelspacing=0.2)
    plt.ylim(10**(-26.), 10**(-12.))
    plt.ylabel("Characteristic Strain", fontweight='bold')
    plt.xlabel("Frequency / Hz", fontweight='bold')
    plt.grid()
    plt.tick_params(axis='x', which='minor')
    plt.tight_layout()
    plt.savefig("saved_plots/srgo_sensitivity.png", dpi=1200)
    # plt.close()


# Creating synthetic signal and adding only stochastic noise:
if flag_sensi != "y" and flag_err != "y" and flag_range != "y":

    t, dt = time(N, T_obs, t_ins, f0, k, Z)
    N = len(t)
    phi, theta, psi = earth_rot(t, ra, lst, dec, psi_eq, "regular")
    signal = SRGO_signal(t, phi, theta, psi, inc, phase, d, f0, k, Z)
    noise = np.random.normal(0.0, scale=max(
        abs(signal_plt)) / psnr, size=N).astype(floatX)
    noisy_signal = (signal + noise).astype(floatX)
    # noisy_signal = wiener(noisy_signal, noise=max(
    #   abs(signal_plt)) / psnr)  # Wiener noise filter
    # Noise filter introduces new correlations between the independent data-points.
    # Then we must account for the covariance matrix in the mcmc.
    # https://www.researchgate.net/post/Does_using_a_noise_filter_before_performing_MCMC_fitting_of_a_model_to_noisy_data_increase_or_decrease_the_accuracy_of_the_fit


# Simulating noisy data and computing sky localization using MCMC:
if flag_mcmc == "y" and flag_sensi != "y" and flag_err != "y" and flag_range != "y":

    # Applying bandpass filter based on predicted mHz astrophysical sources:
    if likelihood_type == "ligo":
        noisy_signal_f = fft(noisy_signal)
        freqs = fftfreq(N, dt)
        # Note: The below inequalities do the opposite of what they appear to do!
        # noisy_signal_f = np.where(
        #    abs(freqs) > 8.*10**(-5.), noisy_signal_f, 0.0)
        # noisy_signal_f = np.where(
        #    abs(freqs) < 5.*10**(-1.), noisy_signal_f, 0.0)

    # Power spectral density of the noise:
    psd = (dt**2.0/t[-1])*abs(fft(noise))**2.0

    def Whittle(model, data, sigma, df):
        """
        2-dimensional Whittle Likelihood (Whittle, 1951); see also Cornish & Romano (2013).
        https://doi.org/10.1017/pasa.2019.2
        """

        likelihood = (1.0/(2.0*pi*sigma)**0.5) * \
            exp**(-2.0*df*abs(model-data)**2.0/sigma)

        return likelihood

    def FFT(x):
        """
        A recursive implementation of the
        1D Cooley-Tukey Fast Fourier Transform.
        The input should have a length of a
        power of 2.
        """
        S = len(x)

        if S == 1:
            return x
        else:
            X_even = FFT(x[::2])
            X_odd = FFT(x[1::2])
            factor = \
                exp**(-2j*pi*arange(S) / S)

            X = concatenate(
                [X_even+factor[:int(S/2)]*X_odd,
                 X_even+factor[int(S/2):]*X_odd])
            return X

    MCMC = pm.Model()

    with MCMC:

        # Priors for unknown model parameters:
        M1_prior = M1  # pm.Uniform(  # pm.Bound(pm.Normal, lower=0.0, upper=360.0)(
        #    "GW source mass #1 posterior [$M_{\odot}$]", lower = 0.9*M1, upper = 1.1*M1
        # )
        M2_prior = M2  # pm.Uniform(  # pm.Bound(pm.Normal, lower=0.0, upper=360.0)(
        #    "GW source mass #2 posterior [$M_{\odot}$]", lower=0.9*M2, upper=1.1*M2
        # )
        r_prior = pm.Uniform(  # pm.Bound(pm.Normal, lower=0.0, upper=360.0)(
            "Initial binary separation posterior [$au$]", lower=0.9*r, upper=1.1*r
        )
        inc_prior = pm.Uniform(  # pm.Bound(pm.Normal, lower=0.0, upper=360.0)(
            "GW source inclination posterior [$\degree$]", lower=inc-180.0, upper=inc+180.0
        )
        Z_prior = Z  # pm.Uniform(  # pm.Bound(pm.Normal, lower=0.0, upper=360.0)(
        #    "GW source redshift posterior", lower=0.0, upper=1.0
        # )
        phase_prior = pm.Uniform(  # pm.Bound(pm.Normal, lower=0.0, upper=360.0)(
            "GW initial phase posterior [$\degree$]", lower=phase-180.0, upper=phase+180.0
        )
        ra_prior = pm.Uniform(  # pm.Bound(pm.Normal, lower=0.0, upper=360.0)(
            "GW source RA posterior [$\degree$]", lower=ra-180.0, upper=ra+180.0
        )
        dec_prior = pm.Uniform(  # pm.Bound(pm.Normal, lower=-90.0, upper=90.0)(
            "GW source DEC posterior [$\degree$]", lower=dec-180.0, upper=dec+180.0
        )
        psi_eq_prior = pm.Uniform(  # pm.Bound(pm.Normal, lower=0.0, upper=180.0)(
            "GW polarization posterior [$\degree$]", lower=psi_eq-180.0, upper=psi_eq+180.0
        )
        # Choose a "weakly-informative" prior instead of a flat prior.
        # https://discourse.pymc.io/t/improving-model-convergence-and-sampling-speed/279/2

        # Model:
        m1 = M1_prior*msol
        m2 = M2_prior*msol
        M = (1.0 + Z_prior) * ((m1 * m2) ** (3.0 / 5.0)) / \
            ((m1 + m2) ** (1.0 / 5.0))  # chirp mass
        # max(m1,m2) without using boolean operations for Theano compatibility
        # max_m1_m2 = (abs(m1 - m2) + m1 + m2)/2.0
        f0 = 2.0*sqrt(G * (m1 + m2)) / (2.0 * pi * (r_prior*au) **
                                        (3.0 / 2.0))  # initial GW frequency
        k = (96.0 / 5.0) * (pi ** (8.0 / 3.0)) * \
            ((G * M / c ** 3.0) ** (5.0 / 3.0))
        # luminosity distance approximation of the binary system
        # https://iopscience.iop.org/article/10.1086/313167
        d = (c/H0)*(Z_prior + (1.0 - 3.0*W0/4.0)*Z_prior **
                    2.0 + (9.0*W0-10.0)*(W0/8.0)*Z_prior**3.0)
        # d = 10**6 * pc * cosmo.luminosity_distance(Z_prior).value
        phim, thetam, psim = earth_rot(
            t, ra_prior, lst, dec_prior, psi_eq_prior, "mcmc"
        )
        print("\nBuilding model: Stage 1/2 complete!\n")
        if likelihood_type == "custom":
            mss = simps(signal_plt**2.0, x=t_plt) / t_plt[-1]
            if N > 4.0*(max(abs(signal_plt)))**2.0/(mss*psnr**2.0):
                denominator = min(
                    abs(sum(noisy_signal)),
                    3.0 * max(abs(signal)) / (psnr * N ** 0.5)
                )
            else:
                denominator = 3.0 * max(abs(signal)) / (psnr * N ** 0.5)
            model = abs(sum(
                noisy_signal -
                SRGO_signal(t, phim, thetam, psim, inc_prior,
                            phase_prior, d, f0, k, Z_prior)
            )) / denominator
            model = (
                model ** (model ** 0.001) - 1.0 +
                abs(model ** (model ** 0.001) - 1.0)
            ) * ones(N, dtype=floatX)
            model = model.reshape(N, 1).astype(floatX)
        elif likelihood_type == "ligo":
            model = SRGO_signal(t, phim, thetam, psim,
                                inc_prior, phase_prior, d, f0, k, Z_prior)
            # all elements must be of the same shape/type, else ValueError
            model[0] = (0.0*model[1]).astype(floatX)
            # https://discourse.pymc.io/t/memory-issues-with-creating-simple-regression-model/3411
            model = abs(FFT(model) - noisy_signal_f)*dt
            model = tt.stack(model, axis=0).astype(floatX)
        print("Building model: Stage 2/2 complete!\n")

        # Likelihood of observations:
        # MCMC accuracy is very sensitive to this !
        if likelihood_type == "custom":
            data = 0.0 * ones(N, dtype=floatX).reshape(N, 1)
            std_dev = 0.0000000001  # float32 limit is 6 decimals
            likelihood = pm.Normal(
                "Noisy signal in seconds",
                mu=model,
                sigma=std_dev,
                observed=data,)
            special_flag = True
        elif likelihood_type == "ligo":
            data = 0.0 * ones(N, dtype=floatX).reshape(N, 1)
            std_dev = psd.reshape(N, 1).astype(floatX)
            # http://www.add.ece.ufl.edu/4511/references/ImprovingFFTResoltuion.pdf
            df = 1.0/(N*dt)
            likelihood = pm.DensityDist("likelihood", Whittle,
                                        observed={'model': model, 'data': data, 'sigma': std_dev, 'df': df})
            special_flag = False

    if __name__ == "__main__":  # needed for MCMC to use multiple CPU cores

        with MCMC:
            # draw 'M' posterior samples
            M = 1000  # there are errors if M is defined elsewhere or with non-integer value
            # http://proceedings.mlr.press/v9/murray10a/murray10a.pdf
            # step = pm.step_methods.EllipticalSlice(prior_cov=np.identity(4))
            # https://docs.pymc.io/notebooks/DEMetropolisZ_EfficiencyComparison.html
            step = pm.step_methods.Metropolis(S=np.identity(2))
            print("Running MCMC...")
            # https://chi-feng.github.io/mcmc-demo/
            posterior = pm.sample(
                step=step,
                draws=M,
                # MCMC is working properly if true parameter values lie within 90% HPDI, 90% of the time.
                chains=100,
                cores=1,
                #start={'m1_prior': m1},
                tune=0,  # burn-in samples
                discard_tuned_samples=True,
                progressbar=True,
                return_inferencedata=True,
                # https://github.com/pymc-devs/pymc3/issues/4002
                idata_kwargs={"density_dist_obs": special_flag},
            )
            # https://colcarroll.github.io/hmc_tuning_talk/
            # https://discourse.pymc.io/t/nuts-uses-all-cores/909
            # https://stackoverflow.com/questions/56442977/where-should-i-insert-os-environmkl-num-threads-1
            # https://discourse.pymc.io/t/regarding-the-use-of-multiple-cores/4249
            # https://discourse.pymc.io/t/weird-error-the-kernel-appears-to-have-died-it-will-restart-automatically/7045
            az.style.use("arviz-darkgrid")

            # MCMC trace plot:
            az.plot_trace(
                posterior,
                kind="trace",
                legend=False,
                rug=True,
                compact=True,
                combined=True,
            )

            # RA-DEC joint posterior:
            ax_az = az.plot_pair(
                posterior,
                var_names=[
                    "GW source RA posterior [$\degree$]",
                    "GW source DEC posterior [$\degree$]",
                ],
                kind="kde",
                marginals=True,
                figsize=(9.25, 9.25),
                kde_kwargs={
                    # Plot HPDI contours (highest posterior density interval)
                    # for (x)% contour, the hdi_probs argument is (100 - x)%
                    "hdi_probs": [0.1],
                    "contour": True,
                    "contourf_kwargs": {"cmap": "spring"},
                    "fill_last": False,
                    "label": "90%% HPDI contour",
                    "legend": True,
                },
            )

            # Plotting sky-localization in Mollweide projection:
            ax_skyloc = az.plot_pair(
                posterior,
                var_names=[
                    "GW source RA posterior [$\degree$]",
                    "GW source DEC posterior [$\degree$]",
                ],
                kind="kde",
                figsize=(9.25, 9.25),
                kde_kwargs={
                    # Plot HPDI contours (highest posterior density interval)
                    # for (x)% contour, the hdi_probs argument is (100 - x)%
                    "hdi_probs": [0.1],
                    "contour": True,
                    "contourf_kwargs": {"cmap": "spring"},
                    "fill_last": False
                },
            )
            xlims = ax_skyloc.get_xlim()
            ylims = ax_skyloc.get_ylim()
            total_area = (xlims[1]-xlims[0])*(ylims[1]-ylims[0])  # sq. deg.
            ax_skyloc.set_axis_off()
            plt.savefig(
                "sky_localization",
                bbox_inches="tight",
                pad_inches=0,
                dpi=1200,
                orientation="portrait",
            )
            plt.close()
            imgm = cv2.imread(
                "sky_localization.png", 0
            )/255.0  # grayscale image => n x n array
            a, b = imgm.shape
            total_pixels = a*b
            white_pixels = len(np.where(imgm >= 0.999)[0])
            # sq. deg.
            sky_loc_area = total_area * \
                (total_pixels - white_pixels)/total_pixels
            imgm = cv2.resize(imgm, (int(a/5), int(b/5)),
                              interpolation=cv2.INTER_LANCZOS4)  # resizing
            a, b = imgm.shape
            imgm = np.ma.masked_where(
                imgm >= 0.999, imgm, copy=True)  # masking white area
            imgm = np.flip(imgm, axis=0).astype(floatX)
            sns.set()
            figm, axm = plt.subplots(subplot_kw={'projection': 'mollweide'})
            ras = np.linspace(xlims[0]*radian, xlims[1]*radian, a)
            decs = np.linspace(ylims[0]*radian, ylims[1]*radian, b)
            ras = -array([ras[i % a] for i in range(a*b)])
            decs = array([decs[int(i/a)] for i in range(a*b)])
            imgm = imgm.flatten()
            axm.scatter(array([-ra*radian]), array([dec*radian]), s=50.0, color='deepskyblue',
                        marker='x', alpha=1.0, zorder=100, label='GW source position')
            axm.scatter(ras, decs, c=imgm, marker='.', s=0.07, linewidths=0.0,
                        cmap='spring', zorder=99,
                        # norm=mpl_colors.Normalize(vmin=0.0, vmax=1.0), alpha=1.0,
                        label="90%% HPDI contour\nLocalization area = %r deg$^2$" % (
                            np.round(sky_loc_area, 4)))
            tick_labels_x = ['10$^h$', '08$^h$', '06$^h$', '04$^h$', '02$^h$', '00$^h$',
                             '22$^h$', '20$^h$',  '18$^h$', '16$^h$', '14$^h$']
            axm.set_xticklabels(tick_labels_x)
            axm.set_xlabel("RA [hrs]")
            axm.xaxis.label.set_fontsize(12)
            axm.set_ylabel("DEC [$\degree$]")
            axm.yaxis.label.set_fontsize(12)
            axm.legend(
                loc="upper right",
                # labelcolor="deepskyblue",
                fontsize=14.5,
            )

            # Corner plot of marginalized variables:
            az.plot_pair(
                posterior,
                kind='kde',
                marginals=True,
                textsize=8,
            )


if flag_sensi != "y" and flag_err != "y" and flag_range != "y":

    # ----------------------------------------------------------------------------------------------------------------------------
    """ Generate and save SRGO visualization aid. """

    img = (
        plt.imread(
            "earth.jpg"
        )
        / 255.0
    )
    # print(img)
    # plt.imshow(img)
    img = np.roll(img, int(1.12 * np.shape(img)[1]))

    lat = 51.0
    long = 293.2

    if dec < lat - 90.0:
        zod = 1
    else:
        zod = 3
    if dec < -80.0:
        zop = 1
    else:
        zop = 4

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    phis = np.linspace(0, 360, img.shape[1]) * radian
    thetas = np.linspace(0, 180, img.shape[0])[::-1] * radian
    x = 1.0 * np.outer(cos(phis), sin(thetas)).T
    y = 1.0 * np.outer(sin(phis), sin(thetas)).T
    z = 1.0 * np.outer(np.ones(np.size(phis)), cos(thetas)).T
    ax.plot_surface(
        x, y, z, rstride=8, cstride=8, facecolors=img, alpha=1.0, norm="Normalize", zorder=2
    )

    theta1 = (90.0 - lat) * radian
    theta2 = (90.0 - dec) * radian
    r1, r2 = 1.0, 2.0
    x1 = r1 * sin(theta1) * cos(long * radian)
    y1 = r1 * sin(theta1) * sin(long * radian)
    z1 = r1 * cos(theta1)
    x2 = x1 + r2 * sin(theta2) * cos((long + ra - lst) * radian)
    y2 = y1 + r2 * sin(theta2) * sin((long + ra - lst) * radian)
    z2 = z1 + r2 * cos(theta2)
    ax.plot([x1, x2], [y1, y2], [z1, z2],
            color="black", linewidth=1.25, zorder=zod)

    vector_d = (r1 / 2.0) * array([1.0, 0.0, 0.0])
    vector_u = (r1 / 2.0) * array([-1.0, 0.0, 0.0])
    vector_r = (r1 / 2.0) * array([0.0, 1.0, 0.0])
    vector_l = (r1 / 2.0) * array([0.0, -1.0, 0.0])
    axis1 = array([0.0, 0.0, 1.0])
    rotation1 = R.from_rotvec(axis1 * ((long + ra - lst) * radian))
    axis2 = rotation1.apply(vector_r)
    axis2 = axis2 / np.linalg.norm(axis2)
    rotation2 = R.from_rotvec(axis2 * theta2)
    axis3 = array([x2 - x1, y2 - y1, z2 - z1]) / np.linalg.norm(
        array([x2 - x1, y2 - y1, z2 - z1])
    )
    rotation3 = R.from_rotvec(axis3 * psi_eq * radian)
    x2d, y2d, z2d = array([x2, y2, z2]) + \
        rotation3.apply(rotation2.apply(rotation1.apply(vector_d)))
    x2u, y2u, z2u = array([x2, y2, z2]) + \
        rotation3.apply(rotation2.apply(rotation1.apply(vector_u)))
    x2r, y2r, z2r = array([x2, y2, z2]) + \
        rotation3.apply(rotation2.apply(rotation1.apply(vector_r)))
    x2l, y2l, z2l = array([x2, y2, z2]) + \
        rotation3.apply(rotation2.apply(rotation1.apply(vector_l)))
    ax.plot([x2d, x2u], [y2d, y2u], [z2d, z2u],
            color="black", linewidth=1.25, zorder=zop)
    ax.plot([x2l, x2r], [y2l, y2r], [z2l, z2r],
            color="black", linewidth=1.25, zorder=zop)

    elev = 20.0
    azim = -70.0
    ax.view_init(elev=elev, azim=azim)
    fig.set_tight_layout(True)
    fig.gca().set_axis_off()
    fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    ax.margins(0, 0, 0, tight=True)
    fig.gca().xaxis.set_major_locator(plt.NullLocator())
    fig.gca().yaxis.set_major_locator(plt.NullLocator())
    fig.gca().zaxis.set_major_locator(plt.NullLocator())
    ax.set_facecolor("silver")

    # Correcting the aspect ratio so that Earth always looks spherical:
    # Unfortunately, matplotlib currently only supports 'auto' on ax.set_aspect()
    # "mayavi" library, made for 3d plotting --> can use this instead of plt
    upper = 1.0
    if max(x2d, x2u, x2r, x2l) > 1.0:
        upper = max(x2d, x2u, x2r, x2l)
    lower = -1.0
    if min(x2d, x2u, x2r, x2l) < -1.0:
        lower = min(x2d, x2u, x2r, x2l)
    N_x = (1.0 / 2.0) * (upper - lower)

    upper = 1.0
    if max(y2d, y2u, y2r, y2l) > 1.0:
        upper = max(y2d, y2u, y2r, y2l)
    lower = -1.0
    if min(y2d, y2u, y2r, y2l) < -1.0:
        lower = min(y2d, y2u, y2r, y2l)
    N_y = (1.0 / 2.0) * (upper - lower)

    upper = 1.0
    if max(z2d, z2u, z2r, z2l) > 1.0:
        upper = max(z2d, z2u, z2r, z2l)
    lower = -1.0
    if min(z2d, z2u, z2r, z2l) < -1.0:
        lower = min(z2d, z2u, z2r, z2l)
    N_z = (1.0 / 2.0) * (upper - lower)

    ax.set_box_aspect((N_x, N_y, N_z))

    plt.savefig(
        "srgo_visual_aid",
        bbox_inches="tight",
        pad_inches=0,
        dpi=1200,
        orientation="landscape",
    )
    plt.close()
    # ----------------------------------------------------------------------------------------------------------------------------

    # Plotting the signal:
    sns.set()
    fig, ax = plt.subplots(1, 1)
    ax.plot(
        t_plt / 3600.0,
        signal_plt,
        label="Expected SRGO signal without noise",
        color="blue",
        linewidth=1.6,
    )
    if flag_show_pts == "y":
        ax.scatter(
            t / 3600.0,
            noisy_signal,
            label="Synthetic noisy data points (%d data points,\nPSNR $=$ %r, Noise Std. Dev. $=$ %.2g s)" % (
                N, psnr, max(abs(signal)) / psnr),
            color="deeppink",
        )
        m_n = 1.1 * min(min(noisy_signal), min(signal_plt)) / (-2.8e-15)
        m_p = (1.1 * max(max(noisy_signal), max(signal_plt)) - 0.451 * m_n * (-2.8e-15)) / (
            (1.0 - 0.451) * 6.4e-15
        )
    else:
        m_n = 1.1 * min(signal_plt) / (-2.8e-15)
        m_p = (1.1 * max(signal_plt) - 0.451 * m_n * (-2.8e-15)) / (
            (1.0 - 0.451) * 6.4e-15
        )
    ax.axvline(x=day / 3600, linewidth=0.95,
               color="orange", label="Sidereal day")
    ax.axvline(x=2.0 * day / 3600, linewidth=0.95, color="orange")
    ax.axvline(x=3.0 * day / 3600, linewidth=0.95, color="orange")
    ax.set_ylim(m_n * -2.8e-15, m_p * 6.4e-15)
    tick = 1.0
    ax.set_xlabel(
        "Observation Time [hrs] \n1 unit = %r hrs" % tick, fontweight="bold"
    )
    ax.set_ylabel("Ion Circulation-Time Deviation [s]", fontweight="bold")
    ax.text(
        27.0,
        m_p * 6.4e-15 + (4.23e-15 - 6.4e-15) *
        ((m_p * 6.4e-15 - m_n * -2.8e-15)/(6.4e-15 + 2.8e-15)),
        "SRGO latitude $=$ %r$\degree$ \nSRGO initial local sidereal time $=$ %r$^h$ %r$^m$ %r$^s$ \nGW source right ascension $=$ %r$^h$ %r$^m$ %r$^s$ \nGW source declination $=$ %r$\degree$ \nGW polarization angle $=$ %r$\degree$ \nGW initial phase $=$ %r$\degree$ \nEqual mass (~$10^{%d} M_{\odot}$) binary at $z=%r$ \nEnd of inspiral phase at %r hrs \nObserver-SMBBH inclination angle $=$ %r$\degree$"
        % (
            np.round(theta_lat / radian, decimals=1),
            int(LST[0]),
            int(LST[1]),
            np.round(LST[2], decimals=1),
            int(RA[0]),
            int(RA[1]),
            np.round(RA[2], decimals=1),
            np.round(dec, decimals=1),
            np.round(psi_eq, decimals=1),
            np.round(phase, decimals=1),
            np.log10(m1/msol),
            np.round(Z, decimals=1),
            np.round(t_ins_plot / 3600.0, decimals=1),
            np.round(inc, decimals=1),
        ),
        fontsize=12.7,
    )
    ax.xaxis.set_major_locator(mpl_ticker.MultipleLocator(6))
    ax.xaxis.set_major_formatter(mpl_ticker.FormatStrFormatter("%d"))
    ax.xaxis.set_minor_locator(mpl_ticker.MultipleLocator(tick))
    ax.yaxis.set_minor_locator(mpl_ticker.AutoMinorLocator())
    ax.grid(b=True, which="major", color="w", linewidth=1.0)
    ax.grid(b=True, which="minor", color="w", linewidth=0.5)
    ax.legend(loc="best", fontsize=12.7)

    # Adding visualization image to the plot:
    img = plt.imread(
        "srgo_visual_aid.png")
    # fig, ax = plt.subplots(1, 1)
    # plt.imshow(img)
    axin = inset_axes(ax, width="45.1%", height="45.1%", loc="upper right")
    axin.imshow(img, origin="upper", zorder=100)
    axin.axis("off")

    # Extra plots:
    if flag_extras == "y":

        # Signal spectrogram:
        fig_sg, ax_sg = plt.subplots(1, 1)
        f_sg, t_sg, Sxx = spectrogram(signal_plt, 1.0/dt_plt)
        im = ax_sg.pcolormesh(t_sg / 3600.0, f_sg, Sxx,
                              shading='gouraud', cmap='viridis')
        fig_sg.colorbar(im, ax=ax_sg)
        ax_sg.set_ylabel("Frequency [Hz]")
        ax_sg.set_xlabel("Time [hrs]")  # \n1 unit = %r hrs" % tick)
        ax_sg.xaxis.set_major_locator(mpl_ticker.MultipleLocator(6))
        ax_sg.xaxis.set_major_formatter(mpl_ticker.FormatStrFormatter("%d"))
        ax_sg.xaxis.set_minor_locator(mpl_ticker.MultipleLocator(tick))
        ax_sg.xaxis.set_ticks_position("bottom")
        ax_sg.yaxis.set_ticks_position("left")
        ax_sg.set_ylim(0.0, 0.002)

        # Time-evolution of Euler angles:
        fig1, ax1 = plt.subplots(1, 1)
        ax1.plot(t_plt / 3600.0, phi_plt / radian, label="$\phi$")
        ax1.plot(t_plt / 3600.0, theta_plt / radian, label="$\Theta$")
        ax1.plot(t_plt / 3600.0, psi_plt / radian, label="$\psi$")
        ax1.set_xlabel("Time [hrs]")
        # \n1 unit = %r hrs" %tick, fontweight="bold")
        ax1.set_ylabel("Angle [$\degree$]")  # , fontweight="bold")
        ax1.xaxis.set_major_locator(mpl_ticker.MultipleLocator(6))
        ax1.xaxis.set_major_formatter(mpl_ticker.FormatStrFormatter("%d"))
        ax1.yaxis.set_major_locator(mpl_ticker.MultipleLocator(30))
        ax1.yaxis.set_major_formatter(mpl_ticker.FormatStrFormatter("%d"))
        ax1.xaxis.set_minor_locator(mpl_ticker.MultipleLocator(tick))
        # ax1.yaxis.set_minor_locator(mpl_ticker.AutoMinorLocator())
        plt.grid(which="minor")
        # ax1.plot(t_plt/3600., f(t_plt, f0, k, Z), label="GW frequency")
        # ax1.plot(t_plt/3600., h_plus(t_plt, inc, phase, d, f0, k, Z), label="GW strain")
        # ax1.plot(t_plt/3600., h_eff(t_plt, phi, theta, psi, inc, phase, d, f0, k, Z), label="Effective GW strain")
        ax1.legend(loc="best")

    # Saving all the figures:
    directory = "saved_plots/"
    fig_nums = plt.get_fignums()
    figs = [plt.figure(n) for n in fig_nums]
    if flag_mcmc == "y":
        if flag_extras == "y":
            plotname = ["trace", "joint_posterior", "sky_map", "corner_plot",
                        "signal", "spectrogram", "angles"]
        else:
            plotname = ["trace", "joint_posterior", "sky_map", "corner_plot",
                        "signal"]
    else:
        if flag_extras == "y":
            plotname = ["signal", "spectrogram", "angles"]
        else:
            plotname = ["signal"]

    # https://stackoverflow.com/questions/42354515/how-to-display-a-plot-in-fullscreen
    f_num = 0
    for fig in figs:
        if plotname[f_num] == "sky_map" or plotname[f_num] == "corner_plot" or plotname[f_num] == "signal":
            fig.canvas.manager.full_screen_toggle()
            fig.tight_layout()
        fig.savefig(directory + plotname[f_num] +
                    ".png", format="png", dpi=fig.dpi)
        f_num += 1
        # plt.close(fig)
