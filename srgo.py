""" Simulating the detection of millihertz (mHz) gravitational waves (GWs)
    from astrophysical sources by a Storage Ring Gravitational-wave Observatory (SRGO).
    Authors: Suvrat Rao, Hamburg Observatory, University of Hamburg
             Julia Baumgarten, Physics department, Jacobs University Bremen """

import numpy as np
from matplotlib import ticker as mpl_ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from scipy.spatial.transform import Rotation as R
from scipy.signal import spectrogram
# from scipy.signal import wiener
# from scipy.ndimage import gaussian_filter
from scipy.integrate import simps
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d
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
prod = np.prod
delete = np.delete
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
M1 = 1000000.0  # mass of object #1 in solar masses
M2 = 1000000.0  # mass of object #2 in solar masses
r = 1.0  # initial separation between bodies in au
# GW SOURCE-OBSERVER PARAMETERS:
inc = 0.0  # inclination angle of observer w.r.t binary system in degrees
Z = 0.1  # redshift of the binary system
phase = 0.0  # GW signal initial phase in degrees
# local sidereal time of SRGO in hh:mm:ss at the start of the observation run
LST = (00.0, 00.0, 00.0)
RA = (00.0, 00.0, 00.0)  # GW source right ascension in hh:mm:ss
dec = 00.0  # GW source declination in degrees
# GW polarization angle (in degrees) as measured in equatorial celestial coordinates
psi_eq = 0.0
# STORAGE RING PARAMETERS:
v0 = 0.999999991*c  # SRGO ion bunch speed (must be ultrarelativistic)
L = 100.0  # 26659.0  # SRGO circumference in meters
n_p = 1  # 2*2808  # SRGO number of bunches
# MCMC PARAMETERS AND FLAGS:
# do parameter estimation with MCMC using synthetic data? ([y]/n)
flag_mcmc = "y"
T_obs = day  # duration of SRGO observation run
N = 16  # no. of SRGO data points acquired during T_obs
psnr = 10  # peak signal to noise ratio
# "ligo" (Whittle likelihood in Fourier space) or "custom" likelihood functions for MCMC
likelihood_type = "ligo"
flag_show_pts = "y"  # show synthetic noisy data-points in signal plot ([y]/n)
flag_cp = "y"  # show corner-plot subplots individually ([y]/n)
flag_extras = "n"  # plot spectrogram & evolution of orientation angles ([y]/n)
flag_sensi = "n"  # compute SRGO sensitivity curve ([y]/n)
flag_err = "n"  # compute numerical integration error ([y]/n)
flag_range = "n"  # compute observational range of SRGO ([y]/n)
flag_srcpos_skyloc = "n"  # plot sky-loc precision vs. src position ([y]/n)
flag_psnr = "n"  # plot sky-loc area vs. PSNR ([y]/n)
flag_tobs = "n"  # plot sky-loc area vs. observation time ([y]/n)
flag_mass = "n"  # plot mass error vs. PSNR ([y]/n)
flag_dist = "n"  # plot distance error vs. PSNR ([y]/n)

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
# GW source luminosity distance
# https://iopscience.iop.org/article/10.1086/313167
# d = (c/H0)*(Z + (1.0 - 3.0*W0/4.0)*Z**2.0 + (9.0*W0-10.0)*(W0/8.0)*Z**3.0)
d = 10**6 * pc * cosmo.luminosity_distance(Z).value


def f(t, f0, k, Z):
    """GW frequency."""
    # Theano tensor variable and numpy object compatibility:
    if flag_mcmc == "y" and flag_sensi != "y" and flag_err != "y" and flag_range != "y" and flag_srcpos_skyloc != "y" and flag_psnr != "y" and flag_tobs != "y" and flag_mass != "y" and flag_dist != "y" and str(type(t)) == "<class 'numpy.ndarray'>":
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


def h_plus(t, inc, phase, d, f0, k, Z, M):
    """GW plus-polarization."""
    inc *= radian
    phase *= radian
    # Theano tensor variable and numpy object compatibility:
    if flag_mcmc == "y" and flag_sensi != "y" and flag_err != "y" and flag_range != "y" and flag_srcpos_skyloc != "y" and flag_psnr != "y" and flag_tobs != "y" and flag_mass != "y" and flag_dist != "y" and str(type(t)) == "<class 'numpy.ndarray'>":
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


def h_cross(t, inc, phase, d, f0, k, Z, M):
    """GW cross-polarization."""
    inc *= radian
    phase *= radian
    # Theano tensor variable and numpy object compatibility:
    if flag_mcmc == "y" and flag_sensi != "y" and flag_err != "y" and flag_range != "y" and flag_srcpos_skyloc != "y" and flag_psnr != "y" and flag_tobs != "y" and flag_mass != "y" and flag_dist != "y" and str(type(t)) == "<class 'numpy.ndarray'>":
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
    if flag_mcmc == "y" and flag_sensi != "y" and flag_err != "y" and flag_range != "y" and flag_srcpos_skyloc != "y" and flag_psnr != "y" and flag_tobs != "y" and flag_mass != "y" and flag_dist != "y":
        # to be able to use FFT trick for computing speed
        n = findPrevPowerOf2(n)
        if n < int(2.0*f(t_obs, f0, k, Z)*t_obs)+1:
            n = 2*n
    # allowed timing interval
    T_timing = int(t_obs / (n * T_sample)) * T_sample
    t = arange(0.0, t_obs, T_timing, dtype=floatX)  # timing data points
    if flag_mcmc == "y" and flag_sensi != "y" and flag_err != "y" and flag_range != "y" and flag_srcpos_skyloc != "y" and flag_psnr != "y" and flag_tobs != "y" and flag_mass != "y" and flag_dist != "y" and len(t) > n:
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


def h_eff(t, phi, theta, psi, inc, phase, d, f0, k, Z, M):
    """Effective longitudinal GW strain for SRGO test masses
    that are circulating rapidly compared to the GW frequency."""
    return (
        -(1.0 / 2.0)
        * (
            F_plus(phi, theta, psi) * h_plus(t, inc, phase, d, f0, k, Z, M)
            + F_cross(phi, theta, psi) * h_cross(t, inc, phase, d, f0, k, Z, M)
        )
    )


def boole_quad(y, x, yofx, a, b, c, d, e, g, k, l, p, q):
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
        + 32.0 * yofx(x[0] + h, a1, b1, c1, d, e, g, k, l, p, q)
        + 12.0 * yofx(x[0] + 2.0 * h, a2, b2, c2, d, e, g, k, l, p, q)
        + 32.0 * yofx(x[0] + 3.0 * h, a3, b3, c3, d, e, g, k, l, p, q)
        + 7.0 * y[1]
    )


def SRGO_signal(t, phi, theta, psi, inc, phase, d, f0, k, Z, M):
    """SRGO signal from response to GWs."""
    integrand = h_eff(t, phi, theta, psi, inc, phase, d, f0, k, Z, M)

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
                Z,
                M
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
                         psi_plt, inc, phase, d, f0, k, Z, M)


# ----------------------------------------------------------------------------------------------------------------------------


def error(p):
    """For obtaining the numerical error of integration given the no. of data pts, 'p'."""
    t_err, dt_err = time(p, 1.0*day, t_ins, f0, k, Z)
    phi_err, theta_err, psi_err = earth_rot(
        t_err, ra, lst, dec, psi_eq, "regular")
    signal_err = SRGO_signal(t_err, phi_err, theta_err,
                             psi_err, inc, phase, d, f0, k, Z, M)

    err = max(abs(signal_err - signal_ref(t_err)))
    print(p, err)
    return(err)


if flag_err == "y":
    t_ref, dt_ref = time(10**5, 3.0*day, t_ins, f0, k, Z)  # increase to 10**7
    phi_ref, theta_ref, psi_ref = earth_rot(
        t_ref, ra, lst, dec, psi_eq, "regular")
    signal_ref = SRGO_signal(t_ref, phi_ref, theta_ref,
                             psi_ref, inc, phase, d, f0, k, Z, M)
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
    fig2.savefig("saved_plots/numerical_error.png", format="png", dpi=800)
    # plt.close(fig2)


def srgo_range(dee, zed):
    """For numerically obtaining the SRGO observational range."""
    M_rng = (1 + zed) * ((m1 * m2) ** (3.0 / 5.0)) / \
        ((m1 + m2) ** (1.0 / 5.0))  # chirp mass
    # final separation between bodies @ ISCO
    Rf_rng = 6.0 * G * max(m1, m2) / c ** 2.0
    # initial GW frequency (realistic lowest possible mHz to give max. signal and cover all sources)
    f0_rng = 5.0*10**-4
    # final GW frequency at the end of inspiral phase
    f1_rng = 2.0*sqrt(G * (m1 + m2)) / (2.0 * pi * Rf_rng ** (3.0 / 2.0))
    k_rng = (96.0 / 5.0) * (pi ** (8.0 / 3.0)) * \
        ((G * M_rng / c ** 3.0) ** (5.0 / 3.0))
    t_ins_rng = (
        (3.0 / 8.0) * (1.0 / k_rng) *
        (f0_rng ** (-8.0 / 3.0) - f1_rng ** (-8.0 / 3.0))
    )  # inspiral time
    inc_rng = 0.0  # inclination angle
    phase_rng = 0.0  # initial GW phase

    # set 3 hours as standard obs time
    t_rng, dt_rng = time(20000, day/8.0, t_ins_rng, f0_rng, k_rng, zed)
    phi_rng, theta_rng, psi_rng = earth_rot(
        t_rng, ra, lst, dec, psi_eq, "regular")
    signal_range = SRGO_signal(
        t_rng, phi_rng, theta_rng, psi_rng, inc_rng, phase_rng, dee, f0_rng, k_rng, zed, M_rng)

    peak_signal = max(abs(signal_range))
    print(zed, m1/msol, t_ins_rng/day, peak_signal)
    return(peak_signal)


if flag_range == "y":
    ra_vals = np.linspace(-45.0, 90.0, 4)
    dec_vals = np.linspace(-45.0, 90.0, 4)
    psi_eq_vals = np.linspace(-45.0, 90.0, 4)

    # up to redshift of first stars (Cosmic Dawn)
    z_array1 = np.linspace(0.0, 15.0, 100000)
    d_array1 = 10**6 * pc * cosmo.luminosity_distance(z_array1).value
    z_of_d = interp1d(d_array1, z_array1)
    # from closest mHz spectroscopic binaries up to Cosmic Dawn distance
    d_array = array([50*pc, 10**2*pc, 599*pc, 600*pc, 10**3*pc, 1499*pc, 1500*pc, 2199*pc, 2200*pc, 7999*pc, 8000*pc, 10**4*pc, 10**5*pc, 777847*pc, 777847.7361*pc,
                    10**6*pc, 10**7*pc, 27.3*pc*10**6, 27.4*pc*10**6, 10**8*pc, 10**9*pc, 10**10*pc, 10**6*pc*cosmo.luminosity_distance(10.0).value])
    z_array = z_of_d(d_array)
    sig_array = np.zeros(len(d_array))

    for aaa in ra_vals:
        ra = aaa
        for bbb in dec_vals:
            dec = bbb
            for ccc in psi_eq_vals:
                psi_eq = ccc
                for i in range(len(d_array)):
                    if z_array[i] > 10.0:  # first mHz SMBBHs due to galaxy mergers
                        # https://arxiv.org/abs/1903.06867
                        # https://iopscience.iop.org/article/10.3847/2041-8213/ab2646
                        # https://iopscience.iop.org/article/10.1088/1475-7516/2016/04/002/pdf
                        m1 = 10**5 * msol  # only seed IMBH black holes beyond z=10
                        m2 = m1  # equal mass binaries
                    elif z_array[i] <= 10.0 and d_array[i] >= 27.4*pc*10**6:
                        # closest SMBBH (NGC 7727)
                        # https://doi.org/10.1051/0004-6361/202140827
                        m1 = 10**7 * msol   # based on evolution of SMBH mass function
                        m2 = m1
                    elif d_array[i] < 27.4*pc*10**6 and d_array[i] >= 777847.7361*pc:
                        # closest galaxy (Andromeda)
                        # https://iopscience.iop.org/article/10.1086/432434
                        m1 = 10**5 * msol  # equivalent to 10^3 & 10^8 msol EMRI
                        m2 = m1
                    elif d_array[i] < 777847.7361*pc and d_array[i] >= 8000*pc:
                        # closest dwarf galaxy (Canis Major) AND Milky Way center
                        # https://academic.oup.com/mnras/article/473/1/1186/4060726
                        # https://iopscience.iop.org/article/10.3847/2041-8213/ac1170
                        m1 = 10**6 * msol
                        m2 = 10**3 * msol
                    elif d_array[i] < 8000*pc and d_array[i] >= 2200*pc:
                        # closest globular cluster within our galaxy (Milky Way)
                        # https://en.wikipedia.org/wiki/List_of_globular_clusters#Milky_Way
                        m1 = 10**4 * msol
                        m2 = 10 * msol
                    elif d_array[i] < 2200*pc and d_array[i] >= 1500*pc:
                        # closest black hole within our galaxy (Milky Way)
                        # https://arxiv.org/abs/2201.13296
                        m1 = 10 * msol
                        m2 = m1
                    elif d_array[i] < 1500*pc and d_array[i] >= 600*pc:
                        # closest mHz binary neutron star within our galaxy
                        # https://iopscience.iop.org/article/10.1088/0004-637X/789/2/119
                        # http://www.johnstonsarchive.net/relativity/binpulstable.html
                        # https://en.wikipedia.org/wiki/PSR_J0737%E2%88%923039
                        # https://journals.aps.org/prx/abstract/10.1103/PhysRevX.11.041050
                        # https://www.science.org/doi/10.1126/science.1094645
                        # https://iopscience.iop.org/article/10.3847/1538-4357/ab12e3
                        # https://iopscience.iop.org/article/10.3847/1538-4357/aa7e89
                        # https://academic.oup.com/mnrasl/article/475/1/L57/4794951
                        # https://iopscience.iop.org/article/10.3847/2041-8213/aaad06
                        # https://iopscience.iop.org/article/10.3847/1538-4357/aabf8a
                        m1 = 1 * msol
                        m2 = m1
                    elif d_array[i] < 600*pc:
                        # closest mHz spectroscopic binaries within Milky Way:
                        # https://doi.org/10.1051/0004-6361:20041213
                        # https://sb9.astro.ulb.ac.be/
                        # https://gaia.ari.uni-heidelberg.de/singlesource.html
                        # https://gea.esac.esa.int/archive/documentation/GDR3/Miscellaneous/sec_credit_and_citation_instructions/
                        # https://en.wikipedia.org/wiki/WZ_Sagittae
                        # https://en.wikipedia.org/wiki/EX_Hydrae
                        # https://en.wikipedia.org/wiki/OY_Carinae
                        # https://en.wikipedia.org/wiki/Gliese_829
                        # https://en.wikipedia.org/wiki/GJ_3991
                        # https://en.wikipedia.org/wiki/Delta_Trianguli
                        # https://en.wikipedia.org/wiki/HR_5553
                        # https://en.wikipedia.org/wiki/HR_9038
                        # https://en.wikipedia.org/wiki/Zeta_Trianguli_Australis
                        # https://en.wikipedia.org/wiki/Delta_Capricorni
                        # https://en.wikipedia.org/wiki/Iota_Pegasi
                        m1 = 0.5 * msol
                        m2 = m1

                    sig_array[i] = max(
                        sig_array[i], srgo_range(d_array[i], z_array[i]))

    sns.reset_orig()
    fig3, axes3 = plt.subplots(1, 1)
    axes3 = plt.gca()
    axes4 = axes3.twiny()  # the spines of the second axes are overdrawn on the first
    facecolour = "snow"
    axes3.set_facecolor(facecolour)
    # fig3.set_facecolor(facecolour)
    textcolour = "black"
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
    #txtsize = axes3.xaxis.label.get_size()
    axes3.plot(d_array/pc, sig_array, linewidth=1.5, color="blue", alpha=0.75,
               zorder=10, label=r'$f_{GW}=5\times10^{-4}$ Hz; $T_{obs}=3.0$ hrs')
    axes3.axvline(x=50, linewidth=1.0, linestyle='dashdot', color="darkorchid",
                  label="nearest ~$0.5M_{\odot}$ WD binaries")
    axes3.axvline(x=600, linewidth=1.0, linestyle='dashdot', color="hotpink",
                  label="nearest ~$1M_{\odot}$ NS binaries")
    axes3.axvline(x=1500, linewidth=1.0, linestyle='dashdot', color="deepskyblue",
                  label="nearest ~$10M_{\odot}$ BH binaries")
    axes3.axvline(x=2200, linewidth=1.0, linestyle='dashdot', color="slategray",
                  label="nearest ~$10M_{\odot}$ & $10^4M_{\odot}$ EMRIs")
    axes3.axvline(x=8000, linewidth=1.0, linestyle='dashdot', color="mediumseagreen",
                  label="nearest ~$10^3M_{\odot}$ & $10^6M_{\odot}$ EMRIs")
    axes3.axvline(x=777847.7361, linewidth=1.0, linestyle='dashdot', color="chocolate",
                  label="nearest ~$10^3M_{\odot}$ & $10^8M_{\odot}$ EMRIs")
    axes3.axvline(x=27.4*10**6, linewidth=1.0, linestyle='dashdot', color="darkorange",
                  label="nearest ~$10^7M_{\odot}$ SMBH binaries")
    axes3.axvline(x=cosmo.luminosity_distance(10.0).value*10**6, linewidth=1.0, color="red",
                  linestyle='dashdot', label="first ~$10^7M_{\odot}$ SMBH binaries")
    axes3.set_xlabel("GW source distance [pc]")
    axes3.set_ylabel(
        "Maximum response signal[s]")
    axes3.set_xscale('log')
    axes3.set_yscale('log')
    axes3.set_ylim(10**-21, 10**-6)
    axes3.set_xticks([10, 10**2, 10**3, 10**4, 10**5, 10 **
                     6, 10**7, 10**8, 10**9, 10**10, 10**11])
    axes3.set_yticks([10**-21, 10**-18, 10**-15, 10**-12, 10**-9, 10**-6])
    axes3.tick_params(axis='y', which='minor')
    leg = axes3.legend(loc='upper right', bbox_to_anchor=(
        0.94, 1.0), facecolor=facecolour, framealpha=1, prop={'size': 5.75})
    for text in leg.get_texts():
        text.set_color(textcolour)
    axes4.plot(z_array, sig_array, color='orange', linewidth=0.0)
    axes4.set_xlabel("GW source redshift (z)")
    axes4.set_xscale('log')
    axes4.set_xticks([10**-9, 10**-8, 10**-7, 10**-6,
                     10**-5, 10**-4, 10**-3, 10**-2, 10**-1, 10**0, 10**1])
    # fig3.canvas.manager.full_screen_toggle()
    fig3.tight_layout()
    fig3.savefig("saved_plots/srgo_range.png", format="png",
                 dpi=800, facecolor=fig3.get_facecolor())
    # plt.close(fig3)


def sensi_curve(b):
    """For numerically obtaining the SRGO sensitivity curve."""  # https://en.wikipedia.org/wiki/Spectral_density#Power_spectral_density
    sigma_noise = 10**(-12)

    M_sns = (1 + Z) * ((m1 * m2) ** (3.0 / 5.0)) / \
        ((m1 + m2) ** (1.0 / 5.0))  # chirp mass
    # final separation between bodies @ ISCO
    Rf_sns = 6.0 * G * max(m1, m2) / c ** 2.0
    f0_sns = 2.0*sqrt(G * (m1 + m2)) / (2.0 * pi * (r*au) **
                                        (3.0 / 2.0))  # initial GW frequency
    # final GW frequency at the end of inspiral phase
    f1_sns = 2.0*sqrt(G * (m1 + m2)) / (2.0 * pi * Rf_sns ** (3.0 / 2.0))
    k_sns = (96.0 / 5.0) * (pi ** (8.0 / 3.0)) * \
        ((G * M_sns / c ** 3.0) ** (5.0 / 3.0))
    t_ins_sns = (
        (3.0 / 8.0) * (1.0 / k_sns) *
        (f0_sns ** (-8.0 / 3.0) - f1_sns ** (-8.0 / 3.0))
    )   # inspiral time

    # Normalize for T_obs = 3.0 hours
    t_sensi, dt_sensi = time(20000, day/8.0, t_ins_sns, f0_sns, k_sns, Z)
    phi_sensi, theta_sensi, psi_sensi = earth_rot(
        t_sensi, ra, lst, dec, psi_eq, "regular")
    # signal_sensi = SRGO_signal(  # for binary GW sources
    #    t_sensi, phi_sensi, theta_sensi, psi_sensi, inc, phase, b, f0_sns, k_sns, Z, M_sns)
    signal_sensi_CW = SRGO_signal(  # for continuous GW sources
        t_sensi, phi_sensi, theta_sensi, psi_sensi, inc, phase, b, f0_sns, 0.0, Z, M_sns)

    # val_lhs = (np.mean(signal_sensi**2.0))**0.5  # RMS signal
    val_lhs_CW = (np.mean(signal_sensi_CW**2.0))**0.5
    val_rhs = sigma_noise / \
        (n_p*(1.0/T_sample)*t_sensi[-1])**0.5  # effective noise
    # global d
    # d = b*val_lhs/val_rhs  # value of GW source distance where val_lhs == val_rhs i.e. SNR=1

    f_mean = f(0.0, f0_sns, 0.0, Z)
    # h_rms = ((np.mean(h_plus(t_sensi, inc, phase, d, f0_sns, k_sns, Z, M_sns)**2.0)
    #         + np.mean(h_cross(t_sensi, inc, phase, d, f0_sns, k_sns, Z, M_sns)**2.0))/2.0)**0.5
    h_rms_CW = val_rhs * h_plus(0.0, 0.0, 0.0, b,
                                f0_sns, 0.0, Z, M_sns) / val_lhs_CW
    print(d/pc, f_mean, h_rms_CW)
    return(f_mean, h_rms_CW)


if flag_sensi == "y":
    M1_vals = np.linspace(3.0, 5000., 1)
    ra_vals = np.linspace(-45.0, 90.0, 4)
    dec_vals = np.linspace(-45.0, 90.0, 4)
    psi_eq_vals = np.linspace(-45.0, 90.0, 4)
    inc_vals = np.linspace(0.0, 90.0, 3)
    phase_vals = np.linspace(0.0, 315.0, 8)

    l = 30
    f0_array = np.zeros(l)
    #h0_array = np.zeros(l)
    #h0_temp = np.zeros(l)
    h0_array_CW = np.zeros(l)
    h0_temp_CW = np.zeros(l)

    # N0 = ones(l) * (len(ra_vals)*len(dec_vals)*len(psi_eq_vals)
    #                * len(inc_vals)*len(phase_vals)*len(M1_vals))
    N0_CW = ones(l) * (len(ra_vals)*len(dec_vals)*len(psi_eq_vals)
                       * len(inc_vals)*len(phase_vals)*len(M1_vals))
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
                                                                              np.logspace(-7.0, -0.0, l)))**(2.0/3.0)
                            for i in range(l):
                                r = separation[i]
                                f_0, h_0_CW = sensi_curve(d)
                                if f0_array[i] == 0.0:
                                    f0_array[i] = f_0
                                # if np.isnan(h_0):
                                #    h_0 = 0.0
                                #    N0[i] -= 1
                                #h0_temp[i] = h_0
                                if np.isnan(h_0_CW):
                                    h_0_CW = 0.0
                                    N0_CW[i] -= 1
                                h0_temp_CW[i] = h_0_CW

                            #h0_array += h0_temp
                            h0_array_CW += h0_temp_CW

    #h0_array *= 1.0/N0
    h0_array_CW *= 1.0/N0_CW

    # To simulate graphic:
    # First, plotting the paper-I sensitivity curve, modified to be normalized to T_obs = 3.0 hrs
    fq = 10**np.linspace(-11, 6, 1000000)
    i = int(np.where(np.abs(fq-10**(-7.0)) ==
            np.min(np.abs(fq-10**(-7.0))))[0])
    j = int(np.where(np.abs(fq-10**(-0.0)) ==
            np.min(np.abs(fq-10**(-0.0))))[0])
    i1 = int(np.where(np.abs(fq-10**(-8.0)) ==
             np.min(np.abs(fq-10**(-8.0))))[0])
    j1 = int(np.where(np.abs(fq-10**(1.0)) == np.min(np.abs(fq-10**(1.0))))[0])

    t_noise = 10.0**(-12)
    f_sampling = 2.0*2808*11245

    # h = (16.0*np.pi/np.sqrt(30.0))*(t_noise/np.sqrt(f_sampling))*fq**(3.0/2.0)   # from paper-I
    T_obs_sns = day/8.0
    h = (16.0*np.pi/np.sqrt(3.0))*(t_noise/np.sqrt(f_sampling*T_obs_sns)) * \
        fq**(1.0)   # modified for T_obs = 3.0 hrs

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
    plt.fill_between(fe, he, facecolor='darkcyan', alpha=0.6,
                     label="Extreme mass-ratio inspirals")
    plt.fill_between(fr, hr, facecolor='lightskyblue', alpha=0.6,
                     label="Resolvable galactic binaries")
    plt.fill_between(fu, hu, facecolor='lime', alpha=0.6,
                     label="Unresolvable galactic binaries")
    plt.plot(fq, h, color='black', linewidth=0.0)  # LHC-GW
    plt.plot(f0_array, h0_array_CW/(T_obs_sns)**0.5, color='black', linewidth=1.5,
             label="SRGO sensitivity curve\n$T_{obs}=3.0$hrs, $\sigma_{noise}=1 $ps, $f_{sample}=2.998$MHz")
    # plt.plot(f0_array, h0_array, color='black', linestyle='dashed', linewidth=1.5,
    #         label="Sensitivity to resolvable binary inspiral GWs,\n$T_{obs}=3.0$hrs, $\sigma_{noise}=1 $ps, $f_{sample}=2.998$MHz")
    plt.legend(loc=2, borderpad=0.2, labelspacing=0.2)
    plt.ylim(10**(-26.), 10**(-10.))
    plt.ylabel(
        "Strain Amplitude Spectral Density [1/$\sqrt{Hz}$]", fontweight='bold')
    plt.xlabel("Frequency [Hz]", fontweight='bold')
    plt.yscale('log')
    plt.xscale('log')
    plt.grid()
    plt.tick_params(axis='x', which='minor')
    plt.tight_layout()
    plt.savefig("saved_plots/srgo_sensitivity.png", dpi=800)
    # plt.close()


# ----------------------------------------------------------------------------------------------------------------------------


# Creating synthetic signal and adding only stochastic noise:
if flag_sensi != "y" and flag_err != "y" and flag_range != "y" and flag_srcpos_skyloc != "y" and flag_psnr != "y" and flag_tobs != "y" and flag_mass != "y" and flag_dist != "y":

    t, dt = time(N, T_obs, t_ins, f0, k, Z)
    N = len(t)
    phi, theta, psi = earth_rot(t, ra, lst, dec, psi_eq, "regular")
    signal = SRGO_signal(t, phi, theta, psi, inc, phase, d, f0, k, Z, M)
    noise = np.random.normal(0.0, scale=max(
        abs(signal_plt)) / psnr, size=N).astype(floatX)
    # noise = np.random.normal(
    #    0.0, scale=2.1155*10.**(-16), size=N).astype(floatX)
    noisy_signal = (signal + noise).astype(floatX)
    # noisy_signal = wiener(noisy_signal, noise=max(
    #   abs(signal_plt)) / psnr)  # Wiener noise filter
    # Noise filter introduces new correlations between the independent data-points.
    # Then we must account for the covariance matrix in the mcmc.
    # https://www.researchgate.net/post/Does_using_a_noise_filter_before_performing_MCMC_fitting_of_a_model_to_noisy_data_increase_or_decrease_the_accuracy_of_the_fit


# Simulating noisy data and computing sky localization using MCMC:
if flag_mcmc == "y" and flag_sensi != "y" and flag_err != "y" and flag_range != "y" and flag_srcpos_skyloc != "y" and flag_psnr != "y" and flag_tobs != "y" and flag_mass != "y" and flag_dist != "y":

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

    def interp(xs, ys, x):
        """Custom implementation of Lagrange polynomial interpolation """

        x_minus_xs = array([x - xs[j] for j in range(len(xs))])
        y_of_x = sum([ys[i]*prod(delete(x_minus_xs, i))/prod(delete(xs[i]-xs, i))
                     for i in range(len(xs))])
        return y_of_x

    Zs = np.linspace(0.0, 15.0, 150)
    ds = 10**6 * pc * cosmo.luminosity_distance(Zs).value

    MCMC = pm.Model()

    with MCMC:

        # Priors for unknown model parameters:
        M1_prior = pm.Uniform(  # pm.Bound(pm.Normal, lower=0.0, upper=360.0)(
            "GW source mass #1 posterior [$M_{\odot}$]", lower=0.00000005*M1, upper=1.99999995*M1
        )
        M2_prior = pm.Uniform(  # pm.Bound(pm.Normal, lower=0.0, upper=360.0)(
            "GW source mass #2 posterior [$M_{\odot}$]", lower=0.00000005*M2, upper=1.99999995*M2
        )
        r_prior = pm.Uniform(  # pm.Bound(pm.Normal, lower=0.0, upper=360.0)(
            "Initial binary separation posterior [$au$]", lower=0.5*r, upper=1.5*r
        )
        inc_prior = pm.Uniform(  # pm.Bound(pm.Normal, lower=0.0, upper=360.0)(
            "GW source inclination posterior [$\degree$]", lower=inc-180.0, upper=inc+180.0
        )
        Z_prior = pm.Uniform(  # pm.Bound(pm.Normal, lower=0.0, upper=360.0)(
            "GW source redshift posterior", lower=6.666666666666667e-11*Z, upper=1.9999999999333333*Z
        )
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
        M_og = M
        M = (1.0 + Z_prior) * ((m1 * m2) ** (3.0 / 5.0)) / \
            ((m1 + m2) ** (1.0 / 5.0))  # chirp mass
        # max(m1,m2) without using boolean operations for Theano compatibility
        # max_m1_m2 = (abs(m1 - m2) + m1 + m2)/2.0
        f0 = 2.0*sqrt(G * (m1 + m2)) / (2.0 * pi * (r_prior*au) **
                                        (3.0 / 2.0))  # initial GW frequency
        k = (96.0 / 5.0) * (pi ** (8.0 / 3.0)) * \
            ((G * M / c ** 3.0) ** (5.0 / 3.0))
        # luminosity distance of the binary system
        # https://iopscience.iop.org/article/10.1086/313167
        d = (c/H0)*(Z_prior + (1.0 - 3.0*W0/4.0)*Z_prior **
                    2.0 + (9.0*W0-10.0)*(W0/8.0)*Z_prior**3.0)
        # d = interp(Zs, ds, Z_prior)
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
                            phase_prior, d, f0, k, Z_prior, M)
            )) / denominator
            model = (
                model ** (model ** 0.001) - 1.0 +
                abs(model ** (model ** 0.001) - 1.0)
            ) * ones(N, dtype=floatX)
            model = model.reshape(N, 1).astype(floatX)
        elif likelihood_type == "ligo":
            model = SRGO_signal(t, phim, thetam, psim,
                                inc_prior, phase_prior, d, f0, k, Z_prior, M)
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
            step = pm.step_methods.Metropolis(S=np.identity(9))
            print("Running MCMC...")
            # https://chi-feng.github.io/mcmc-demo/
            posterior = pm.sample(
                step=step,
                draws=M,
                # MCMC is working properly if true parameter values lie within 90% HPD region, 90% of the time.
                chains=100,
                cores=1,
                # start={'m1_prior': m1},
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

            # Mass estimation error:
            ax_m1m2 = az.plot_pair(
                posterior,
                var_names=[
                    "GW source mass #1 posterior [$M_{\odot}$]",
                    "GW source mass #2 posterior [$M_{\odot}$]",
                ],
                kind="kde",
                # marginals=True,  #causes error in get_xlim()
                figsize=(9.25, 9.25),
                kde_kwargs={
                    # Plot HPD region (highest posterior density)
                    # for (x)% HPDR, the hdi_probs argument is x/100
                    "hdi_probs": [0.67],
                    "contour": True,
                    "contourf_kwargs": {"cmap": "spring"},
                    "fill_last": False,
                },
            )
            xlims = ax_m1m2.get_xlim()
            ylims = ax_m1m2.get_ylim()
            total_area = (xlims[1]-xlims[0])*(ylims[1]-ylims[0])  # msol^2
            ax_m1m2.set_axis_off()
            plt.savefig(
                "M1_M2_posterior",
                bbox_inches="tight",
                pad_inches=0,
                dpi=800,
                orientation="portrait",
            )
            plt.close()
            imgm = cv2.imread(
                "M1_M2_posterior.png", 0
            )/255.0  # grayscale image => n x n array
            a, b = imgm.shape
            total_pixels = a*b
            white_pixels = len(np.where(imgm >= 0.999)[0])
            # relative mass estimation error
            rel_m_err = (1/M1)*((4/pi) * total_area *
                                (total_pixels - white_pixels)/total_pixels)**0.5
            # If the true parameter values don't lie within the HPD region, we ignore the result
            if imgm[int((a-1)*(ylims[1]-M2)/(ylims[1]-ylims[0]))][int((b-1)*(M1-xlims[0])/(xlims[1]-xlims[0]))] >= 0.999:
                rel_m_err = np.nan

            # Distance estimation error:
            ax_zs = az.plot_pair(
                posterior,
                var_names=[
                    "GW source redshift posterior",
                    "Initial binary separation posterior [$au$]",
                ],
                kind="kde",
                # marginals=True,
                figsize=(9.25, 9.25),
                kde_kwargs={
                    # Plot HPD region (highest posterior density)
                    # for (x)% HPDR, the hdi_probs argument is x/100
                    "hdi_probs": [0.67],
                    "contour": True,
                    "contourf_kwargs": {"cmap": "spring"},
                    "fill_last": False,
                },
            )
            xlims = ax_zs.get_xlim()
            ylims = ax_zs.get_ylim()
            ax_zs.set_axis_off()
            plt.savefig(
                "z_sep_posterior",
                bbox_inches="tight",
                pad_inches=0,
                dpi=800,
                orientation="portrait",
            )
            plt.close()
            imgm = cv2.imread(
                "z_sep_posterior.png", 0
            )/255.0  # grayscale image => n x n array
            a, b = imgm.shape
            # % relative distance estimation error
            rel_d_err = (1/Z)*(max(np.where(imgm < 0.999)[1]) - min(
                np.where(imgm < 0.999)[1])) * (xlims[1]-xlims[0]) / b
            # If the true parameter values don't lie within the HPD region, we ignore the result
            if imgm[int((a-1)*(ylims[1]-r)/(ylims[1]-ylims[0]))][int((b-1)*(Z-xlims[0])/(xlims[1]-xlims[0]))] >= 0.999:
                rel_d_err = np.nan

            # Plotting sky-localization in Mollweide projection:
            ax_skyloc = az.plot_pair(
                posterior,
                var_names=[
                    "GW source RA posterior [$\degree$]",
                    "GW source DEC posterior [$\degree$]",
                ],
                kind="kde",
                # marginals=True,
                figsize=(9.25, 9.25),
                kde_kwargs={
                    # Plot HPD region (highest posterior density)
                    # for (x)% HPDR, the hdi_probs argument is x/100
                    "hdi_probs": [0.67],
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
                dpi=800,
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
            # If the true parameter values don't lie within the HPD region, we ignore the result
            if imgm[int((a-1)*(ylims[1]-dec)/(ylims[1]-ylims[0]))][int((b-1)*(ra-xlims[0])/(xlims[1]-xlims[0]))] >= 0.999:
                sky_loc_area = np.nan
            imgm = cv2.resize(imgm, (int(a/5), int(b/5)),
                              interpolation=cv2.INTER_LANCZOS4)  # resizing
            a, b = imgm.shape
            imgm = np.ma.masked_where(
                imgm >= 0.999, imgm, copy=True)  # masking white area
            imgm = np.flip(imgm, axis=0).astype(floatX)
            sns.set()
            figm, axm = plt.subplots(subplot_kw={'projection': 'mollweide'})
            ras = np.linspace(xlims[0]*radian, xlims[1]*radian, a)
            # RA angles must lie between -180 and 180 degrees to be plotted.
            ras = np.arctan2(sin(ras), cos(ras))
            decs = np.linspace(ylims[0]*radian, ylims[1]*radian, b)
            ras = -array([ras[i % a] for i in range(a*b)])
            decs = array([decs[int(i/a)] for i in range(a*b)])
            imgm = imgm.flatten()
            # RA angles must lie between -180 and 180 degrees to be plotted.
            ra_plt = np.arctan2(sin(ra*radian), cos(ra*radian))/radian
            axm.scatter(array([-ra_plt*radian]), array([dec*radian]), s=50.0, color='deepskyblue',
                        marker='x', alpha=1.0, zorder=100, label='GW source position')
            axm.scatter(ras, decs, c=imgm, marker='.', s=0.1, linewidths=0.0,
                        cmap='spring', zorder=99, alpha=0.6,
                        # norm=mpl_colors.Normalize(vmin=0.0, vmax=1.0), alpha=1.0,
                        label="67%% HPD region\nLocalization area = %r deg$^2$" % (
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
                bbox_to_anchor=(0.725, 1.2),
                # labelcolor="deepskyblue",
                fontsize=14.5,
                prop={'size': 9.00}
            )

            # Corner plot of marginalized variables:
            if flag_cp == 'y':  # show individual subplots
                names = ["GW source mass #1 posterior [$M_{\odot}$]", "GW source mass #2 posterior [$M_{\odot}$]", "Initial binary separation posterior [$au$]", "GW source inclination posterior [$\degree$]",
                         "GW source redshift posterior", "GW initial phase posterior [$\degree$]", "GW source RA posterior [$\degree$]", "GW source DEC posterior [$\degree$]", "GW polarization posterior [$\degree$]"]
                for i in range(8):
                    for j in range(8-i):
                        ax_az = az.plot_pair(
                            posterior,
                            var_names=[
                                names[i],
                                names[j+i+1],
                            ],
                            kind="kde",
                            marginals=True,
                            figsize=(9.25, 9.25),
                            kde_kwargs={
                                # Plot HPD region (highest posterior density)
                                # for (x)% HPDR, the hdi_probs argument is x/100
                                "hdi_probs": [0.67],
                                "contour": True,
                                "contourf_kwargs": {"cmap": "Reds"},
                                "fill_last": False,
                                "label": "67%% HPD region",
                                "legend": True,
                            },
                        )
                        plt.tight_layout()
                        plt.tight_layout()
                        plt.tight_layout()
                        plt.savefig(
                            "saved_plots/cp_%r_%r.png" % (i+1, j+i+2),
                            format="png",
                            dpi=800,
                        )
                        plt.close()

            else:  # regular corner-plot
                az.plot_pair(
                    posterior,
                    kind='kde',
                    marginals=True,
                    textsize=8,
                )


# ----------------------------------------------------------------------------------------------------------------------------


# Scaled sky localization precision vs. GW source position sky map
if flag_srcpos_skyloc == "y":

    H = array([[37.64651429, 50.88035, 14.95589, 11.9545, 15.14886, 4.868622222, 4.457183333, 11.78771, 37.64651429, 50.88035, 14.95589, 11.9545, 15.14886, 4.868622222, 4.457183333, 11.78771, 37.64651429],
               [293.5616857, 171.56432, 81.05966, 59.26694, 98.99138, 21.6073, 11.10388889, 14.06457778, 33.05057,
                   16.0310875, 53.14833333, 20.1996625, 26.32324444, 108.3118333, 130.0436111, 75.90716, 293.5616857],
               [213.8694333, 102.36182, 34.32106, 144.1369556, 27.79148, 43.3066, 14.78962, 66.12642, 15.5045,
                   39.27969, 27.196, 91.332, 27.28316, 55.2306125, 124.0707889, 104.27043, 213.8694333],
               [54.4227, 29.28161429, 47.9090625, 84.74205, 80.26987778, 65.61522222, 61.34698, 18.20961,
                   62.0339, 33.85117, 76.18151, 24.84968889, 9.183511111, 94.4223125, 146.29374, 90.1001, 54.4227],
               [58.5141875, 29.96669, 102.64203, 41.22351111, 19.68511111, 10.11997143, 74.5084, 81.79991111, 58.5141875,
                   29.96669, 102.64203, 41.22351111, 19.68511111, 10.11997143, 74.5084, 81.79991111, 58.5141875],
               [62.0339, 33.85117, 76.18151, 24.84968889, 9.183511111, 94.4223125, 146.29374, 90.1001, 54.4227,
                   29.28161429, 47.9090625, 84.74205, 80.26987778, 65.61522222, 61.34698, 18.20961, 62.0339],
               [15.5045, 39.27969, 27.196, 91.332, 27.28316, 55.2306125, 124.0707889, 104.27043, 213.8694333,
                   102.36182, 34.32106, 144.1369556, 27.79148, 43.3066, 14.78962, 66.12642, 15.5045],
               [33.05057, 16.0310875, 53.14833333, 20.1996625, 26.32324444, 108.3118333, 130.0436111, 75.90716,
                   293.5616857, 171.56432, 81.05966, 59.26694, 98.99138, 21.6073, 11.10388889, 14.06457778, 33.05057],
               [37.64651429, 50.88035, 14.95589, 11.9545, 15.14886, 4.868622222, 4.457183333, 11.78771, 37.64651429,
                   50.88035, 14.95589, 11.9545, 15.14886, 4.868622222, 4.457183333, 11.78771, 37.64651429],
               ])

    H1 = H
    #H = gaussian_filter(H, sigma=18, mode='wrap')

    # RA angles must lie between -180 and 180 to be plotted
    ras = array([-180.0, -157.5, -135.0, -112.5, -90.0, -67.5, -45.0, -22.5,
                0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5, 180.0])
    # DEC angles must lie between 90 and -90 to be plotted
    decs = array([90.0, 67.5, 45.0, 22.5, 0.0, -22.5, -45.0, -67.5, -90.0])
    # 2d-interpolation
    H_func = interp2d(ras, decs, H, kind='cubic')

    ras = np.linspace(-180.0, 180.0, 10*360)
    decs = np.linspace(90.0, -90.0, 10*180)
    H = H_func(ras, decs)

    H = np.fliplr(H)
    #H = gaussian_filter(H, sigma=18, mode='wrap')
    # H = np.floor(np.log10(H))  # converting to order of magnitude
    H = (1.0 - H/41253.)  # sky area to precision conversion
    # linear activation/transfer function
    H = (H - np.min(H))/(np.max(H) - np.min(H))

    sns.set()
    figm, axm = plt.subplots(subplot_kw={'projection': 'mollweide'})
    bkg = axm.pcolormesh(-ras*radian, decs*radian, H,
                         cmap='cividis', alpha=1.0, zorder=97)
    figm.colorbar(bkg, orientation='vertical')
    axm.scatter(array([0.0*radian]), array([51.0*radian]), s=50.0, color='red',
                marker='x', alpha=0.9, zorder=100, label="SRGO initial position")
    # axm.plot(-np.linspace(-180*radian, 180*radian, 360), 51.0*radian*ones(360), color='red',
    #         linestyle='dashdot', alpha=0.5, zorder=98)
    axm.plot(-np.linspace(0.0*radian, 45.0*radian, 45), 51.0*radian*ones(45), color='red',
             linestyle='dashdot', alpha=0.9, zorder=99, label="SRGO track in 3.0 hrs")
    axm.plot(
        [], [], ' ', label="$\sigma_{effective}=0.529$ fs \nMean PSNR = 10 \nMean sky-loc. area = $%r \pm %r$ deg$^2$" % (np.round(np.mean(H1), 1), np.round(np.std(H1), 1)))

    tick_labels_x = ['10$^h$', '08$^h$', '06$^h$', '04$^h$', '02$^h$', '00$^h$',
                     '22$^h$', '20$^h$',  '18$^h$', '16$^h$', '14$^h$']
    axm.set_xticklabels(tick_labels_x)
    axm.set_xlabel("RA [hrs]")
    axm.xaxis.label.set_fontsize(12)
    axm.set_ylabel("DEC [$\degree$]")
    axm.yaxis.label.set_fontsize(12)
    axm.set_title(
        "Scaled sky-localization precision VS. source position")
    axm.legend(
        loc="upper right",
        bbox_to_anchor=(1.0, 1.05),
        # labelcolor="deepskyblue",
        fontsize=18,
        prop={'size': 10.00}
    )
    figm.canvas.manager.full_screen_toggle()
    figm.tight_layout()
    figm.tight_layout()
    figm.tight_layout()
    figm.savefig("saved_plots/skyloc_vs_srcpos.png", format="png", dpi=800)


# Sky localization area vs. PSNR plot:
if flag_psnr == "y":

    data = [
        [54.2974, 30.9326, 32.3975, 9.8388, 19.7761, 3.4194, 93.8582],
        [0.3839, 8.6111, 0.6604, 12.6581, 12.3016, 17.3632, 201.6091, ],
        [88.9723, 1.6421, 10.5339, 73.2712, 1.8327],
        [0.4611, 330.3427, 8.4976, 17.9202, 62.0821, 110.976,
            108.0756, 64.8699, 11.3995, 33.5356, ],
        [72.7121, 253.6095, 335.5866, 307.7102, 178.1994,
            169.7616, 63.2611, 751.4792, 11.585, ],
        [133.7488, 126.8111, 129.1958, 685.3148, 887.608, 350.0481],
        [454.2418, 383.4355, 773.23, 380.4481, ],
        [275.1915, 260.7243, 499.3266, 409.477,
            752.3838, 347.9836, 744.1954, 557.2729, 821.0532, ],
        [613.0696, 241.1615, 208.913, 909.4469, 1388.0782, ],
        [655.1943, 277.3036, 316.4839, 825.6289, 1592.4125, ],
        [850.8318, 516.5499, 635.0277, 540.5476, 1399.1057,
            603.4509, 673.038, 1221., 1761.7849, ],
    ]

    data2 = [
        [64.4095, 0.3651, 3.7, 15.9293, 4.1394, 60.9253, 27.6868, 2.8807, ],
        [0.0551, 2.0319, 57.6353, 1.1492, 0.3867, 13.6993, 20.1, 58.1378, 24.5401],
        [10.3384, 7.9591, 49.5154, 52.0727, 4.1733, 25.1845, ],
        [1.9359, 29.3951, 18.8862, 6.9267, 43.4034, ],
        [11.5411, 4.632, 160.8169, 8.8286, 68.4859, ],
        [176.7386, 470.3654, 206.6546, 221.4386, 3.8034, 245.043, 375.9964],
        [107.3379, 112.1709, 750.6679, 156.4678, 417.7573],
        [421.6446, 142.8398, 369.1796, 751.0536, 128.0552],
        [450.5255, 570.6638, 319.5606, 977.8645, 940.4872, 85.6063],
        [625.1592, 1294.1399, 361.6337, 400.69, 473.0975, 492.39],
        [600.3436, 1399.1493, 1076.9964, 1905.25, 124.939,
            468.771, 145.6531, 1529.1918, 530.474, ],
    ]

    data3 = [
        [0.5369, 20.1462, 0.0188, 19.842, 0.0777, 19.7282, 22.0763, 0.3137],
        [1.2829, 1.4752, 68.1812, 18.3513, 0.1567, 0.0494],
        [79.862, 0.0494, 23.7125, 0.0613, 0.0312, 0.0271, 0.4548],
        [53.3419, 0.036, 3.4267, 1.3863],
        [0.031, 0.8855, 2.1626, 56.3731, 2.1877,
            0.4853, 30.5116, 0.6725, 49.7102],
        [189.6986, 195.548, 3.7012, 546.8485, 358.0272, 3.9286],
        [639.846, 74.5202, 280.987, 167.5608, 83.7818],
        [455.3631, 213.0583, 232.918, 265.7279],
        [147.3303, 408.688, 619.9115, 264.4028, 209.7221],
        [678.0858, 119.6549],
        [823.4711, 694.6945, 221.1757, 931.8852, 797.7094],
    ]

    means = [np.mean(x) for x in data]
    devs = [np.std(x) for x in data]

    means2 = [np.mean(x) for x in data2]
    devs2 = [np.std(x) for x in data2]

    means3 = [np.mean(x) for x in data3]
    devs3 = [np.std(x) for x in data3]

    psnr = np.array([10., 8., 5., 3., 2., 1., 0.5, 0.333, 0.2, 0.125, 0.1])

    sns.reset_orig()
    plt.figure()
    plt.scatter(psnr, means, marker='P',
                label="16 data points over 1 day", linewidth=1.5, color='lightseagreen')
    plt.errorbar(psnr, means, yerr=devs, alpha=0.5,
                 linewidth=1.5, color='lightseagreen')
    plt.scatter(psnr, means2, marker='D',
                label="32 data points over 1 day", linewidth=1.5, color='orange')
    plt.errorbar(psnr, means2, yerr=devs2, alpha=0.5,
                 linewidth=1.5, color='orange')
    plt.scatter(psnr, means3, marker='v',
                label="64 data points over 1 day", linewidth=1.5, color='darkmagenta')
    plt.errorbar(psnr, means3, yerr=devs3, alpha=0.5,
                 linewidth=1.5, color='darkmagenta')
    # plt.ylim(bottom=0.1)
    plt.xlabel("PSNR (peak signal-to-noise ratio)")
    plt.ylabel("Sky localization area [deg$^2$]")
    plt.minorticks_on()
    plt.xticks([0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])
    # plt.yscale("log")
    plt.xscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig("saved_plots/psnr.png", format="png", dpi=800)


# Sky localization area vs. T_obs plot:
if flag_tobs == "y":

    data = [
        [127.8209, 622.3632, 371.8536, 47.6526, ],
        [139.1622, 218.8283, 36.5101, 97.2105],
        [31.7614, 4.8019, 35.8134, 194.0278, 13.4596, 56.,
            136.5, 4.5058, 13.5092, 129.9437, 4.2936],
        [20.6947, 6.4195, 3.0848, 153.1319, 3.6901,
            6.2226, 2.8403, 80., 80., 80.],
    ]

    data2 = [
        [59.2455, 54.1445, 21.1558, 103.1234, 161.6895, 89.2977],
        [45.9196, 1.1095, 4.115, 150.2512, 7.5419, 140.1806, 151.4658, ],
        [109.7957, 10.54, 4.4776, 15.6679, 3.4296,
            3.2233, 53.0034, 91.4887, 3.3405, 30.7084, 10., 59.782, 5.6194, 2.0819, 10.7899, 14.9327, 1.327, 19.7461, ],
        [2.9428, 6.3762, 88.3463, 1.4873, 81.2556, 2.6576,
            1.1546, 0.8567, 30.6002, 53.699, 2.],
    ]

    data3 = [
        [143.4535, 503.1823, 854.1078, 297.7498, 283.7162, 226.1793, 378.1311],
        [143.0624, 107.6629, 303.0361],
        [202.0586, 219.7042, 151.5069,
            281.292, 119.5193, 12.0287, 219.7042, 84.32, 100.4391, 165.3476, 219.3187, 156.243, 21.3157, ],
        [16.2549, 42.2138, 324.6406, 270.9527,
            159.9031, 30.0849, 98.4113, 249.4565, 77.69, ],
    ]

    means = [np.mean(x) for x in data]
    devs = [np.std(x) for x in data]

    means2 = [np.mean(x) for x in data2]
    devs2 = [np.std(x) for x in data2]

    means3 = [np.mean(x) for x in data3]
    devs3 = [np.std(x) for x in data3]

    tobs = np.array([3.0, 6.0, 12.0, 24.0])

    sns.reset_orig()
    plt.figure()
    plt.scatter(tobs, means3, marker='o',
                label="PSNR = 0.333", linewidth=1.5, color='black')
    plt.errorbar(tobs, means3, yerr=devs3, alpha=0.5,
                 linewidth=1.5, color='black')
    plt.scatter(tobs, means, marker='v',
                label="PSNR = 1", linewidth=1.5, color='red')
    plt.errorbar(tobs, means, yerr=devs, alpha=0.5,
                 linewidth=1.5, color='red')
    plt.scatter(tobs, means2, marker='s',
                label="PSNR = 3", linewidth=1.5, color='gold')
    plt.errorbar(tobs, means2, yerr=devs2, alpha=0.5,
                 linewidth=1.5, color='gold')
    plt.xlim(0.0, 27.0)
    plt.xlabel(
        "Observation time [hrs]\n($f_{sample}=2.666$ data points per hour)")
    plt.ylabel("Sky localization area [deg$^2$]")
    plt.minorticks_on()
    plt.xticks([0., 3., 6., 9., 12., 15., 18., 21., 24, 27.])
    # plt.yscale("log")
    # plt.xscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig("saved_plots/tobs.png", format="png", dpi=800)


# Relative error of mass estimation vs. PSNR plot:
if flag_mass == "y":

    data = [
        [0.000419, 0.0258, 0.05224, 0.01875, 0.03125, 0.028125,
            0.0, 0.015625, 0.000645, 0.01846, 0.0],
        [0.0019, 0.0424, 0.025, 0.01953, 0.021875, 0.025, 0.03,
            3.125e-05, 0.000667, 0.000625, 0.002],
        [0.11, 0.0125, 0.015625, 0.00106, 0.0, 0.025, 0.00125, ],
        [0.0025, 0.0375, 0.0, 0.125, 0.05, 0.0375, 0.05, 0.0, 0.025, 0.05, ],
        [0.094, 0.05, 0.025, 0.00625, 0.05, 0.05, 0.025, 0.05, 0.00625, 0.05, ],
        [0.05, 0.05, 0.08, 0.025, 0.0625, 0.1, 0.0625, 0.0625, 0.025],
        [0.125, 0.05, 0.125, 0.05, 0.06, 0.05, 0.025, 0.05, 0.025, 0.05],
        [0.05, 0.0625, 0.1, 0.1, 0.05, 0.075, 0.05, 0.075, 0.075, 0.05, ],
        [0.075, 0.1, 0.078125, 0.15625, 0.046875, 0.0625, 0.0625, 0.05, 0.125],
        [0.1, 0.125, 0.09375, 0.11, 0.09375, 0.11, 0.0875, 0.140625, ],
        [0.1, 0.125, 0.15, 0.14, 0.09375, 0.11, 0.125, 0.1375, 0.125],
    ]

    data2 = [
        [0.0, 0.025, 0.015, 0.046875, 0.00375],
        [0.008, 0.0375, 0.0364, 0.02424, 0.00125, 0.025, 0.000625],
        [0.00484, 0.03226, 0.03125, 0.021875, 0.0],
        [0.0375, 0.0129, 0.0375, 0.0387, 0.0258, 0.05, 0.005],
        [0.05, 0.04, 0.0375, 0.00125, 0.07576, 0.0125, 0.02424, ],
        [0.04242, 0.0625, 0.06451, 0.0258,
            0.06154, 0.0588, 0.009677, 0.0375, 0.046875, ],
        [0.0485, 0.0546875, 0.06452, 0.0258, 0.0258, 0.0258, 0.0597, 0.09375],
        [0.0258, 0.0387, 0.0806, 0.06452, 0.09375,
            0.058, 0.0625, 0.0412, 0.080645, 0.0984615],
        [0.09375, 0.0625, 0.08088, 0.07164, 0.0606,
            0.0909, 0.07647, 0.09333],
        [0.10588, 0.096774, 0.1129, 0.096774,
            0.08065, 0.14, 0.14516, 0.06, 0.1167, ],
        [0.103225, 0.07576, 0.11765, 0.125, 0.125,
            0.089, 0.1212, 0.1129, 0.0836, 0.15],
    ]

    data3 = [
        [0.0003125, 0.04546, 0.0003, 0.0396825,
            0.000452, 0.0, 0.0003226, 0.03125, 0.001129],
        [0.04375, 0.025, 0.03226, 0.0129, 0.0009375, 0.02424, 0.0022, ],
        [0.00091, 0.046875, 0.04762, 0.0011, 0.00968,
            0.0036923, 0.0002424, 0.00036923, 0.04516, 0.00088],
        [0.0588, 0.046875, 0.001613, 0.00125,
            0.01791, 0.0006, 0.01945, 0.039, ],
        [0.009375, 0.0267, 0.00375, 0.00788, 0.0294, 0.021],
        [0.0546875, 0.0129, 0.009677, 0.0379, 0.04167, ],
        [0.053, 0.0424, 0.00923, 0.0273, 0.025, 0.0546875, 0.046875, 0.04297, 0.0],
        [0.0375, 0.0375, 0.01875, 0.05, 0.0258, 0.00182,
            0.04375, 0.078125, 0.0258, 0.0546875],
        [0.025, 0.004375, 0.0534, 0.0546875, 0.05625,
            0.07143, 0.0369, 0.1, 0.0606, 0.09375],
        [0.1061, 0.09375, 0.06061, 0.0588, 0.023,
            0.129, 0.025, 0.0662, 0.05, 0.077],
        [0.109375, 0.1125, 0.06875, 0.103, 0.078125,
            0.0834, 0.061, 0.05, 0.0353, 0.0588],
    ]

    means = [np.mean(x)*100 for x in data]
    devs = [np.std(x)*100 for x in data]

    means2 = [np.mean(x)*100 for x in data2]
    devs2 = [np.std(x)*100 for x in data2]

    means3 = [np.mean(x)*100 for x in data3]
    devs3 = [np.std(x)*100 for x in data3]

    psnr = np.array([10., 8., 5., 3., 2., 1., 0.5, 0.333, 0.2, 0.125, 0.1])

    sns.reset_orig()
    plt.figure()
    plt.scatter(psnr, means, marker='o',
                label="16 data points over 1 day", linewidth=1.5, color='rosybrown')
    plt.errorbar(psnr, means, yerr=devs, alpha=0.5,
                 linewidth=1.5, color='rosybrown')
    plt.scatter(psnr, means2, marker='p',
                label="32 data points over 1 day", linewidth=1.5, color='slategray')
    plt.errorbar(psnr, means2, yerr=devs2, alpha=0.5,
                 linewidth=1.5, color='slategray')
    plt.scatter(psnr, means3, marker='d',
                label="64 data points over 1 day", linewidth=1.5, color='darkseagreen')
    plt.errorbar(psnr, means3, yerr=devs3, alpha=0.5,
                 linewidth=1.5, color='darkseagreen')

    # plt.ylim(bottom=0.1)
    plt.xlabel("PSNR (peak signal-to-noise ratio)")
    plt.ylabel("Relative error of mass estimation (%)")
    plt.minorticks_on()
    plt.xticks([0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])
    # plt.yscale("log")
    plt.xscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig("saved_plots/mass.png", format="png", dpi=800)


# Relative error of distance estimation vs. PSNR plot:
if flag_dist == "y":

    data = [
        [0.00294, 0.00044776, 0.007273,
            0.005, 0.00296, 0.00328, 0.00353, 0.004583],
        [0.0047, 0.00521, 0.0053, 0.004348, 0.00457,
            0.002, 0.004546, 6.67e-05, 0.00079, 0.0013, ],
        [0.002353, 0.00515, 0.00167, 0.0059, 0.0041, 0.0039, 0.000734, ],
        [0.004932, 0.0152, 0.00123, 0.00554, 0.008125,
            0.007, 0.0071, 0.00432, 0.00564, 0.0067],
        [0.0137, 0.015625, 0.00647, 0.002687, 0.0175,
            0.0071, 0.0142857, 0.02258, 0.00742, 0.007742, ],
        [0.00742, 0.012, 0.0067, 0.0194, 0.01364, 0.0212, 0.01],
        [0.01855, 0.0139, 0.02643, 0.0134, 0.01484,
            0.02676, 0.0182, 0.01167, 0.0046875, 0.00452],
        [0.0108, 0.01458, 0.0131, 0.021875, 0.0168,
            0.01, 0.023, 0.0177, ],
        [0.014, 0.0234, 0.022, 0.01, 0.0169, 0.00735, 0.0158, 0.0167, 0.0267, ],
        [0.0172, 0.0125, 0.011, 0.016, 0.03, 0.01875, ],
        [0.0265, 0.01167, 0.0147, 0.01765, 0.02, ]
    ]

    data2 = [
        [0.0021, 0.0044, 0.0057, 0.0032, 0.0006],
        [0.000235, 0.0036, 0.0009,  0.00028, 0.0081, 0.005, 0.00426],
        [0.0009934, 0.00222, 0.00455, 0.00444, 0.0016, ],
        [0.0091, 0.0046875, 0.00353, 0.00588, 0.0025, 0.0044, 0.00314],
        [0.0144, 0.01458, 0.00424, 0.0075, 0.00647, 0.00742, ],
        [0.0067, 0.0123, 0.01765, 0.01143, 0.0075,
            0.01985, 0.00035, 0.0004, 0.011, 0.0164],
        [0.007353, 0.021, 0.015, 0.0095, 0.0089,
            0.0075, 0.0056, 0.0052, 0.015],
        [0.0045, 0.01797, 0.0162, 0.0156, 0.0206,
            0.0125, 0.01212, 0.009375, 0.011, ],
        [],
        [],
        []
    ]

    data3 = [
        [0.011764, 0.00041, 0.00645, 0.00052, 0.00533,
            0.000364, 0.000758, 0.0034375, 0.00046875],
        [0.00025, 0.0094, 0.00733, 0.0029, 0.00288,
            0.000355, 0.0006875, 0.00057, 0.0015],
        [0.004375, 0.01167, 0.00033, 0.0022, 0.000182,
            0.000212, 9.375e-05, 0.0078125, 0.00140625, ],
        [0.00588, 0.00154, 0.00734, 0.006, 0.0015625, 0.0025, 0.00588],
        [0.001143, 0.003824, 0.0026, 0.00485, 0.00234,
            0.00284, 0.0103, 0.005, 0.005224, ],
        [0.0113, 0.0061, 0.0056, 0.00444, 0.01136,
            0.0087, 0.0147, 0.0097, ],
        [0.0175, 0.0125, 0.0039, 0.001136, 0.0061,
            0.01617, 0.0125, 0.0121, 0.006875, ],
        [0.0176, 0.014, 0.0105, 0.0066, 0.0047, 0.01, 0.026, 0.0097],
        [],
        [],
        []
    ]

    means = [(cosmo.luminosity_distance(Z+np.mean(x)).value - cosmo.luminosity_distance(Z -
              np.mean(x)).value)*100/(2*cosmo.luminosity_distance(Z).value) for x in data]
    devs = [(cosmo.luminosity_distance(Z+np.mean(x)+np.std(x)).value - cosmo.luminosity_distance(Z+np.mean(x)).value + cosmo.luminosity_distance(Z -
             np.mean(x)).value - cosmo.luminosity_distance(Z-np.mean(x)-np.std(x)).value)*100/(2*cosmo.luminosity_distance(Z).value) for x in data]

    means2 = [(cosmo.luminosity_distance(Z+np.mean(x)).value - cosmo.luminosity_distance(Z -
              np.mean(x)).value)*100/(2*cosmo.luminosity_distance(Z).value) for x in data2]
    devs2 = [(cosmo.luminosity_distance(Z+np.mean(x)+np.std(x)).value - cosmo.luminosity_distance(Z+np.mean(x)).value + cosmo.luminosity_distance(Z -
             np.mean(x)).value - cosmo.luminosity_distance(Z-np.mean(x)-np.std(x)).value)*100/(2*cosmo.luminosity_distance(Z).value) for x in data2]

    means3 = [(cosmo.luminosity_distance(Z+np.mean(x)).value - cosmo.luminosity_distance(Z -
              np.mean(x)).value)*100/(2*cosmo.luminosity_distance(Z).value) for x in data3]
    devs3 = [(cosmo.luminosity_distance(Z+np.mean(x)+np.std(x)).value - cosmo.luminosity_distance(Z+np.mean(x)).value + cosmo.luminosity_distance(Z -
             np.mean(x)).value - cosmo.luminosity_distance(Z-np.mean(x)-np.std(x)).value)*100/(2*cosmo.luminosity_distance(Z).value) for x in data3]

    psnr = np.array([10., 8., 5., 3., 2., 1., 0.5, 0.333, 0.2, 0.125, 0.1])

    sns.reset_orig()
    plt.figure()
    plt.scatter(psnr, means, marker='*',
                label="16 data points over 1 day", linewidth=1.5, color='violet')
    plt.errorbar(psnr, means, yerr=devs, alpha=0.4,
                 linewidth=1.5, color='violet')
    plt.scatter(psnr, means2, marker='o',
                label="32 data points over 1 day", linewidth=1.5, color='mediumpurple')
    plt.errorbar(psnr, means2, yerr=devs2, alpha=0.4,
                 linewidth=1.5, color='slateblue')
    plt.scatter(psnr, means3, marker='v',
                label="64 data points over 1 day", linewidth=1.5, color='skyblue')
    plt.errorbar(psnr, means3, yerr=devs3, alpha=0.4,
                 linewidth=1.5, color='cornflowerblue')

    # plt.ylim(bottom=0.1)
    plt.xlabel("PSNR (peak signal-to-noise ratio)")
    plt.ylabel("Relative error of distance estimation (%)")
    plt.minorticks_on()
    plt.xticks([0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])
    # plt.yscale("log")
    plt.xscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig("saved_plots/dist.png", format="png", dpi=800)


# ----------------------------------------------------------------------------------------------------------------------------


if flag_sensi != "y" and flag_err != "y" and flag_range != "y" and flag_srcpos_skyloc != "y" and flag_psnr != "y" and flag_tobs != "y" and flag_mass != "y" and flag_dist != "y":

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
        dpi=800,
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
        m_n = 1.1 * min(min(noisy_signal), min(signal_plt))
        m_p = (1.1 * max(max(noisy_signal), max(signal_plt)) -
               0.451 * m_n) / (1.0 - 0.451)
    else:
        m_n = 1.1 * min(signal_plt)
        m_p = (1.1 * max(signal_plt) - 0.451 * m_n) / (1.0 - 0.451)
    ax.axvline(x=day / 3600, linewidth=0.95,
               color="orange", label="Sidereal day")
    ax.axvline(x=2.0 * day / 3600, linewidth=0.95, color="orange")
    ax.axvline(x=3.0 * day / 3600, linewidth=0.95, color="orange")
    ax.set_ylim(m_n, m_p)
    tick = 1.0
    ax.set_xlabel(
        "Observation Time [hrs] \n1 unit = %r hrs" % tick, fontweight="bold"
    )
    ax.set_ylabel("$\mathbf{\Delta T_{GW}}$ [s]", fontweight="bold")
    ax.text(
        27.0,
        m_p + (4.23e-15 - 6.4e-15) *
        ((m_p - m_n)/(6.4e-15 + 2.8e-15)),
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
            np.log10(M1),
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
        ax_sg.set_ylim(0.0, 0.0025)
        wd, ht = fig_sg.get_size_inches()
        fig_sg.set_size_inches(2.*wd, ht)

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
        if flag_cp == "y":
            if flag_extras == "y":
                plotname = ["trace", "sky_map",
                            "signal", "spectrogram", "angles"]
            else:
                plotname = ["trace", "sky_map",
                            "signal"]
        else:
            if flag_extras == "y":
                plotname = ["trace", "sky_map", "corner_plot",
                            "signal", "spectrogram", "angles"]
            else:
                plotname = ["trace", "sky_map", "corner_plot",
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
        fig.tight_layout()
        fig.tight_layout()
        fig.savefig(directory + plotname[f_num] +
                    ".png", format="png", dpi=800)
        f_num += 1
        plt.close(fig)

    # Saving parameter estimation data:
    file = open("saved_plots/data.txt", "w")
    file.write("sky_loc_area:" + str(sky_loc_area) + "\n" +
               "rel_m_err:" + str(rel_m_err) + "\n" + "rel_d_err:" + str(rel_d_err))
    file.close
