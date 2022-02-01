""" Simulating the detection of millihertz (mHz) gravitational waves (GWs)
    from astrophysical sources by a Storage Ring Gravitational-wave Observatory (SRGO).
    Author: Suvrat Rao, Hamburg Observatory, University of Hamburg """

import numpy as np
from matplotlib import ticker as mpl_ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from scipy.spatial.transform import Rotation as R
from scipy.signal import spectrogram
#from scipy.signal import wiener
from scipy.integrate import simps
from scipy.interpolate import interp1d
import pymc3 as pm
import arviz as az
import theano.tensor as tt
#from theano import config
#import os

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
#os.environ["MKL_NUM_THREADS"] = "5"

# Constants in SI units:
G = 6.67430 * 10 ** (-11)  # gravitational constant
c = 299792458.0  # speed of light
pc = 3.085677581 * 10 ** 16  # parsec
au = 1.495978707 * 10 ** 11  # astronomical unit
msol = 1.98847 * 10 ** 30  # solar mass
day = 86164.0905  # sidereal Earth day
radian = pi / 180.0  # degree to radian
we = 7.2921150 * 10 ** (-5)  # angular speed of Earth's rotation in radian/s.


# ---------------------------------------------------------------------------------------------------------
# "DASHBOARD" for user input parameters:
"""Cannot use input() function since it gives EOFError when MCMC scans the code for global variables."""
# GW SOURCE PARAMETERS:
m1 = 1000000.0 * msol  # mass of object #1 in solar masses
m2 = 1000000.0 * msol  # mass of object #2 in solar masses
r = 0.55 * au  # initial separation between bodies in au
# GW SOURCE-OBSERVER PARAMETERS:
inc = 0.0  # inclination angle of observer w.r.t binary system in degrees
# luminosity distance of the binary system in parsec
d = 987.7 * (10.0 ** 6) * pc
Z = 0.2  # redshift of the binary system
phase = 0.0  # GW signal initial phase in degrees
# local sidereal time of SRGO in hh:mm:ss at the start of the observation run
LST = (00.0, 00.0, 00.0)
RA = (00.0, 00.0, 00.0)  # GW source right ascension in hh:mm:ss
dec = 0.0  # GW source declination in degrees
# GW polarization angle (in degrees) as measured in equatorial celestial coordinates
psi_eq = 0.0
# STORAGE RING PARAMETERS:
v0 = 0.70 * c  # SRGO ion bunch speed
L = 100.0  # SRGO circumference in meters
n_p = 1  # SRGO number of bunches
# MCMC PARAMETERS AND FLAGS:
# input("Compute sky localization with MCMC using synthetic data? ([y]/n)\n")
flag_mcmc = "y"
T_obs = 1.0 * day  # duration of SRGO observation run
N = 32  # no. of SRGO data points acquired during T_obs
psnr = 20.0  # peak signal to noise ratio
# "ligo" (Whittle likelihood in Fourier space) or "custom" likelihood functions for MCMC
likelihood_type = "ligo"
flag_show_pts = "y"  # show synthetic noisy data-points in signal plot ([y]/n)
# input("Plot evolution of orientation angles & numerical error vs. no. of data points? ([y]/n)\n")
flag_extras = "y"
flag_sensi = "n"  # numerically find SRGO sensitivity curve ([y]/n)
flag_err = "n"  # find numerical integration error ([y]/n)
# ---------------------------------------------------------------------------------------------------------


# GW waveform:
""" Dominant harmonic of inspiral phase GWs from non-spinning binaries, modelled via post-Newtonian theory.
    Approximation of LSCO (Last Stable Circular Orbit) used for the end of the inspiral phase. """
M = (1 + Z) * ((m1 * m2) ** (3.0 / 5.0)) / \
    ((m1 + m2) ** (1.0 / 5.0))  # chirp mass
Rf = max(
    (6.0 * G * m1 / c ** 2.0), (6.0 * G * m2 / c ** 2.0)
)  # final separation between bodies @ LSCO
f0 = sqrt(G * (m1 + m2)) / (2.0 * pi * r **
                            (3.0 / 2.0))  # initial GW frequency
f1 = sqrt(G * (m1 + m2)) / (
    2.0 * pi * Rf ** (3.0 / 2.0)
)  # final GW frequency at the end of inspiral phase
k = (96.0 / 5.0) * (pi ** (8.0 / 3.0)) * \
    ((G * M * (1 + Z)**-1 / c ** 3.0) ** (5.0 / 3.0))
t_ins = (
    (3.0 / 8.0) * (1.0 / k) * (f0 ** (-8.0 / 3.0) - f1 ** (-8.0 / 3.0))
)  # inspiral time
# t_mrg = ( 5./256. )*( c**5/G**3 )*( r**4/((m1*m2)*(m1+m2)) )  #time to merger


def f(t, f0):
    """GW frequency."""
    return (
        (1.0 / (1.0 + Z)) * (f0 ** (-8.0 / 3.0) -
                             (8.0 / 3.0) * k * t) ** (-3.0 / 8.0)
    ).astype(floatX)
    # return ((1./(1.+Z))* f0).astype(floatX)  #constant GW frequency => continuous wave case


def h_plus(t, inc, phase, d, f0):
    """GW plus-polarization."""
    inc *= radian
    phase *= radian
    return (
        4.0 / d
        * (G * M / c ** 2) ** (5.0 / 3.0)
        * ((pi * f(t, f0) / c) ** (2.0 / 3.0)).astype(floatX)
        * ((1.0 + cos(inc) ** 2.0) / 2.0).astype(floatX)
        * cos(2 * pi * f(t, f0) * t + phase).astype(floatX)
    ).astype(floatX)


def h_cross(t, inc, phase, d, f0):
    """GW cross-polarization."""
    inc *= radian
    phase *= radian
    return (
        4.0 / d
        * (G * M / c ** 2) ** (5.0 / 3.0)
        * ((pi * f(t, f0) / c) ** (2.0 / 3.0)).astype(floatX)
        * cos(inc).astype(floatX)
        * sin(2 * pi * f(t, f0) * t + phase).astype(floatX)
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


def time(n, t_obs):
    """"Return the corrected timing data points allowed by the experiment measurement."""
    # observation period cannot exceed inspiral period in our code
    t_obs = min(t_obs, t_ins)
    n = min(n, int(t_obs / T_sample))  # allowed N
    # Nyquist–Shannon sampling theorem
    n = max(n, int(2.0*f(t_obs, f0)*t_obs)+1)
    if flag_mcmc == "y":
        # to be able to use FFT trick for computing speed
        n = findPrevPowerOf2(n)
        if n < int(2.0*f(t_obs, f0)*t_obs)+1:
            n = 2*n
    # allowed timing interval
    T_timing = int(t_obs / (n * T_sample)) * T_sample
    t = arange(0.0, t_obs, T_timing, dtype=floatX)  # timing data points
    if flag_mcmc == "y" and len(t) > n:
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


def h_eff(t, phi, theta, psi, inc, phase, d, f0):
    """Effective longitudinal GW strain for SRGO test masses
    that are circulating rapidly compared to the GW frequency."""
    return (
        -(1.0 / 2.0)
        * (
            h_plus(t, inc, phase, d, f0) * F_plus(phi, theta, psi)
            + h_cross(t, inc, phase, d, f0) * F_cross(phi, theta, psi)
        )
    )


def boole_quad(y, x, yofx, a, b, c, d, e, g, k):
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
        + 32.0 * yofx(x[0] + h, a1, b1, c1, d, e, g, k)
        + 12.0 * yofx(x[0] + 2.0 * h, a2, b2, c2, d, e, g, k)
        + 32.0 * yofx(x[0] + 3.0 * h, a3, b3, c3, d, e, g, k)
        + 7.0 * y[1]
    )


def SRGO_signal(t, phi, theta, psi, inc, phase, d, f0):
    """SRGO signal from response to GWs."""
    integrand = h_eff(t, phi, theta, psi, inc, phase, d, f0)

    signal = [0.0]
    append = signal.append  # functional programming optimization

    gen = (                 # generator comprehension
        append(
            signal[-1]
            - (1.0 - v0 ** 2.0 / (2.0 * c ** 2.0))
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
                f0
            )
        )
        for i in range(len(t) - 1)
    )

    dummy = list(gen)

    return array(signal)


# Some quantities for plotting:
t_plt, dt_plt = time(20000, 3.0*day)
phi_plt, theta_plt, psi_plt = earth_rot(t_plt, ra, lst, dec, psi_eq, "regular")
signal_plt = SRGO_signal(t_plt, phi_plt, theta_plt, psi_plt, inc, phase, d, f0)


def sensi_curve(d):
    """For numerically obtaining the SRGO sensitivity curve."""

    sigma_noise = 10**(-12)
    f0 = sqrt(G * (m1 + m2)) / (2.0 * pi * r ** (3.0 / 2.0))
    phi_sensi, theta_sensi, psi_sensi = earth_rot(
        t_plt, ra, lst, dec, psi_eq, "regular")
    signal_sensi = SRGO_signal(
        t_plt, phi_sensi, theta_sensi, psi_sensi, inc, phase, d, f0)
    val_lhs = (simps(signal_sensi**2.0, x=t_plt) / t_plt[-1])**0.5
    val_rhs = 2.0 * sigma_noise / ((1.0/T_sample)*t_plt[-1])**0.5
    #global d
    d = d*val_lhs/val_rhs
    return(f(0.0, f0), max(abs(h_plus(t_plt, inc, phase, d, f0))))


if flag_sensi == "y":

    ra_vals = np.linspace(0.0, 360.0, 6)
    dec_vals = np.linspace(-90.0, 90.0, 6)
    psi_eq_vals = np.linspace(0.0, 180.0, 6)
    inc_vals = np.linspace(0.0, 180.0, 6)
    phase_vals = np.linspace(0.0, 360.0, 6)
    distance = np.linspace(0.000127, 0.27, 50) * au

    f0_array = np.zeros(len(distance))
    h0_array = np.zeros(len(distance))
    h0_temp = np.zeros(len(distance))

    for aaa in ra_vals:
        for bbb in dec_vals:
            for ccc in psi_eq_vals:
                for ddd in inc_vals:
                    for eee in phase_vals:
                        for i in range(len(distance)):
                            ra = aaa
                            dec = bbb
                            psi_eq = ccc
                            inc = ddd
                            phase = eee
                            r = distance[i]
                            f_0, h_0 = sensi_curve(d)
                            print(f_0, h_0)
                            if f0_array[i] == 0.0:
                                f0_array[i] = f_0
                            h0_temp[i] = h_0

                        h0_array += h0_temp

    h0_array *= 1.0/(len(ra_vals)*len(dec_vals) *
                     len(psi_eq_vals)*len(inc_vals)*len(phase_vals))

    print(f0_array, h0_array)


def error(l):
    """For manually calculating the numerical error of integration given the no. of data pts, 'l'. """
    signal_ref = interp1d(t_plt, signal_plt)

    t_err, dt_err = time(l, 1.0*day)
    phi_err, theta_err, psi_err = earth_rot(
        t_err, ra, lst, dec, psi_eq, "regular")
    signal_err = SRGO_signal(t_err, phi_err, theta_err,
                             psi_err, inc, phase, d, f0)

    sampling_rate = l/(1.0*day)
    err = max(abs(signal_err - signal_ref(t_err)))

    print(
        "Sampling rate [s^-1] ; numerical intg. err. [s] =\n%r, %r\n\n" % (sampling_rate, err))


if flag_err == 'y':
    error(2**1)
    error(2**2)
    error(2**3)
    error(2**4)
    error(2**5)
    error(2**6)
    error(2**7)
    error(2**8)
    error(2**9)
    error(2**10)
    error(2**11)
    error(2**12)
    error(2**13)
    error(2**14)
    error(2**15)
    # error(2**16)
    # error(2**17)
    # error(2**18)
    # error(2**19)
    # error(2**20)


# Creating synthetic signal and adding only stochastic noise:
t, dt = time(N, T_obs)
N = len(t)
phi, theta, psi = earth_rot(t, ra, lst, dec, psi_eq, "regular")
signal = SRGO_signal(t, phi, theta, psi, inc, phase, d, f0)
noise = np.random.normal(0.0, scale=max(
    abs(signal_plt)) / psnr, size=N).astype(floatX)
noisy_signal = (signal + noise).astype(floatX)
# noisy_signal = wiener(noisy_signal, noise=max(
#   abs(signal_plt)) / psnr)  # Wiener noise filter
# Noise filter introduces new correlations between the independent data-points.
# Then we must account for the covariance matrix in the mcmc.
# https://www.researchgate.net/post/Does_using_a_noise_filter_before_performing_MCMC_fitting_of_a_model_to_noisy_data_increase_or_decrease_the_accuracy_of_the_fit


# Simulating noisy data and computing sky localization using MCMC:
if flag_mcmc == "y":

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
        ra_prior = pm.Uniform(  # pm.Bound(pm.Normal, lower=0.0, upper=360.0)(
            "GW source RA posterior [$\degree$]", lower=ra-180.0, upper=ra+180.0
        )
        dec_prior = pm.Uniform(  # pm.Bound(pm.Normal, lower=-90.0, upper=90.0)(
            "GW source DEC posterior [$\degree$]", lower=dec-180.0, upper=dec+180.0
        )
        psi_eq_prior = psi_eq  # pm.Uniform(  # pm.Bound(pm.Normal, lower=0.0, upper=180.0)(
        #    "GW polarization posterior [$\degree$]", lower=psi_eq-180.0, upper=psi_eq+180.0
        # )
        phase_prior = phase  # pm.Uniform(  # pm.Bound(pm.Normal, lower=0.0, upper=360.0)(
        #    "GW initial phase posterior [$\degree$]", lower=phase-180.0, upper=phase+180.0
        # )  # Choose a "weakly-informative" prior instead of a flat prior.
        # https://discourse.pymc.io/t/improving-model-convergence-and-sampling-speed/279/2

        # Model:
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
                SRGO_signal(t, phim, thetam, psim, inc, phase_prior, d, f0)
            )) / denominator
            model = (
                model ** (model ** 0.001) - 1.0 +
                abs(model ** (model ** 0.001) - 1.0)
            ) * ones(N, dtype=floatX)
            model = model.reshape(N, 1).astype(floatX)
        elif likelihood_type == "ligo":
            model = SRGO_signal(t, phim, thetam, psim, inc, phase_prior, d, f0)
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

            az.plot_trace(
                posterior,
                kind="trace",
                legend=False,
                rug=True,
                compact=True,
                combined=True,
            )

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

            # Plotting sky localization in Mollweide projection:
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


if flag_sensi == "n" and flag_err == "n":

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
        "SRGO latitude $=$ %r$\degree$ \nSRGO initial local sidereal time $=$ %r$^h$ %r$^m$ %r$^s$ \nGW source right ascension $=$ %r$^h$ %r$^m$ %r$^s$ \nGW source declination $=$ %r$\degree$ \nGW polarization angle $=$ %r$\degree$ \nGW initial phase $=$ %r$\degree$ \nEqual mass (~$10^{%d} M_{\odot}$) SMBBH at $z=%r$ \nEnd of inspiral phase at %r hrs \nObserver-SMBBH inclination angle $=$ %r$\degree$"
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
            np.round(t_ins / 3600.0, decimals=1),
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

    # Signal spectrogram:
    fig_sg, ax_sg = plt.subplots(1, 1)
    f_sg, t_sg, Sxx = spectrogram(signal_plt, 1.0/dt_plt)
    im = ax_sg.pcolormesh(t_sg / 3600.0, f_sg, Sxx,
                          shading='gouraud', cmap='viridis')
    fig_sg.colorbar(im, ax=ax_sg)
    ax_sg.set_ylabel("Frequency [Hz]")
    ax_sg.set_xlabel("Time [hrs] \n1 unit = %r hrs" % tick)
    ax_sg.xaxis.set_major_locator(mpl_ticker.MultipleLocator(6))
    ax_sg.xaxis.set_major_formatter(mpl_ticker.FormatStrFormatter("%d"))
    ax_sg.xaxis.set_minor_locator(mpl_ticker.MultipleLocator(tick))
    ax_sg.xaxis.set_ticks_position("bottom")
    ax_sg.yaxis.set_ticks_position("left")
    ax_sg.set_ylim(0.0, 0.002)

    # Extra plots:
    if flag_extras == "y":
        fig1, ax1 = plt.subplots(1, 1)
        ax1.plot(t_plt / 3600.0, phi_plt / radian, label="$\phi$")
        ax1.plot(t_plt / 3600.0, theta_plt / radian, label="$\Theta$")
        ax1.plot(t_plt / 3600.0, psi_plt / radian, label="$\psi$")
        ax1.set_xlabel("Time [hrs] \n1 unit = %r hrs" %
                       tick, fontweight="bold")
        ax1.set_ylabel("Angle [$\degree$]", fontweight="bold")
        ax1.xaxis.set_major_locator(mpl_ticker.MultipleLocator(6))
        ax1.xaxis.set_major_formatter(mpl_ticker.FormatStrFormatter("%d"))
        ax1.yaxis.set_major_locator(mpl_ticker.MultipleLocator(30))
        ax1.yaxis.set_major_formatter(mpl_ticker.FormatStrFormatter("%d"))
        ax1.xaxis.set_minor_locator(mpl_ticker.MultipleLocator(tick))
        # ax1.yaxis.set_minor_locator(mpl_ticker.AutoMinorLocator())
        plt.grid(which="minor")
        # ax1.plot(t_plt/3600., f(t_plt), label="GW frequency")
        # ax1.plot(t_plt/3600., h_plus(t_plt), label="GW strain")
        # ax1.plot(t_plt/3600., h_eff(t_plt, phi_plt, theta_plt, psi_plt, inc, phase), label="Effective GW strain")
        ax1.legend(loc="best")

        sns.reset_orig()
        fig2, axes1 = plt.subplots(1, 1)
        pts = 2.**arange(1, 20, 1)
        f_samprates = pts/day
        errs = [3.4327206112105946e-14, 8.558974137347506e-15, 6.439777565091142e-15, 6.1885055061462055e-16, 9.234550377265257e-18, 1.7642185433193133e-18, 4.2554435610589642e-19, 1.1182092039457022e-19, 2.78232977667702e-20,
                9.867617104329e-21, 5.186720907241702e-21, 2.6470256103265923e-21, 1.5761843958739643e-21, 1.2838104151250327e-21, 1.0035726743401969e-21, 7.202031389490357e-22, 2.002130754360117e-22, 1.71553763679713e-22, 1.1937587571271998e-22]
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

    # Saving all the figures:
    directory = "saved_plots/"
    fig_nums = plt.get_fignums()
    figs = [plt.figure(n) for n in fig_nums]
    if flag_mcmc == "y":
        if flag_extras == "y":
            plotname = ["trace", "joint_posterior", "sky_map", "corner_plot",
                        "signal", "spectrogram", "angles", "numerical_error"]
        else:
            plotname = ["trace", "joint_posterior", "sky_map", "corner_plot",
                        "signal", "spectrogram"]
    else:
        if flag_extras == "y":
            plotname = ["signal", "spectrogram", "angles", "numerical_error"]
        else:
            plotname = ["signal", "spectrogram"]

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
