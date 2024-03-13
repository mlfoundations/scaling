import numpy as np
from scipy.optimize import curve_fit


def powlaw_irreducible(x, a, b, c):
    return a * np.power(x, b) + c


def powlaw_ours(x, alpha, beta, b, E):
    # x is tuple of N and m
    N = x[0]
    M = x[1]

    return (alpha * np.power(N, b)) + (beta * np.power(6, b / 2) * np.power(M * N, b)) + E


def powlaw(x, a, b):
    return a * np.power(x, b)


def linlaw(x, a, b):
    return a + x * b


def decay_ours(x, a, b, e):
    return e - a * np.exp(x) ** (-b)


"""curve fitting functions"""


def curve_fit_powlaw_irreducible(xdata, ydata):
    # try many different fits and retain the best one as done in chinchilla
    a_p0 = [3e1, 3e2, 3e3]
    b_p0 = [-1e-1]
    c_p0 = [0.0]

    min_residual = float("inf")
    ret = None

    for a0 in a_p0:
        for b0 in b_p0:
            for c0 in c_p0:

                popt, _ = curve_fit(
                    powlaw_irreducible,
                    xdata,
                    ydata,
                    p0=[a0, b0, c0],
                    maxfev=10000,
                )

                ydatafit = powlaw_irreducible(xdata, *popt)

                residuals = ydata - ydatafit
                curr_residual = (np.sum(residuals**2) / (residuals.size - 2)) ** 0.5

                if curr_residual < min_residual:
                    min_residual = curr_residual
                    # ret = (popt, pcov, ydatafit)
                    ret = popt
    return ret


def curve_fit_powlaw(xdata, ydata):
    # https://stackoverflow.com/questions/41109122/fitting-a-curve-to-a-power-law-distribution-with-curve-fit-does-not-work

    """Fit data to a power law with weights according to a log scale"""
    # Weights according to a log scale
    # Apply fscalex
    xdata_log = np.log10(xdata)
    # Apply fscaley
    ydata_log = np.log10(ydata)
    # Fit linear
    popt_log, pcov_log = curve_fit(linlaw, xdata_log, ydata_log)
    # Apply fscaley^-1 to fitted data
    ydatafit_log = np.power(10, linlaw(xdata_log, *popt_log))
    # There is no need to apply fscalex^-1 as original data is already available

    return np.array([np.power(10, popt_log[0]), popt_log[1]])


def curve_fit_powlaw_ours(xdata, ydata):
    # try many different fits and retain the best one as done in chinchilla

    alpha_p0 = [1e2, 3e2, 1e3, 3e3]
    beta_p0 = [1e2, 3e2, 1e3, 3e3]
    b_p0 = [
        -1e-1,
        -3e-1,
    ]
    e_p0 = [0.0, 1.0, 2.0, 10.0]

    min_residual = float("inf")
    ret = None

    for a0 in alpha_p0:
        for b0 in beta_p0:
            for c0 in b_p0:
                for e0 in e_p0:

                    popt, _ = curve_fit(
                        powlaw_ours,
                        xdata,
                        ydata,
                        p0=[a0, b0, c0, e0],
                        maxfev=10000,
                    )

                    ydatafit = powlaw_ours(xdata, *popt)
                    residuals = ydata - ydatafit
                    curr_residual = (np.sum(residuals**2) / (residuals.size - 2)) ** 0.5

                    if curr_residual < min_residual:
                        min_residual = curr_residual
                        ret = popt

    return ret


def curve_fit_decay_ours(xdata, ydata):
    popt, _ = curve_fit(
        decay_ours,
        xdata,
        ydata,
        maxfev=10000,
    )

    return popt
