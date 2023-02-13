import numpy as np


def calc_series(iv_curves, breakdown_voltage, diode_threshold=None, bypass=True):
    # I_data = curves[:, :, 0]
    # V_data = curves[:, :, 1]
    I_data = iv_curves[0, :, :]
    V_data = iv_curves[1, :, :]

    substring_Isc = calc_short_circuit(iv_curves)
    substring_Imax = calc_current_max(iv_curves, breakdown_voltage)

    Isub, Vsub = assemble_series(I_data,
                                 V_data,
                                 np.mean(substring_Isc),
                                 np.max(substring_Imax))
    if bypass == True:
        if diode_threshold is None:
            diode_threshold = -0.5
        Vsub = np.clip(Vsub, a_min=diode_threshold, a_max=None)
    else:
        pass
    return Isub, Vsub


def assemble_series(I, V, meanIsc, Imax):
    """
    This is the calc series method used by sunpowers pvmismatch tool

    Calculate IV curve for cells and substrings in series given current and
    voltage in increasing order by voltage, the average short circuit
    current and the max current at the breakdown voltage.
    :param I: cell or substring currents [A]
    :param V: cell or substring voltages [V]
    :param meanIsc: average short circuit current [A]
    :param Imax: maximum current [A]
    :return: current [A] and voltage [V] of series
    """
    I = np.asarray(I)  # currents [A]
    V = np.asarray(V)  # voltages [V]
    # make sure all inputs are numpy arrays, but don't make extra copies
    _npts = 101
    pts = (11. - np.logspace(np.log10(11.), 0., _npts)) / 10.
    pts[0] = 0.  # first point must be exactly zero
    pts = pts.reshape((_npts, 1))
    Imod_pts = 1 - np.flipud(pts)
    Imod_pts_sq = Imod_pts ** 2 + np.finfo(np.float64).eps

    meanIsc = np.asarray(meanIsc)  # mean Isc [A]
    Imax = np.asarray(Imax)  # max current [A]
    # create array of currents optimally spaced from mean Isc to  max VRBD
    Ireverse = (Imax - meanIsc) * Imod_pts_sq + meanIsc
    # range of currents in forward bias from 0 to mean Isc
    Iforward = meanIsc * pts
    Imin = np.minimum(I.min(), 0.)  # minimum cell current, at most zero
    # range of negative currents in the 4th quadrant from min current to 0
    negpts = (11. - np.logspace(np.log10(11. - 1. / float(_npts)),
                                0., _npts)) / 10.
    negpts = negpts.reshape((_npts, 1))
    Imod_negpts = 1 + 1. / float(_npts) / 10. - negpts
    Iquad4 = Imin * Imod_negpts
    # create range for interpolation from forward to reverse bias
    Itot = np.concatenate((Iquad4, Iforward, Ireverse), axis=0).flatten()
    Vtot = np.zeros((3 * _npts,))
    # add up all series cell voltages
    for i, v in zip(I, V):
        # interp requires x, y to be sorted by x in increasing order
        Vtot += np.interp(Itot, np.flipud(i), np.flipud(v))
    return np.flipud(Itot), np.flipud(Vtot)


def calc_parallel(iv_curves):
    I_data = iv_curves[0, :, :]
    V_data = iv_curves[1, :, :]
    Vmax = np.max(V_data)
    Vmin = np.min(V_data)
    I, V = assemble_parallel(I_data, V_data, Vmax, Vmin)
    return I, V


def assemble_parallel(I, V, Vmax, Vmin, Voc=None):
    """
    Calculate IV curve for cells and substrings in parallel.
    :param I: currents [A]
    :type: I: list, :class:`numpy.ndarray`
    :param V: voltages [V]
    :type: V: list, :class:`numpy.ndarray`
    :param Vmax: max voltage limit, should be max Voc [V]
    :param Vmin: min voltage limit, could be zero or Vrbd [V]
    :param Voc: (``None``) open circuit voltage [V]
    """

    _npts = 101
    pts = (11. - np.logspace(np.log10(11.), 0., _npts)) / 10.
    negpts = (11. - np.logspace(np.log10(11. - 1. / float(_npts)),
                                0., _npts)) / 10.
    negpts = negpts.reshape((_npts, 1))
    pts = pts.reshape((_npts, 1))
    Imod_negpts = 1 + 1. / float(_npts) / 10. - negpts
    Vmod_q4pts = np.flipud(Imod_negpts)

    if Voc is None:
        Voc = Vmax

    I, V = np.asarray(I), np.asarray(V)
    Vmax = np.asarray(Vmax)
    Vmin = np.asarray(Vmin)
    Voc = np.asarray(Voc)
    Vff = Voc
    delta_Voc = Vmax - Voc
    if np.isclose(delta_Voc, 0):
        Vff = 0.8 * Voc
        delta_Voc = 0.2 * Voc
    elif delta_Voc < 0:
        Vff = Vmax
        delta_Voc = -delta_Voc
    Vquad4 = Vff + delta_Voc * Vmod_q4pts
    Vreverse = Vmin * negpts
    Vforward = Vff * pts
    Vtot = np.concatenate((Vreverse, Vforward, Vquad4), axis=0).flatten()
    Itot = np.zeros((3 * _npts,))
    for i, v in zip(I, V):
        Itot += np.interp(Vtot, v, i)
    return Itot, Vtot


# ================================ Utilities ================================

def calc_short_circuit(iv_curves):
    # substring_Isc = [np.interp(0,
    #                         sub[:,1], # V curve valeus
    #                         sub[:,0]) # I curve values
    #                 for sub in iv_curves]
    substring_Isc = [np.interp(0,
                               iv_curves[1][c],
                               iv_curves[0][c])
                     for c in range(0, iv_curves.shape[1])]
    return np.array(substring_Isc)


def calc_current_max(iv_curves, breakdown_voltage):
    # substring_Imax = [np.interp(cell.breakdown_voltage,
    #                         sub[:,1], # V curve valeus
    #                         sub[:,0]) # I curve values
    #                 for sub in iv_curves]
    substring_Imax = [np.interp(breakdown_voltage,
                                iv_curves[1][c],
                                iv_curves[0][c])
                      for c in range(0, iv_curves.shape[1])]
    return np.array(substring_Imax)


def find_mpp(iv_arr):
    power_arr = iv_arr[0, :] * iv_arr[1, :]
    idx = np.argmax(power_arr)
    return power_arr, iv_arr[0, :][idx], iv_arr[1, :][idx]


def apply_bypass_diode(Vsub, module_params):
    return np.clip(Vsub, a_min=module_params['diode_threshold'], a_max=None)
