from pvlib import pvsystem, singlediode
import numpy as np


def simulate_cell_curve(parameters, Geff, Tcell, ivcurve_pnts=1000):
    """
    Use De Soto and Bishop to simulate a full IV curve with both
    forward and reverse bias regions.
    :param parameters: the cell parameters
    :param Geff: the irradiance at the cell
    :param Tcell: the cell temperature
    :param ivcurve_pnts: the number of points to build the IV curve from Default 1000
    :return: tuple output of bishop88 IV curve calculation (currents [A], voltages [V])
    """
    # adjust the reference parameters according to the operating conditions of Geff and Tcell using the De Soto model:
    sde_args = pvsystem.calcparams_cec (
        Geff,
        Tcell,
        alpha_sc=parameters['alpha_sc'],
        a_ref=parameters['a_ref'],
        I_L_ref=parameters['I_L_ref'],
        I_o_ref=parameters['I_o_ref'],
        R_sh_ref=parameters['R_sh_ref'],
        R_s=parameters['R_s'],
    )
    # sde_args has values:
    # (photocurrent, saturation_current, resistance_series,
    # resistance_shunt, nNsVth)

    # Use Bishop's method to calculate points on the IV curve with V ranging
    # from the reverse breakdown voltage to open circuit
    kwargs = {
        'breakdown_factor': parameters['breakdown_factor'],
        'breakdown_exp': parameters['breakdown_exp'],
        'breakdown_voltage': parameters['breakdown_voltage'],
    }
    v_oc = singlediode.bishop88_v_from_i(
        0.0, *sde_args, **kwargs
    )
    # ideally would use some intelligent log-spacing to concentrate points
    # around the forward- and reverse-bias knees, but this is good enough:
    vd = np.linspace(0.99 * kwargs['breakdown_voltage'], v_oc, ivcurve_pnts)

    I, V, P = singlediode.bishop88(vd, *sde_args, **kwargs)
    return I, V