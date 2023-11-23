from pvlib import pvsystem, singlediode
import numpy as np

from workbench.utilities import circuits, general


def solve_iv_curve(parameters, Geff, Tcell, iv_curve_pnts=1000):
    """
    Use De Soto and Bishop to simulate a full IV curve with both
    forward and reverse bias regions.
    :param parameters: the cell parameters
    :param Geff: the irradiance at the cell [W/sqm.]
    :param Tcell: the cell temperature [˚C]
    :param ivcurve_pnts: the number of points to build the IV curve from Default 1000
    :return: tuple output of bishop88 IV curve calculation (currents [A], voltages [V])
    """
    if Geff < parameters['minimum_irradiance_cell']:
        I, V, P = (np.zeros(iv_curve_pnts), np.zeros(iv_curve_pnts), np.zeros(iv_curve_pnts))
    else:
        sde_args = pvsystem.calcparams_desoto(
            Geff,
            Tcell,
            alpha_sc=parameters['desoto_short_circuit_temp_coe_alpha_sc'],
            a_ref=parameters['desoto_diode_factor_a_ref'],
            I_L_ref=parameters['desoto_photocurrent_I_L_ref'],
            I_o_ref=parameters['desoto_saturation_current_I_o_ref'],
            R_sh_ref=parameters['desoto_shunt_resist_R_sh_ref'],
            R_s=parameters['desoto_series_resist_R_s_ref'],
            EgRef=parameters['desoto_energy_bandgap_Egref'],
            dEgdT=parameters['desoto_bandgap_temp_coe_dEgdT']
        )
        # sde_args has values:
        # (photocurrent, saturation_current, resistance_series,
        # resistance_shunt, nNsVth)

        # Use Bishop's method to calculate points on the IV curve with V ranging
        # from the reverse breakdown voltage to open circuit
        kwargs = {
            'breakdown_factor': parameters['bishop_breakdown_factor'],
            'breakdown_exp': parameters['bishop_breakdown_exp'],
            'breakdown_voltage': parameters['bishop_breakdown_voltage'],
        }

        # ideally would use some intelligent log-spacing to concentrate points
        # around the forward- and reverse-bias knees, but this is good enough:
        evaluated_voltages = general.create_voltage_range(sde_args, kwargs, iv_curve_pnts=iv_curve_pnts)

        I, V, P = singlediode.bishop88(evaluated_voltages, *sde_args, **kwargs)
    return np.array(I), np.array(V)
