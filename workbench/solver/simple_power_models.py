


def module_efficiency_method(eff_stc, area_mod, G_eff, B_ref, T_cell, T_stc=25, I_misc=0.10):
    """
    :param eff_stc: nominal module efficiency (as ratio)
    :param area_mod: area of the module (sqm.)
    :param G_eff: effective irradiance (W/sqm.)
    :param B_ref: maximum power temperature coefficient (% loss / ˚C)
    :param T_cell: current cell temperature (˚C)
    :param T_stc: nominal cell temperature (˚C) Default to 25
    :param I_misc: losses due to miscellaneous system anomalies (<1 ratio) Default to 0.10
    :return: P: power (W)

    from Sec. 23.2 in:
    J. A. Dufﬁe and W. A. Beckman, Solar Engineering of Thermal Processes, Fourth. Wiley, 2013.



    """

    eff_mp = eff_stc + B_ref * (T_cell - T_stc)
    losses = (1 - I_misc)
    P_mp = G_eff * area_mod * eff_mp

    # P_mp2 = (G_eff * area_mod) * (ef_stc * (1.0 - B_ref * (T_cell - T_stc)))

    return P_mp * losses

def pv_watts_method(G_eff, T_cell, P_ref, gamma, T_ref=25, G_ref=1000, I_misc=0.1):
    """
    :param G_eff: effective irradiance (W/sqm.)
    :param T_cell: current cell temperature (˚C)
    :param P_ref: peak power (W)
    :param gamma: maximum power temperature coefficient (% loss / ˚C) (typ -0.00485)
    :param T_ref: cell temperature at test conditions (˚C) Default to 25
    :param G_ref: irradiance at test conditions (W/m2) Default to 1000
    :param I_misc: system losses (-) Detaul to 0.1
    :return: power with system losses
    """
    if G_eff > 125:
        P_mp = (G_eff / G_ref) * P_ref * (1 + gamma * (T_cell - T_ref))
    else:
        P_mp = ((0.008 * G_eff**2) / G_ref) * P_ref * (1 + gamma * (T_cell - T_ref))
    return P_mp * (1 - I_misc)
