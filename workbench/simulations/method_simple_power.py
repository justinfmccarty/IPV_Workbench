import numpy as np

from workbench.device import temperature
from workbench.utilities import general
from workbench.simulations import method_effective_irradiance


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
    :param I_misc: system losses (-) Default to 0.1
    :return: power with system losses
    """

    if gamma < -0.02:
        gamma = gamma / 100

    if G_eff > 125:
        P_mp = (G_eff / G_ref) * P_ref * (1 + gamma * (T_cell - T_ref))
    else:
        P_mp = ((0.008 * G_eff ** 2) / G_ref) * P_ref * (1 + gamma * (T_cell - T_ref))
    return P_mp * (1 - I_misc)


def module_center_pt(module_dict, sensor_pts_xyz_arr, direct, diffuse, all_hoy, tmy_location, psl, dbt):
    module_center = np.stack([module_dict['Details']['panelizer_center_pt']])

    T_noct = module_dict['Parameters']['performance_NOCT_T_degC']
    # nom_eff = module_dict['Parameters']['param_nom_eff']
    peak_power = module_dict['Parameters']['param_actual_capacity_Wp']
    area_mod = module_dict['Parameters']['param_actual_module_area_m2']
    area_cells = module_dict['Parameters']['param_actual_total_cell_area_m2']
    module_normal = module_dict['Details']['panelizer_normal']
    gamma_ref = module_dict['Parameters']['performance_temp_coe_gamma_pctC']
    front_cover = module_dict['Layers']['panelizer_front_film']

    G_dir_ann = general.collect_raw_irradiance(module_center, sensor_pts_xyz_arr, direct)
    G_diff_ann = general.collect_raw_irradiance(module_center, sensor_pts_xyz_arr, diffuse)
    # calculate the effective irradiance for the year
    G_eff_ann = method_effective_irradiance.calculate_effective_irradiance_timeseries(G_dir_ann,
                                                                                      G_diff_ann,
                                                                                      module_normal,
                                                                                      all_hoy,
                                                                                      tmy_location,
                                                                                      psl,
                                                                                      dbt,
                                                                                      front_cover)
    G_eff_ann = G_eff_ann.flatten()
    T_cell = temperature.calculate_cell_temperature(G_eff_ann, dbt, T_noct)
    # P = simple_power_models.module_efficiency_method(nom_eff, area_mod, G_eff_ann, gamma_ref, T_cell)
    power = np.vectorize(pv_watts_method)(G_eff_ann, T_cell, peak_power, gamma_ref)
    module_area = np.zeros_like(power) + area_cells
    irradiance = G_eff_ann * area_mod

    return module_area, irradiance, power

def module_cell_pt(module_dict, pv_cells_xyz_arr, sensor_pts_xyz_arr, direct, diffuse, all_hoy, tmy_location, psl, dbt):
    T_noct = module_dict['Parameters']['performance_NOCT_T_degC']
    # nom_eff = module_dict['Parameters']['nom_eff']
    peak_power = module_dict['Parameters']['param_cell_peak_Wp']
    area_cell = module_dict['Parameters']['param_one_cell_area_m2']
    module_area = module_dict['Parameters']['param_actual_module_area_m2']
    module_normal = module_dict['Details']['panelizer_normal']
    gamma_ref = module_dict['Parameters']['performance_temp_coe_gamma_pctC']

    G_dir_ann = general.collect_raw_irradiance(pv_cells_xyz_arr, sensor_pts_xyz_arr, direct)
    G_diff_ann = general.collect_raw_irradiance(pv_cells_xyz_arr, sensor_pts_xyz_arr, diffuse)

    front_cover = module_dict['Layers']['panelizer_front_film']
    # calculate the effective irradiance for the year
    G_eff_ann = method_effective_irradiance.calculate_effective_irradiance_timeseries(G_dir_ann,
                                                                                      G_diff_ann,
                                                                                      module_normal,
                                                                                      all_hoy,
                                                                                      tmy_location,
                                                                                      psl,
                                                                                      dbt,
                                                                                      front_cover)

    Pmod_results = {}
    Gmod_results = {}
    area_results = {}
    for hoy_n, hoy in enumerate(all_hoy):
        Gmod = G_eff_ann[hoy_n].flatten()
        Tcells = temperature.calculate_cell_temperature(Gmod, dbt[hoy_n], T_noct)

        # Gmod_total = np.sum(Gmod.flatten()*module_dict['PARAMETERS']['one_cell_area_m2']) / module_dict['PARAMETERS']['actual_module_area_m2']
        # if Gmod_total < module_dict['PARAMETERS']['minimum_irradiance_module']

        # For power
        if np.sum(Gmod < module_dict['Parameters']['minimum_irradiance_cell']) > 0:

            # print(hoy_n, hoy, time_utils.hoy_to_date(hoy), "Zero Array", np.sum(Gmod))
            Pcells = 0
        else:
            # Pcells = simple_power_models.module_efficiency_method(nom_eff,
            #                                                         area_cell,
            #                                                         G_eff_hoy,
            #                                                         gamma_ref,
            #                                                         Tcells)
            Pcells = np.vectorize(pv_watts_method)(Gmod, Tcells, peak_power, gamma_ref)
        Pmod_results[hoy] = np.sum(Pcells)

        # For area
        area_results[hoy] = module_area

        # For irradiance
        # Gmod is originally an array of W/m2 for each cell. Need to convert this array to W by multiply by cell area
        # then take the sum of irradiance for all the cells
        # this statement takes the mean of the subcells if necesssary (just like in the simulation)
        if module_dict['Parameters']['param_n_subcells'] > 1:
            if module_dict['Parameters']['orientation'] == 'portrait':
                Gmod = np.mean(Gmod, axis=0)
            else:
                Gmod = np.mean(Gmod, axis=1)
        else:
            pass
        Gmod = Gmod * area_cell
        Gmod_results.update({hoy: np.round(np.sum(Gmod), 1)})

    return area_results, Gmod_results, Pmod_results