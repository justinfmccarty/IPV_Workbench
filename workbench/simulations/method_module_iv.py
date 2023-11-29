from pvlib import pvsystem, singlediode
import numpy as np
from workbench.utilities import circuits, general
from workbench.simulations import method_iv_solver, method_effective_irradiance
from workbench.device import temperature




def solve_module_iv_curve(host_object, G_eff_ann_mod, module_dict, temporal_idx,
                          dbt):
    Imod = []
    Vmod = []

    for time_n, time_ in enumerate(temporal_idx):
        Gmod = G_eff_ann_mod[time_n].flatten()
        Tmod = temperature.calculate_cell_temperature(Gmod, dbt[time_n],
                                                      module_dict['Parameters']['performance_NOCT_T_degC'])
        Imod_hoy, Vmod_hoy = solve_module_map_dependent(Gmod, Tmod, module_dict, host_object.device_iv_dict, host_object.irradiance_range,
                                   host_object.temperature_range)
        Imod.append(Imod_hoy)
        Vmod.append(Vmod_hoy)
    return Imod, Vmod




def solve_module_map_dependent(Gmod, Tmod, module_dict, device_iv_dict, irradiance_range, temperature_range):
    if module_dict['Parameters']['param_n_subcells'] > 1:
        if module_dict['Parameters']['shape_orientation'] == 'portrait':
            pass
            # Imod, Vmod = calculate_module_curve_single_row(Gmod,
            #                                                Tmod,
            #                                                module_dict['PARAMETERS'],
            #                                                module_dict['MAPS']['SUBMODULES'],
            #                                                module_dict['MAPS']['DIODES'],
            #                                                module_dict['MAPS']['SUBCELLS'],
            #                                                ivcurve_pnts=ivcurve_pnts)
        else:
            pass
            # Imod, Vmod = calculate_module_curve_single_column(Gmod,
            #                                                   Tmod,
            #                                                   module_dict['PARAMETERS'],
            #                                                   module_dict['MAPS']['SUBMODULES'],
            #                                                   module_dict['MAPS']['DIODES'],
            #                                                   module_dict['MAPS']['SUBCELLS'],
            #                                                   ivcurve_pnts=ivcurve_pnts)

    else:
        Imod, Vmod = module_curve_multiple_column(Gmod,
                                                  Tmod,
                                                  module_dict['map_idx_arr'],
                                                  module_dict['Parameters'],
                                                  module_dict['Maps']['Submodules'],
                                                  module_dict['Maps']['Diodes'],
                                                  device_iv_dict,
                                                  irradiance_range,
                                                  temperature_range
                                                  )

    return Imod, Vmod


def module_curve_multiple_column(irradiance_hoy, temperature_hoy, module_idx_arr, module_parameters,
                                 submodule_map, subdiode_map, iv_dict, irradiance_range,
                                 temperature_range):
    #reshape and reindex based on the modules actual layout
    irradiance_hoy = irradiance_hoy[module_idx_arr]
    temperature_hoy = temperature_hoy[module_idx_arr]

    submodules = np.unique(submodule_map)
    diodes = np.unique(subdiode_map)

    submodule_i = []
    submodule_v = []
    for submodule_key in submodules:
        submodule_mask = submodule_map == submodule_key

        submodule_diode = general.mask_nd(np.array(subdiode_map), submodule_mask)

        submodule_irrad = general.mask_nd(irradiance_hoy, submodule_mask)
        submodule_temp = general.mask_nd(temperature_hoy, submodule_mask)

        diode_i = []
        diode_v = []
        for diode_key in diodes:
            diode_mask = submodule_diode == diode_key

            submodule_subdiode_irrad = general.mask_nd(submodule_irrad, diode_mask)
            submodule_subdiode_temp = general.mask_nd(submodule_temp, diode_mask)

            Icell_list = []
            Vcell_list = []
            for Geff, Tcell in list(zip(submodule_subdiode_irrad.flatten(), submodule_subdiode_temp.flatten())):
                irradiance_key = general.find_nearest(irradiance_range, Geff)
                temperature_key = general.find_nearest(temperature_range, Tcell)
                iv_key = str((irradiance_key, temperature_key))
                Icell, Vcell = iv_dict[iv_key]

                # current is stored in Amp/sqm.
                # with some modules (thin film) the cell size can change when modified by panelizer
                Icell_list.append(Icell * module_parameters['param_one_cell_area_m2'])
                Vcell_list.append(Vcell)

            Icell = np.array(Icell_list)
            Vcell = np.array(Vcell_list)
            sub_diode_curves = np.array([Icell, Vcell])


            i, v = circuits.calc_series(sub_diode_curves,
                                        breakdown_voltage=module_parameters['bishop_breakdown_voltage'],
                                        # diode_threshold=module_parameters['diode_threshold'],
                                        bypass=False)
            diode_i.append(i)
            diode_v.append(v)

        # calc series with bypass diodes to get submodule strings
        diode_curves = np.array([diode_i, diode_v])
        i, v = circuits.calc_series(diode_curves,
                                    breakdown_voltage=module_parameters['bishop_breakdown_voltage'],
                                    # diode_threshold=module_parameters['diode_threshold'],
                                    bypass=True)
        submodule_i.append(i)
        submodule_v.append(v)

    # calc parallel connection of submodule strings to get module curves
    submodule_curves = np.array([submodule_i, submodule_v])

    # the parallel interpolation will work on lists of length 1 but can lead to weird results
    # safer to just skip it if possible
    if len(submodules) > 1:
        Imod, Vmod = circuits.calc_parallel(submodule_curves)
    else:
        Imod = submodule_i[0]
        Vmod = submodule_v[0]
    return Imod, Vmod
