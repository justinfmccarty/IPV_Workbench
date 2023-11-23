# ================================ Irradiance Corrections ================================
# irradiance at cell
from workbench.old_solver import simulations_mp_archive, simulations
from workbench.utilities import circuits, general
import numpy as np


# def calculate_effective_irradiance(direct_irrad, diffuse_irrad, film=None):
#     if film == None:
#         return account_for_cover(direct_irrad, diffuse_irrad)
#     else:
#         step_a_direct, step_a_diffuse = account_for_film(direct_irrad, diffuse_irrad)
#         return account_for_cover(step_a_direct, step_a_diffuse)
#
#
# def account_for_cover(direct_irrad, diffuse_irrad):
#     # space filler for now
#     # ideally able to represent glass, polycarbonate, PLA, etc.
#     # Likely section 5.3 of Solar Engineering of Thermal Processes p.206
#     return direct_irrad + diffuse_irrad
#
#
# def account_for_film(direct_irrad, diffuse_irrad):
#     return direct_irrad, diffuse_irrad



# ================================ Curve Calculations ================================




def calculate_module_curve(irradiance_hoy, temperature_hoy, module_dict, cell_params, ivcurve_pnts=250):

    # TODO break apart into constituent pieces
    active_submodule_map = module_dict['MAPS']['SUBMODULES']
    active_diode_map = module_dict['MAPS']['DIODES']
    submodules = np.unique(active_submodule_map)
    diodes = np.unique(active_diode_map)

    submodule_i = []
    submodule_v = []

    for submodule_key in submodules:
        submodule_mask = active_submodule_map == submodule_key
        submodule_diode = active_diode_map[submodule_mask]
        submodule_irrad = irradiance_hoy[submodule_mask]
        submodule_temp = temperature_hoy[submodule_mask]
        diode_i = []
        diode_v = []
        for diode_key in diodes:
            diode_mask = submodule_diode == diode_key
            submodule_subdiode_irrad = submodule_irrad[diode_mask]
            submodule_subdiode_temp = submodule_temp[diode_mask]
            Imod, Vmod = simulations.solve_iv_curve(cell_params,
                                                    submodule_subdiode_irrad,
                                                    submodule_subdiode_temp, ivcurve_pnts=ivcurve_pnts)

            sub_diode_curves = np.array([Imod.T, Vmod.T])
            # sub_diode_curves = cell.retrieve_curves_multiple_cells(submodule_subdiode_irrad,
            #                                                             submodule_subdiode_temp)

            i, v = circuits.calc_series(sub_diode_curves,
                                        breakdown_voltage=cell_params['breakdown_voltage'],
                                        diode_threshold=cell_params['diode_threshold'],
                                        bypass=False)
            diode_i.append(i)
            diode_v.append(v)

        # calc series with bypass diodes
        diode_curves = np.array([diode_i, diode_v])
        i, v = circuits.calc_series(diode_curves,
                                    breakdown_voltage=cell_params['breakdown_voltage'],
                                    diode_threshold=cell_params['diode_threshold'],
                                    bypass=True)
        submodule_i.append(i)
        submodule_v.append(v)
    submodule_curves = np.array([submodule_i, submodule_v])
    Imod, Vmod = circuits.calc_parallel(submodule_curves)
    return Imod, Vmod



def calculate_module_curve_single_row(irradiance_hoy, temperature_hoy, parameters,
                                         submodule_map, subdiode_map, subcell_map, ivcurve_pnts=250):
    # TODO break apart into constituent pieces

    submodules = np.unique(submodule_map)
    diodes = np.unique(subdiode_map)

    # we are going to solve the IV curves for current given voltages.
    # Since we are solving at the level of the cell we need to parse down from the module range

    # Basic assumption here: Module IV-curve can be converted to a cell IV-curve by
    # dividing the module voltage by the number of cells the subcell current is calculated by
    # dividing currents by the number of subcells.

    submodule_i = []
    submodule_v = []
    for submodule_key in submodules:
        submodule_mask = submodule_map == submodule_key
        submodule_colmask = submodule_mask.flatten()

        submodule_diode = np.array(subdiode_map)[:, submodule_colmask]
        submodule_subcell = np.array(subcell_map)[:, submodule_colmask]

        submodule_irrad = irradiance_hoy[:, submodule_colmask]
        submodule_temp = temperature_hoy[:, submodule_colmask]

        diode_i = []
        diode_v = []
        for diode_key in diodes:
            diode_mask = submodule_diode == diode_key
            diode_colmask = diode_mask.flatten()

            submodule_subdiode_subcell = submodule_subcell[:, diode_colmask]

            submodule_subdiode_irrad = submodule_irrad[:, diode_colmask]
            submodule_subdiode_temp = submodule_temp[:, diode_colmask]

            cells_i = []
            cells_v = []
            # print(submodule_subdiode_irrad.shape)
            # print(np.mean(submodule_subdiode_irrad, axis=0).flatten().shape)
            Icell, Vcell = simulations.solve_cells(parameters,
                                                   np.mean(submodule_subdiode_irrad, axis=0).flatten(),
                                                   np.mean(submodule_subdiode_temp, axis=0).flatten(),
                                                   ivcurve_pnts=ivcurve_pnts)

            sub_diode_curves = np.array([Icell.T, Vcell.T])
            i, v = circuits.calc_series(sub_diode_curves,
                                        breakdown_voltage=parameters['breakdown_voltage'],
                                        diode_threshold=parameters['diode_threshold'],
                                        bypass=False)
            diode_i.append(i)
            diode_v.append(v)
            #
            # for subcell_key in np.unique(submodule_subdiode_subcell):
            #     submodule_subdiode_subcell_mask = submodule_subdiode_subcell == subcell_key
            #     submodule_subdiode_subcell_irrad = submodule_subdiode_irrad[submodule_subdiode_subcell_mask]
            #     submodule_subdiode_subcell_temp = submodule_subdiode_temp[submodule_subdiode_subcell_mask]
            #
            #     Icell, Vcell = simulations.solve_subcells(parameters,
            #                                               submodule_subdiode_subcell_irrad,
            #                                               submodule_subdiode_subcell_temp, ivcurve_pnts=ivcurve_pnts)
            #     cells_i.append(Icell)
            #     cells_v.append(Vcell)
            #
            # sub_diode_curves = np.array([cells_i, cells_v])
            #
            # # sub_diode_curves = cell.retrieve_curves_multiple_cells(submodule_subdiode_irrad,
            # #                                                              submodule_subdiode_temp)
            #
            # i, v = circuits.calc_series(sub_diode_curves,
            #                             breakdown_voltage=parameters['breakdown_voltage'],
            #                             diode_threshold=parameters['diode_threshold'],
            #                             bypass=False)
            # diode_i.append(i)
            # diode_v.append(v)

        # calc series with bypass diodes to get submodule strings
        diode_curves = np.array([diode_i, diode_v])
        i, v = circuits.calc_series(diode_curves,
                                    breakdown_voltage=parameters['breakdown_voltage'],
                                    diode_threshold=parameters['diode_threshold'],
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

def calculate_module_curve_single_column(irradiance_hoy, temperature_hoy, parameters,
                                         submodule_map, subdiode_map, subcell_map, ivcurve_pnts=250):
    # TODO break apart into constituent pieces

    submodules = np.unique(submodule_map)
    diodes = np.unique(subdiode_map)

    # we are going to solve the IV curves for current given voltages.
    # Since we are solving at the level of the cell we need to parse down from the module range

    # Basic assumption here: Module IV-curve can be converted to a cell IV-curve by
    # dividing the module voltage by the number of cells the subcell current is calculated by
    # dividing currents by the number of subcells.

    submodule_i = []
    submodule_v = []
    for submodule_key in submodules:
        submodule_mask = submodule_map == submodule_key
        submodule_rowmask = submodule_mask.flatten()

        submodule_diode = np.array(subdiode_map)[submodule_rowmask, :]
        submodule_subcell = np.array(subcell_map)[submodule_rowmask, :]

        submodule_irrad = irradiance_hoy[submodule_rowmask, :]
        submodule_temp = temperature_hoy[submodule_rowmask, :]

        diode_i = []
        diode_v = []
        for diode_key in diodes:
            diode_mask = submodule_diode == diode_key
            diode_rowmask = diode_mask.flatten()

            submodule_subdiode_subcell = submodule_subcell[diode_rowmask, :]

            submodule_subdiode_irrad = submodule_irrad[diode_rowmask, :]
            submodule_subdiode_temp = submodule_temp[diode_rowmask, :]

            # cells_i = []
            # cells_v = []

            Icell, Vcell = simulations.solve_cells(parameters,
                                                   np.mean(submodule_subdiode_irrad, axis=1).flatten(),
                                                   np.mean(submodule_subdiode_temp, axis=1).flatten(),
                                                   ivcurve_pnts=ivcurve_pnts)

            sub_diode_curves = np.array([Icell.T, Vcell.T])
            i, v = circuits.calc_series(sub_diode_curves,
                                        breakdown_voltage=parameters['breakdown_voltage'],
                                        diode_threshold=parameters['diode_threshold'],
                                        bypass=False)
            diode_i.append(i)
            diode_v.append(v)


            # for subcell_key in np.unique(submodule_subdiode_subcell):
            #     submodule_subdiode_subcell_mask = submodule_subdiode_subcell == subcell_key
            #     submodule_subdiode_subcell_irrad = submodule_subdiode_irrad[submodule_subdiode_subcell_mask]
            #     submodule_subdiode_subcell_temp = submodule_subdiode_temp[submodule_subdiode_subcell_mask]
            #
            #     Icell, Vcell = simulations.solve_subcells(parameters,
            #                                               submodule_subdiode_subcell_irrad,
            #                                               submodule_subdiode_subcell_temp,
            #                                               ivcurve_pnts=ivcurve_pnts)
            #
            #     cells_i.append(Icell)
            #     cells_v.append(Vcell)

            # sub_diode_curves = np.array([cells_i, cells_v])
            #
            # i, v = circuits.calc_series(sub_diode_curves,
            #                             breakdown_voltage=parameters['breakdown_voltage'],
            #                             diode_threshold=parameters['diode_threshold'],
            #                             bypass=False)
            # diode_i.append(i)
            # diode_v.append(v)

        # calc series with bypass diodes to get submodule strings
        diode_curves = np.array([diode_i, diode_v])
        i, v = circuits.calc_series(diode_curves,
                                    breakdown_voltage=parameters['breakdown_voltage'],
                                    diode_threshold=parameters['diode_threshold'],
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


# def calculate_module_curve_multiple_column(irradiance_hoy, temperature_hoy, parameters,
#                                            submodule_map, subdiode_map, ivcurve_pnts=250):
#     # TODO break apart into constituent pieces
#
#     submodules = np.unique(submodule_map)
#     diodes = np.unique(subdiode_map)
#
#     # we are going to solve the IV curves for current given voltages.
#     # Since we are solving at the level of the cell we need to parse down from the module range
#
#     # Basic assumption here: Module IV-curve can be converted to a cell IV-curve by
#     # dividing the module voltage by the number of cells the subcell current is calculated by
#     # dividing currents by the number of subcells.
#
#     submodule_i = []
#     submodule_v = []
#     for submodule_key in submodules:
#         submodule_mask = submodule_map == submodule_key
#
#         submodule_diode = general.mask_nd(np.array(subdiode_map), submodule_mask)
#
#         submodule_irrad = general.mask_nd(irradiance_hoy, submodule_mask)
#         submodule_temp = general.mask_nd(temperature_hoy, submodule_mask)
#
#         diode_i = []
#         diode_v = []
#         for diode_key in diodes:
#             diode_mask = submodule_diode == diode_key
#
#             submodule_subdiode_irrad = general.mask_nd(submodule_irrad, diode_mask)
#             submodule_subdiode_temp = general.mask_nd(submodule_temp, diode_mask)
#
#             Icell, Vcell = simulations.solve_cells(parameters,
#                                                    submodule_subdiode_irrad.flatten(),
#                                                    submodule_subdiode_temp.flatten(),
#                                                    ivcurve_pnts=ivcurve_pnts)
#
#             sub_diode_curves = np.array([Icell.T, Vcell.T])
#             i, v = circuits.calc_series(sub_diode_curves,
#                                         breakdown_voltage=parameters['breakdown_voltage'],
#                                         diode_threshold=parameters['diode_threshold'],
#                                         bypass=False)
#             diode_i.append(i)
#             diode_v.append(v)
#
#         # calc series with bypass diodes to get submodule strings
#         diode_curves = np.array([diode_i, diode_v])
#         i, v = circuits.calc_series(diode_curves,
#                                     breakdown_voltage=parameters['breakdown_voltage'],
#                                     diode_threshold=parameters['diode_threshold'],
#                                     bypass=True)
#         submodule_i.append(i)
#         submodule_v.append(v)
#
#     # calc parallel connection of submodule strings to get module curves
#     submodule_curves = np.array([submodule_i, submodule_v])
#
#     # the parallel interpolation will work on lists of length 1 but can lead to weird results
#     # safer to just skip it if possible
#     if len(submodules) > 1:
#         Imod, Vmod = circuits.calc_parallel(submodule_curves)
#     else:
#         Imod = submodule_i[0]
#         Vmod = submodule_v[0]
#     return Imod, Vmod
