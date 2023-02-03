from pvlib import pvsystem, singlediode
import numpy as np

from ipv_workbench.utilities import circuits, utils


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
    sde_args = pvsystem.calcparams_cec(
        Geff,
        Tcell,
        alpha_sc=parameters['alpha_sc'],
        a_ref=parameters['a_ref'],
        I_L_ref=parameters['I_L_ref'],
        I_o_ref=parameters['I_o_ref'],
        R_sh_ref=parameters['R_sh_ref'],
        R_s=parameters['R_s'],
        Adjust=parameters['Adjust']
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


#
# def calc_yield(panelizer_object, topology, tmy_df, surface_name, hour_of_year):
#         topology = panelizer_object.electrical_topology
#         if topology == "central_inverter":
#              Isys, Vsys = simulate_central_inverter(tmy_df,
#                                                     panelizer_object,
#                                                     surface_name,
#                                                     hour_of_year)
#         elif topology == topology_list[1]:
#             string_yield, strings = calc_yield_string_inverter(ds_hour, unq_strings, library, g_space, t_space)
#             return (string_yield, strings)
#         elif topology == topology_list[2]:
#             module_yield, modules = calc_yield_micro_inverter(ds_hour, unq_modules, library, g_space, t_space)
#             return (module_yield, modules)
#     else:
#         print("Specify topology from:\n"
#               f"{topology_list}")
#         return None

def calcMPP_IscVocFF(Isys, Vsys):
    """from PVmismatch"""
    Psys = Isys * Vsys
    mpp = np.argmax(Psys)
    if Psys[mpp] == 0:
        Imp, Vmp, Pmp, Isc, Voc, FF = 0, 0, 0, 0, 0, 0
    else:
        P = Psys[mpp - 1:mpp + 2]
        V = Vsys[mpp - 1:mpp + 2]
        I = Isys[mpp - 1:mpp + 2]

        if any(P) == 0 or any(V) == 0 or any(I) == 0:
            Imp, Vmp, Pmp, Isc, Voc, FF = 0, 0, 0, 0, 0, 0
        else:
            # calculate derivative dP/dV using central difference
            dP = np.diff(P, axis=0)  # size is (2, 1)
            dV = np.diff(V, axis=0)  # size is (2, 1)
            if any(dP) == 0 or any(dV) == 0:
                Imp, Vmp, Pmp, Isc, Voc, FF = 0, 0, 0, 0, 0, 0
            else:
                Pv = dP / dV  # size is (2, 1)
                # dP/dV is central difference at midpoints,
                Vmid = (V[1:] + V[:-1]) / 2.0  # size is (2, 1)
                Imid = (I[1:] + I[:-1]) / 2.0  # size is (2, 1)
                # interpolate to find Vmp

                Vmp = (-Pv[0] * np.diff(Vmid, axis=0) / np.diff(Pv, axis=0) + Vmid[0]).item()
                Imp = (-Pv[0] * np.diff(Imid, axis=0) / np.diff(Pv, axis=0) + Imid[0]).item()
                # calculate max power at Pv = 0
                Pmp = Imp * Vmp
                # calculate Voc, current must be increasing so flipup()
                Voc = np.interp(np.float64(0), np.flipud(Isys),
                                np.flipud(Vsys))
                Isc = np.interp(np.float64(0), Vsys, Isys)  # calculate Isc
                FF = Pmp / Isc / Voc

    return dict(zip(['imp', 'vmp', 'pmp', 'isc', 'voc', 'ff'],[Imp, Vmp, Pmp, Isc, Voc, FF]))


def simulation_central_inverter(panelizer_object, surface, hoy):
    strings_i, strings_v, strings_g = simulation_string_inverter(panelizer_object, surface, hoy)

    Isrf, Vsrf = circuits.calc_parallel(np.array([strings_i, strings_v]))
    Gsrf = np.sum(strings_g)

    panelizer_object.get_dict_instance([surface])['YIELD'][panelizer_object.topology][
        'irrad'].update({hoy: [np.round(Gsrf, 1)]})
    panelizer_object.get_dict_instance([surface])['CURVES'][panelizer_object.topology]['Isrf'].update(
        {hoy: np.round([Isrf], 3)})
    panelizer_object.get_dict_instance([surface])['CURVES'][panelizer_object.topology]['Vsrf'].update(
        {hoy: np.round([Vsrf], 3)})

    return [Isrf], [Vsrf], [Gsrf]


def simulation_string_inverter(panelizer_object, surface, hoy):
    strings_i = []
    strings_v = []
    strings_g = []
    for string in panelizer_object.get_strings(surface):
        modules_i, modules_v, modules_g = loop_module_simulation(panelizer_object, surface, string, hoy)
        module_curves = np.array([modules_i, modules_v])
        Istr, Vstr = circuits.calc_series(module_curves, panelizer_object.cell)
        input_energy = np.sum(modules_g)
        strings_i.append(Istr)
        strings_v.append(Vstr)
        strings_g.append(input_energy)

        panelizer_object.get_dict_instance([surface, string])['YIELD'][panelizer_object.topology][
            'irrad'].update({hoy: np.round(input_energy, 1)})
        panelizer_object.get_dict_instance([surface, string])['CURVES'][panelizer_object.topology]['Istr'].update({hoy: np.round(Istr, 3)})
        panelizer_object.get_dict_instance([surface, string])['CURVES'][panelizer_object.topology]['Vstr'].update({hoy: np.round(Vstr, 3)})

    return strings_i, strings_v, strings_g


def simulation_micro_inverter(panelizer_object, surface, hoy):
    modules_i = []
    modules_v = []
    modules_g = []
    for string in panelizer_object.get_strings(surface):
        Imods, Vmods, Gmods = loop_module_simulation(panelizer_object, surface, string, hoy)
        modules_i.append(Imods)
        modules_v.append(Vmods)
        modules_g.append(Gmods)
    return utils.flatten_list(modules_i), utils.flatten_list(modules_v), utils.flatten_list(modules_g)


def loop_module_simulation(panelizer_object, surface, string, hoy):
    modules_i = [] #i array of all modules in the string
    modules_v = []
    modules_g = []
    for module in panelizer_object.get_modules(surface, string):
        # chunk hoy here and MP the module simulation
        # write back to a dict for Imod, VMod, and input_energy
        Imod, Vmod, Gmod = simulation_module_yield(panelizer_object, surface, string, module, hoy)
        modules_i.append(Imod)
        modules_v.append(Vmod)
        modules_g.append(Gmod)

        panelizer_object.get_dict_instance([surface, string, module])['YIELD'][panelizer_object.topology][
            'irrad'].update({hoy: np.round(Gmod, 1)})
        panelizer_object.get_dict_instance([surface, string, module])['CURVES'][panelizer_object.topology][
            'Imod'].update({hoy: np.round(Imod, 3)})
        panelizer_object.get_dict_instance([surface, string, module])['CURVES'][panelizer_object.topology][
            'Vmod'].update({hoy: np.round(Vmod, 3)})

    return modules_i, modules_v, modules_g


def archive_wrap_module_simulation(panelizer_object, surface, string, module, hoy):
    if panelizer_object.tmy_dataframe.loc[hoy]['exthorrad_Whm2'] == 0:
        Imod, Vmod = (np.zeros(303), np.zeros(303))
        input_energy = 0
    else:
        Imod, Vmod, input_energy = simulation_module_yield(panelizer_object, surface, string, module, hoy)
    return Imod, Vmod, input_energy


def simulation_module_yield(panelizer_object, surface, string, module, hoy):
    panelizer_object.get_submodule_map(surface, string, module)
    panelizer_object.get_diode_map(surface, string, module)

    module_irrad = panelizer_object.get_cells_irrad_eff(surface, string, module)
    full_irrad = utils.expand_ndarray_2d_3d(module_irrad)
    irrad_hoy = full_irrad[:, :, hoy]

    module_temp = panelizer_object.get_cells_temp(surface, string, module)
    full_temp = utils.expand_ndarray_2d_3d(module_temp)
    temp_hoy = full_temp[:, :, hoy]

    Gmod = np.sum(irrad_hoy * (panelizer_object.cell.width * panelizer_object.cell.width))
    if np.sum(irrad_hoy < panelizer_object.cell.minimum_irradiance_cell) > 0:
        Imod, Vmod = (np.zeros(303), np.zeros(303))
    else:
        Imod, Vmod = panelizer_object.calculate_module_curve(irrad_hoy, temp_hoy)

    return Imod, Vmod, Gmod
