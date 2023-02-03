from pvlib import pvsystem, singlediode
import numpy as np
import multiprocessing as mp
from ipv_workbench.utilities import circuits, utils, time_utils


def simulation_central_inverter(panelizer_object, surface):
    strings_i, strings_v, strings_g = simulation_string_inverter(panelizer_object, surface)

    surface_i = {}
    surface_v = {}
    surface_g = {}

    for hoy in panelizer_object.all_hoy:
        Isrf, Vsrf = circuits.calc_parallel(np.array([strings_i[hoy], strings_v[hoy]]))
        Gsrf = np.sum(strings_g[hoy])

        surface_i.update({hoy:Isrf})
        surface_v.update({hoy: Vsrf})
        surface_g.update({hoy: Gsrf})

        panelizer_object.get_dict_instance([surface])['CURVES'][panelizer_object.topology]['Isrf'].update(
            {hoy: np.round([Isrf], 3)})
        panelizer_object.get_dict_instance([surface])['CURVES'][panelizer_object.topology]['Vsrf'].update(
            {hoy: np.round([Vsrf], 3)})
        panelizer_object.get_dict_instance([surface])['YIELD'][panelizer_object.topology][
            'irrad'].update({hoy: [np.round(Gsrf, 1)]})

    return surface_i, surface_v, surface_g


def simulation_string_inverter(panelizer_object, surface):
    strings_i = {}
    strings_v = {}
    strings_g = {}
    for string in panelizer_object.get_strings(surface):
        modules_i, modules_v, modules_g = loop_module_simulation(panelizer_object, surface, string)

        for hoy in panelizer_object.all_hoy:
            module_curves = np.array([modules_i[hoy], modules_v[hoy]])
            Istr, Vstr = circuits.calc_series(module_curves, panelizer_object.cell)
            input_energy = np.sum(modules_g[hoy])

            strings_i.update({hoy: Istr})
            strings_v.update({hoy: Vstr})
            strings_g.update({hoy: input_energy})

            panelizer_object.get_dict_instance([surface, string])['CURVES'][panelizer_object.topology][
                'Istr'].update({hoy: np.round(Istr, 3)})
            panelizer_object.get_dict_instance([surface, string])['CURVES'][panelizer_object.topology][
                'Vstr'].update({hoy: np.round(Vstr, 3)})
            panelizer_object.get_dict_instance([surface, string])['YIELD'][panelizer_object.topology][
                'irrad'].update({hoy: np.round(input_energy, 1)})

    return strings_i, strings_v, strings_g


def simulation_micro_inverter(panelizer_object, surface):
    for string in panelizer_object.get_strings(surface):
        loop_module_simulation(panelizer_object, surface, string)

#
# def loop_module_simulation(panelizer_object, surface, string, hoy):
#     modules_i = []
#     modules_v = []
#     modules_g = []
#     for module in panelizer_object.get_modules(surface, string):
#         # chunk hoy here and MP the module simulation
#         # write back to a dict for Imod, VMod, and input_energy
#         Imod, Vmod, Gmod = simulation_module_yield(panelizer_object, surface, string, module, hoy)
#         modules_i.append(Imod)
#         modules_v.append(Vmod)
#         modules_g.append(Gmod)
#
#         panelizer_object.get_dict_instance([surface, string, module])['YIELD'][panelizer_object.topology][
#             'irrad'].update({hoy: np.round(Gmod, 1)})
#         panelizer_object.get_dict_instance([surface, string, module])['CURVES'][panelizer_object.topology][
#             'Imod'].update({hoy: np.round(Imod, 3)})
#         panelizer_object.get_dict_instance([surface, string, module])['CURVES'][panelizer_object.topology][
#             'Vmod'].update({hoy: np.round(Vmod, 3)})
#
#     return modules_i, modules_v, modules_g


def loop_module_simulation(panelizer_object, surface, string):

    modules_i = {}
    modules_v = {}
    modules_g = {}

    for module in panelizer_object.get_modules(surface, string):
        mp_results = run_mp_simulation(panelizer_object, surface, string, module)
        Imod = mp_results[0]
        Vmod = mp_results[1]
        Gmod = mp_results[2]

        for hoy in panelizer_object.all_hoy:
            modules_i.update({hoy:Imod})
            modules_v.update({hoy:Vmod})
            modules_g.update({hoy:Gmod})

            panelizer_object.get_dict_instance([surface, string, module])['YIELD'][panelizer_object.topology][
                'irrad'].update({hoy: np.round(Gmod, 1)})
            panelizer_object.get_dict_instance([surface, string, module])['CURVES'][panelizer_object.topology][
                'Imod'].update({hoy: np.round(Imod, 3)})
            panelizer_object.get_dict_instance([surface, string, module])['CURVES'][panelizer_object.topology][
                'Vmod'].update({hoy: np.round(Vmod, 3)})

    return modules_i, modules_v, modules_g

def run_mp_simulation(panelizer_object, surface, string, module):

    total_timesteps = len(panelizer_object.all_hoy)

    ncpu = panelizer_object.ncpu
    hoy_chunks = time_utils.create_timestep_chunks(total_timesteps, ncpu)

    pool = mp.Pool(ncpu)

    args = zip([panelizer_object]*ncpu,
                [surface]*ncpu,
                [string]*ncpu,
                [module]*ncpu,
                hoy_chunks)

    result = pool.starmap(mp_simulation_wrapper, args)
    pool.close()
    return result

def mp_simulation_wrapper(panelizer_object, surface, string, module, hoy_chunk):
    modules_i_dict = {}
    modules_v_dict = {}
    modules_g_dict = {}

    for hoy in hoy_chunk:
        Imod, Vmod, Gmod = simulation_module_yield(panelizer_object, surface, string, module, hoy)
        modules_i_dict.update({hoy: Imod})
        modules_v_dict.update({hoy: Vmod})
        modules_g_dict.update({hoy: Gmod})

    return modules_i_dict, modules_v_dict, modules_g_dict

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
