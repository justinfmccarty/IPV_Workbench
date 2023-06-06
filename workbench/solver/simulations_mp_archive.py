from pvlib import pvsystem, singlediode
import numpy as np
import multiprocess as mp

from workbench.solver import calculations
from workbench.utilities import circuits, utils, time_utils
import time


def simulation_central_inverter(panelizer_object, surface):
    strings_i, strings_v, strings_g = simulation_string_inverter(panelizer_object, surface)

    surface_i = {}
    surface_v = {}
    surface_g = {}

    for hoy in panelizer_object.all_hoy:
        Isrf, Vsrf = circuits.calc_parallel(np.array([strings_i[hoy], strings_v[hoy]]))
        Gsrf = np.sum(strings_g[hoy])

        surface_i.update({hoy: Isrf})
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
            Istr, Vstr = circuits.calc_series(module_curves,
                                              breakdown_voltage=panelizer_object.cell.cell_params['breakdown_voltage'],
                                              diode_threshold=panelizer_object.cell.cell_params['diode_threshold'],
                                              bypass=True)
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


def loop_module_simulation(panelizer_object, surface, string):
    modules_i = []
    modules_v = []
    modules_g = []

    for module in panelizer_object.get_modules(surface, string):
        print(module)
        panelizer_object.get_submodule_map(surface, string, module)
        panelizer_object.get_diode_map(surface, string, module)
        mp_results = run_mp_simulation(panelizer_object, surface, string, module)

        Imod = {}
        Vmod = {}
        Gmod = {}

        for n in np.arange(0, panelizer_object.ncpu):
            Imod.update(mp_results[n]['I'])
            Vmod.update(mp_results[n]['V'])
            Gmod.update(mp_results[n]['G'])

        modules_i.append(Imod)
        modules_v.append(Vmod)
        modules_g.append(Gmod)

        for hoy in panelizer_object.all_hoy:
            panelizer_object.get_dict_instance([surface, string, module])['CURVES'][panelizer_object.topology][
                'Imod'].update({hoy: np.round(Imod[hoy], 3)})
            panelizer_object.get_dict_instance([surface, string, module])['CURVES'][panelizer_object.topology][
                'Vmod'].update({hoy: np.round(Vmod[hoy], 3)})
            panelizer_object.get_dict_instance([surface, string, module])['YIELD'][panelizer_object.topology][
                'irrad'].update({hoy: np.round(Gmod[hoy], 1)})

    return modules_i, modules_v, modules_g


def run_mp_simulation(panelizer_object, surface, string, module):
    total_timesteps = len(panelizer_object.all_hoy)
    module_dict = panelizer_object.get_dict_instance([surface, string, module]).copy()
    ncpu = panelizer_object.ncpu
    hoy_chunks = time_utils.create_timestep_chunks(total_timesteps, ncpu)
    cell_area = panelizer_object.cell.cell_area
    cell_params = panelizer_object.cell.parameters_dict

    time_start = time.time()

    with mp.Pool(processes=ncpu) as pool:
        print("    Pool Opened")
        # time.sleep(1)
        args = list(zip([module_dict] * ncpu,
                        [surface] * ncpu,
                        [string] * ncpu,
                        [module] * ncpu,
                        [cell_area] * ncpu,
                        [cell_params] * ncpu,
                        hoy_chunks))
        # module_dict, surface, string, module, cell_area, cell_params, hoy_chunk
        result = pool.starmap(mp_simulation_wrapper, args)
        print("    Result Gathered")
        # time.sleep(1)
        pool.close()
        print("    Pool closed")
        pool.join()
        print("    Pool joined")

    time_end = time.time()
    print(f"Time elapsed: {round(time_end - time_start, 2)}s")
    return result


def mp_simulation_wrapper(module_dict, surface, string, module, cell_area, cell_params, hoy_chunk):
    modules_i_dict = {}
    modules_v_dict = {}
    modules_g_dict = {}
    for hoy in hoy_chunk:
        Imod, Vmod, Gmod = simulation_module_yield(module_dict, surface, string, module, cell_area, cell_params, hoy)
        modules_i_dict.update({hoy: Imod})
        modules_v_dict.update({hoy: Vmod})
        modules_g_dict.update({hoy: Gmod})

    return {"I": modules_i_dict, "V": modules_v_dict, "G": modules_g_dict}


def simulation_module_yield(module_dict, surface, string, module, cell_area, cell_params, hoy):
    module_irrad = module_dict['CELLSIRRADEFF']
    full_irrad = utils.expand_ndarray_2d_3d(module_irrad)
    irrad_hoy = full_irrad[:, :, hoy]

    module_temp = module_dict['CELLSTEMP']
    full_temp = utils.expand_ndarray_2d_3d(module_temp)
    temp_hoy = full_temp[:, :, hoy]

    Gmod = np.sum(irrad_hoy * cell_area)
    if np.sum(irrad_hoy < cell_params['minimum_irradiance_cell']) > 0:
        Imod, Vmod = (np.zeros(303), np.zeros(303))
    else:
        #     # TODO test if changing this to an acutal simualtion will change the mutlirpcoessing performance
        #     # bottleneck may be that the library is a single object
        Imod, Vmod = calculations.calculate_module_curve(irrad_hoy, temp_hoy, module_dict, cell_params)

        # np.array([Imod.T,Vmod.T])
    return Imod, Vmod, Gmod
