from pvlib import pvsystem, singlediode
import numpy as np
import multiprocess as mp
from tqdm import notebook
from ipv_workbench.simulator import calculations
from ipv_workbench.utilities import circuits, utils, time_utils
import time
import copy


def simulation_central_inverter(panelizer_object, surface):
    strings_i, strings_v, strings_g = simulation_string_inverter(panelizer_object, surface)

    surface_i = {}
    surface_v = {}
    surface_g = {}

    for hoy in panelizer_object.all_hoy:
        strings_i_hoy = [strings_i[string][hoy] for string in panelizer_object.get_strings(surface)]
        strings_v_hoy = [strings_v[string][hoy] for string in panelizer_object.get_strings(surface)]
        strings_g_hoy = [strings_g[string][hoy] for string in panelizer_object.get_strings(surface)]

        Isrf, Vsrf = circuits.calc_parallel(np.array([strings_i_hoy, strings_v_hoy]))
        Gsrf = np.sum(strings_g_hoy)

        surface_i.update({hoy: Isrf})
        surface_v.update({hoy: Vsrf})
        surface_g.update({hoy: Gsrf})

        panelizer_object.get_dict_instance([surface])['CURVES'][panelizer_object.topology]['Isrf'].update(
            {hoy: np.round([Isrf], 5)})
        panelizer_object.get_dict_instance([surface])['CURVES'][panelizer_object.topology]['Vsrf'].update(
            {hoy: np.round([Vsrf], 5)})
        panelizer_object.get_dict_instance([surface])['YIELD'][panelizer_object.topology][
            'irrad'].update({hoy: [np.round(Gsrf, 1)]})

    return surface_i, surface_v, surface_g


def simulation_string_inverter(panelizer_object, surface):
    strings_i = {}
    strings_v = {}
    strings_g = {}
    for string in panelizer_object.get_strings(surface):
        modules = panelizer_object.get_modules(surface, string)

        if panelizer_object.simulation_suite == False:
            module_results_dict = run_mp_simulation(panelizer_object, surface, string)
            modules_i = [module_results_dict[module][0] for module in panelizer_object.get_modules(surface, string)]
            modules_v = [module_results_dict[module][1] for module in panelizer_object.get_modules(surface, string)]
            modules_g = [module_results_dict[module][2] for module in panelizer_object.get_modules(surface, string)]

        strings_i_hoy = {}
        strings_v_hoy = {}
        strings_g_hoy = {}

        for hoy in panelizer_object.all_hoy:

            if panelizer_object.simulation_suite == False:
                modules_i_hoy = [modules_i[n][hoy] for n, m in enumerate(panelizer_object.get_modules(surface, string))]
                modules_v_hoy = [modules_v[n][hoy] for n, m in enumerate(panelizer_object.get_modules(surface, string))]
                modules_g_hoy = [modules_g[n][hoy] for n, m in enumerate(panelizer_object.get_modules(surface, string))]
            else:

                modules_i_hoy = [
                    panelizer_object.get_dict_instance([surface, string, module])['CURVES']['initial_simulation'][
                        'Imod'][
                        hoy] for module in modules]
                modules_v_hoy = [
                    panelizer_object.get_dict_instance([surface, string, module])['CURVES']['initial_simulation'][
                        'Vmod'][
                        hoy] for module in modules]
                modules_g_hoy = [
                    panelizer_object.get_dict_instance([surface, string, module])['YIELD']['initial_simulation'][
                        'irrad'][
                        hoy] for module in modules]

            module_curves = np.array([modules_i_hoy, modules_v_hoy])
            Istr, Vstr = circuits.calc_series(module_curves,
                                              breakdown_voltage=panelizer_object.cell.parameters_dict[
                                                  'breakdown_voltage'],
                                              diode_threshold=panelizer_object.cell.parameters_dict['diode_threshold'],
                                              bypass=False)
            Gstr = np.sum(modules_g_hoy)

            strings_i_hoy.update({hoy: Istr})
            strings_v_hoy.update({hoy: Vstr})
            strings_g_hoy.update({hoy: Gstr})

            panelizer_object.get_dict_instance([surface, string])['CURVES'][panelizer_object.topology][
                'Istr'].update({hoy: np.round(Istr, 5)})
            panelizer_object.get_dict_instance([surface, string])['CURVES'][panelizer_object.topology][
                'Vstr'].update({hoy: np.round(Vstr, 5)})
            panelizer_object.get_dict_instance([surface, string])['YIELD'][panelizer_object.topology][
                'irrad'].update({hoy: np.round(Gstr, 1)})

        strings_i.update({string: strings_i_hoy})
        strings_v.update({string: strings_v_hoy})
        strings_g.update({string: strings_g_hoy})

    return strings_i, strings_v, strings_g


def simulation_micro_inverter(panelizer_object, surface):
    for string in panelizer_object.get_strings(surface):
        if panelizer_object.simulation_suite == False:
            run_mp_simulation(panelizer_object, surface, string)
        else:
            for module in panelizer_object.get_modules(surface, string):

                Imod = panelizer_object.get_dict_instance([surface, string, module])['CURVES']["initial_simulation"][
                    'Imod']
                Vmod = panelizer_object.get_dict_instance([surface, string, module])['CURVES']["initial_simulation"][
                    'Vmod']
                Gmod = panelizer_object.get_dict_instance([surface, string, module])['YIELD']["initial_simulation"][
                    'irrad']

                for hoy in panelizer_object.all_hoy:
                    panelizer_object.get_dict_instance([surface, string, module])['CURVES'][panelizer_object.topology][
                        'Imod'].update({hoy: np.round(Imod[hoy], 5)})
                    panelizer_object.get_dict_instance([surface, string, module])['CURVES'][panelizer_object.topology][
                        'Vmod'].update({hoy: np.round(Vmod[hoy], 5)})
                    panelizer_object.get_dict_instance([surface, string, module])['YIELD'][panelizer_object.topology][
                        'irrad'].update({hoy: np.round(Gmod[hoy], 1)})


def run_mp_simulation(panelizer_object, surface, string):
    # print(string)
    timeseries = panelizer_object.all_hoy
    ncpu = panelizer_object.ncpu
    modules = panelizer_object.get_modules(surface, string)
    module_dict_list = [panelizer_object.get_dict_instance([surface, string, module]) for module in modules]
    module_dict_chunks = np.array_split(module_dict_list, ncpu)
    module_name_chunks = np.array_split(modules, ncpu)

    cell_area = panelizer_object.cell.cell_area
    cell_params = panelizer_object.cell.parameters_dict

    time_start = time.time()

    with mp.Pool(processes=ncpu) as pool:
        # print("    Pool Opened")
        print("    -----------")
        time.sleep(.05)
        args = list(zip(module_dict_chunks,
                        module_name_chunks,
                        [cell_area] * ncpu,
                        [cell_params] * ncpu,
                        [timeseries] * ncpu, ))
        # module_dict, surface, string, module, cell_area, cell_params, hoy_chunk
        result = pool.starmap(mp_module_simulation, args)
        # print("    Result Gathered")
        # time.sleep(1)
        pool.close()
        # print("    Pool closed")
        pool.join()
        # print("    Pool joined")

    # compile results list into one dict 'module',['Imod'['hoy'],'Vmod'['hoy'],'Gmod'['hoy']]
    results_dict = {}
    for r in result:
        results_dict.update(r)

    for module in modules:
        module_dict = results_dict[module]
        Imod = module_dict[0]
        Vmod = module_dict[1]
        Gmod = module_dict[2]

        for hoy in timeseries:
            if panelizer_object.simulation_suite == False:
                panelizer_object.get_dict_instance([surface, string, module])['CURVES'][panelizer_object.topology][
                    'Imod'].update({hoy: np.round(Imod[hoy], 5)})
                panelizer_object.get_dict_instance([surface, string, module])['CURVES'][panelizer_object.topology][
                    'Vmod'].update({hoy: np.round(Vmod[hoy], 5)})
                panelizer_object.get_dict_instance([surface, string, module])['YIELD'][panelizer_object.topology][
                    'irrad'].update({hoy: np.round(Gmod[hoy], 1)})
            else:
                panelizer_object.get_dict_instance([surface, string, module])['CURVES']["initial_simulation"][
                    'Imod'].update({hoy: np.round(Imod[hoy], 5)})
                panelizer_object.get_dict_instance([surface, string, module])['CURVES']["initial_simulation"][
                    'Vmod'].update({hoy: np.round(Vmod[hoy], 5)})
                panelizer_object.get_dict_instance([surface, string, module])['YIELD']["initial_simulation"][
                    'irrad'].update({hoy: np.round(Gmod[hoy], 1)})
        if panelizer_object.simulation_suite == True:
            for topology in panelizer_object.simulation_suite_topologies:
                panelizer_object.get_dict_instance([surface, string, module])['CURVES'][topology] = \
                    copy.deepcopy(
                        panelizer_object.get_dict_instance([surface, string, module])['CURVES']['initial_simulation'])
                panelizer_object.get_dict_instance([surface, string, module])['YIELD'][topology] = \
                    copy.deepcopy(
                        panelizer_object.get_dict_instance([surface, string, module])['YIELD']['initial_simulation'])
    time_end = time.time()
    print(f"Time elapsed for string {string}: {round(time_end - time_start, 2)}s")
    return results_dict


def mp_module_simulation(module_dict_chunk, module_name_chunk, cell_area, cell_params, timeseries):
    module_results = {}

    for n, module_dict in enumerate(module_dict_chunk):
        module = module_name_chunk[n]
        Imod, Vmod, Gmod = timeseries_module_simulation(module_dict, cell_area, cell_params, timeseries)
        module_results.update({module: [Imod, Vmod, Gmod]})

    return module_results


def timeseries_module_simulation(module_dict, cell_area, cell_params, timeseries):
    modules_i_dict = {}
    modules_v_dict = {}
    modules_g_dict = {}

    active_submodule_map = module_dict['MAPS']['SUBMODULES']
    active_diode_map = module_dict['MAPS']['DIODES']
    active_subcell_map = module_dict['MAPS']['SUBCELLS']
    submodules = np.unique(active_submodule_map)
    diodes = np.unique(active_diode_map)
    subcells = np.unique(active_subcell_map)

    module_irrad = module_dict['CELLSIRRADEFF']
    whole_module_irrad = utils.expand_ndarray_2d_3d(module_irrad)

    module_temp = module_dict['CELLSTEMP']
    whole_module_temp = utils.expand_ndarray_2d_3d(module_temp)

    for hoy in timeseries:
        Imod, Vmod, Gmod = simulation_module_yield(whole_module_irrad, whole_module_temp, cell_area, cell_params, hoy,
                                                   active_submodule_map, active_diode_map, active_subcell_map, submodules, diodes, subcells)
        modules_i_dict.update({hoy: Imod})
        modules_v_dict.update({hoy: Vmod})
        modules_g_dict.update({hoy: Gmod})

    return modules_i_dict, modules_v_dict, modules_g_dict


def simulation_module_yield(full_irrad, full_temp, cell_area, cell_params, hoy, active_submodule_map, active_diode_map, active_subcell_map,
                            submodules, diodes, subcells):
    irrad_hoy = full_irrad[:, :, hoy]
    temp_hoy = full_temp[:, :, hoy]

    Gmod = np.sum(irrad_hoy * cell_area)
    if np.sum(irrad_hoy < cell_params['minimum_irradiance_cell']) > 0:
        Imod, Vmod = (np.zeros(303), np.zeros(303))
    else:
        Imod, Vmod = calculations.calculate_module_curve_v2(irrad_hoy, temp_hoy, cell_params, active_submodule_map,
                                                            active_diode_map, active_subcell_map, submodules, diodes, subcells)

    return Imod, Vmod, Gmod
