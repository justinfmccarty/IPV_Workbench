from pvlib import pvsystem, singlediode
import numpy as np
import multiprocess as mp
from tqdm import notebook
from ipv_workbench.translators import module_mapping as ipv_mm
from ipv_workbench.simulator import calculations
from ipv_workbench.translators import panelizer
from ipv_workbench.utilities import circuits, utils, time_utils
import time
import copy
import pandas as pd


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

                module_dict = panelizer_object.get_dict_instance([surface, string, module])

                Imod = module_dict['CURVES']["initial_simulation"][
                    'Imod']
                Vmod = module_dict['CURVES']["initial_simulation"][
                    'Vmod']
                Gmod = module_dict['YIELD']["initial_simulation"][
                    'irrad']

                for hoy in panelizer_object.all_hoy:
                    module_dict['CURVES'][panelizer_object.topology][
                        'Imod'].update({hoy: np.round(Imod[hoy], 5)})
                    module_dict['CURVES'][panelizer_object.topology][
                        'Vmod'].update({hoy: np.round(Vmod[hoy], 5)})
                    module_dict['YIELD'][panelizer_object.topology][
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
        mp_results = pool.starmap(mp_module_simulation, args)
        # print("    Result Gathered")
        # time.sleep(1)
        pool.close()
        # print("    Pool closed")
        pool.join()
        # print("    Pool joined")
    unpack_mp_results(mp_results, panelizer_object, surface, string, modules, timeseries)
    time_end = time.time()
    print(f"Time elapsed for string {string}: {round(time_end - time_start, 2)}s")

    # compile results list into one dict 'module',['Imod'['hoy'],'Vmod'['hoy'],'Gmod'['hoy']]

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
        Imod, Vmod = calculations.calculate_module_curve(irrad_hoy, temp_hoy, cell_params, active_submodule_map)

    return Imod, Vmod, Gmod


def compile_system_mp_wrapper_module_loop(panelizer_object, surface, string, tmy_location, dbt, psl, grid_pts, direct_ill, diffuse_ill):
    timeseries = panelizer_object.all_hoy
    ncpu = panelizer_object.ncpu
    modules = panelizer_object.get_modules(surface, string)
    module_dict_list = [panelizer_object.get_dict_instance([surface, string, module_name]) for module_name in modules]
    pv_cells_xyz_arr_list = [np.array(panelizer_object.get_cells_xyz(surface, string, module_name)) for module_name in modules]

    module_dict_chunks = np.array_split(module_dict_list, ncpu)
    module_name_chunks = np.array_split(modules, ncpu)
    pv_cells_xyz_arr_chunks = np.array_split(pv_cells_xyz_arr_list, ncpu)

    string_dict = panelizer_object.get_dict_instance([surface, string])
    string_details = string_dict['DETAILS']
    base_parameters = utils.get_cec_data(string_details['cec_key'], file_path=panelizer_object.CEC_DATA)
    custom_module_data = pd.read_csv(panelizer_object.MODULE_CELL_DATA, index_col='scenario').loc[
        string_details['module_type']].to_dict()

    module_template = string_dict['DETAILS']['module_type']
    cell_type = ipv_mm.get_cell_type(module_template[0])
    orientation = ipv_mm.get_orientation(module_template[1])
    map_file = [fp for fp in panelizer_object.map_files if f"{cell_type}_{orientation}" in fp][0]
    default_submodule_map, default_diode_map, default_subcell_map = utils.read_map_excel(map_file)

    time_start = time.time()

    with mp.Pool(processes=ncpu) as pool:
        # print("    Pool Opened")
        print("    -----------")
        time.sleep(.05)
        args = list(zip(module_dict_chunks,
                        module_name_chunks,
                        [timeseries] * ncpu,
                        [tmy_location] * ncpu,
                        [dbt] * ncpu,
                        [psl] * ncpu,
                        pv_cells_xyz_arr_chunks,
                        [grid_pts] * ncpu,
                        [direct_ill] * ncpu,
                        [diffuse_ill] * ncpu,
                        [base_parameters] * ncpu,
                        [custom_module_data] * ncpu,
                        [default_submodule_map] * ncpu,
                        [default_diode_map] * ncpu,
                        [default_subcell_map] * ncpu,
                        [cell_type] * ncpu))
        # module_dict, surface, string, module, cell_area, cell_params, hoy_chunk

        mp_results = pool.starmap(panelizer.compile_system_multi_core, args)
        # print("    Result Gathered")
        # time.sleep(1)
        pool.close()
        # print("    Pool closed")
        pool.join()
        # print("    Pool joined")
    unpack_mp_results(mp_results, panelizer_object, surface, string, modules, timeseries)
    time_end = time.time()
    print(f"Time elapsed for string {string}: {round(time_end - time_start, 2)}s")


def unpack_mp_results(mp_results, panelizer_object, surface, string, modules, timeseries):
    results_dict = {}
    for r in mp_results:
        results_dict.update(r)

    for module in modules:
        module_dict = panelizer_object.get_dict_instance([surface, string, module])

        module_results_dict = results_dict[module]
        Imod = module_results_dict[0]
        Vmod = module_results_dict[1]
        Gmod = module_results_dict[2]

        for hoy in timeseries:
            if panelizer_object.simulation_suite == False:
                module_dict['CURVES'][panelizer_object.topology][
                    'Imod'].update({hoy: np.round(Imod[hoy], 5)})
                module_dict['CURVES'][panelizer_object.topology][
                    'Vmod'].update({hoy: np.round(Vmod[hoy], 5)})
                module_dict['YIELD'][panelizer_object.topology][
                    'irrad'].update({hoy: np.round(Gmod[hoy], 1)})
            else:
                module_dict['CURVES']["initial_simulation"][
                    'Imod'].update({hoy: np.round(Imod[hoy], 5)})
                module_dict['CURVES']["initial_simulation"][
                    'Vmod'].update({hoy: np.round(Vmod[hoy], 5)})
                module_dict['YIELD']["initial_simulation"][
                    'irrad'].update({hoy: np.round(Gmod[hoy], 1)})

        if panelizer_object.simulation_suite == True:
            for topology in panelizer_object.simulation_suite_topologies:
                module_dict['CURVES'][topology] = \
                    copy.deepcopy(
                        module_dict['CURVES']['initial_simulation'])
                module_dict['YIELD'][topology] = \
                    copy.deepcopy(
                        module_dict['YIELD']['initial_simulation'])

