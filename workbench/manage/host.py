import time
import os
import numpy as np
import pandas as pd
import multiprocessing as mp

import workbench.device.devices
from workbench.utilities import general, circuits, temporal, io, multi
from workbench.old_solver import calculations as ipv_calc, compile_mp
from workbench.old_solver import simulations
from workbench.device import devices, stringer
from workbench.simulations import method_effective_irradiance as ipv_irrad, method_simple_power, method_module_iv, \
    method_effective_irradiance, method_topology_solver
from workbench.manage import module_mapping as ipv_mm


class Host:
    def __init__(self, project):

        self.project = project
        self.cell = None
        self.module = self.project.analysis_device_id
        self.topology = None
        self.tmy_dataframe = io.read_epw(self.project.TMY_FILE)
        self.tmy_location = general.tmy_location(self.project.TMY_FILE)
        self.analysis_period = (None, None, None)  # start, end, increment
        self.all_hoy = temporal.datetime_index_to_hoy_array(self.tmy_dataframe.index)
        self.annual_hoy = np.arange(0, 8760, project.analysis_temporal_resolution)
        self.ncpu = mp.cpu_count() - 1
        self.multiprocess = True
        self.simulation_suite = False
        self.simulation_suite_topologies = ['micro_inverter', 'string_inverter', 'central_inverter']
        self.analysis_location = None
        self.analysis_year = None
        self.hourly_resolution = None

        # v1
        self.raw_panelizer = f"{self.module}_solar_glass_{self.project.management_host_name}_raw.pickle"
        self.panelizer_file = os.path.join(self.project.SCEN_DIR, 'panelizer', self.raw_panelizer)
        self.irradiance_range = general.create_linspace(0, 1500, project.analysis_map_irradiance_resolution)
        self.temperature_range = general.create_linspace(-40, 40, project.analysis_map_temperature_resolution)
        self.set_panelizer_dict()

    def set_panelizer_dict(self):
        # print("Setting dict from raw panelizer file")
        if type(self.raw_panelizer) is dict:
            self.input_type = 'dict'
            self.panelizer_file = None
            self.panelizer_dict = self.raw_panelizer
        else:
            # print(object_path)
            if os.path.exists(self.panelizer_file):
                self.input_type = 'file'
                self.file_name = self.panelizer_file.split(os.sep)[-1].split(".")[0]
                self.panelizer_dict = io.read_pickle(self.panelizer_file)
            else:
                print(self.panelizer_file)
                print("Input detected as string but file path does not exist.")

        self.object_type = list(self.panelizer_dict.keys())[0]
        self.object_surfaces = self.get_surfaces()
        self.add_all_module_data()

        # build device IV library to pull from later
        surface = self.object_surfaces[0]
        module = self.get_modules(surface)[0]
        module_dict = self.get_dict_instance([surface, module])
        self.device_iv_dict = devices.build_device_iv_library(module_dict['Parameters'], self.irradiance_range,
                                                              self.temperature_range)

    def add_module_level_data(self, surface, module):
        surface_details = self.get_dict_instance([surface])['Details']
        custom_module_data = pd.read_csv(self.project.module_cell_data, index_col='general_device_id').loc[
            surface_details['device_id']].to_dict()

        module_dict = self.get_dict_instance([surface, module])
        module_dict['Parameters'] = workbench.device.devices.build_parameter_dict(module_dict,
                                                                                  custom_module_data)
        module_dict['map_idx_arr'] = devices.create_module_idx_arr(module_dict)
        module_dict['Maps']['Submodules'] = devices.find_device_map(module_dict['Parameters'], self.project,
                                                                    map_type='submodule')
        module_dict['Maps']['Subcells'] = devices.find_device_map(module_dict['Parameters'], self.project,
                                                                  map_type='subcell')
        module_dict['Maps']['Diodes'] = devices.find_device_map(module_dict['Parameters'], self.project,
                                                                map_type='subdiode')

    def add_all_module_data(self):
        for surface in self.object_surfaces:
            for n, module in enumerate(self.get_modules(surface)):
                self.add_module_level_data(surface, module)

    # def set_tmy_data(self):
    #     self.TMY_DIR = os.path.join(self.locator.RESOURCES_DIR, 'tmy')
    #     utils.directory_creator(self.TMY_DIR)
    #     self.wea_file = os.path.join(self.TMY_DIR, f"{self.analysis_location}_{self.analysis_year}.wea")
    #     self.tmy_file = os.path.join(self.TMY_DIR, f"{self.analysis_location}_{self.analysis_year}.epw")
    #     self.tmy_dataframe = utils.read_epw(self.tmy_file)
    #     self.tmy_location = utils.tmy_location(self.tmy_file)

    # def correct_maps(self):
    #     if self.input_type == 'file':
    #         for surface in self.get_surfaces():
    #             strings = self.get_strings(surface)
    #             for string in strings:
    #                 modules = self.get_modules(surface, string)
    #                 for module in modules:
    #                     mod_dict = self.get_dict_instance([surface, string, module])
    #                     mod_dict['Maps']['Diodes'] = general.flip_maps(mod_dict['Maps']['Diodes'])
    #                     mod_dict['Maps']['Submodules'] = general.flip_maps(mod_dict['Maps']['Submodules'])
    #                     mod_dict['Maps']['Subcells'] = general.flip_maps(mod_dict['Maps']['Subcells'])

    def set_analysis_period(self):
        if self.hourly_resolution is None:
            hourly_resolution = 1
        else:
            hourly_resolution = self.hourly_resolution
        self.all_hoy = temporal.build_analysis_period(self.project.sunup_array, hourly_resolution)

        return self.all_hoy

    # def reset_results_dict(self, surface):
    #     self.get_dict_instance([surface])['RESULTSDICT'] = {}
    #
    # def get_results_dict(self, surface):
    #     return self.get_dict_instance([surface])['RESULTSDICT']

    def get_surfaces(self):
        if self.project.management_exclude_surfaces == None:
            self.active_surfaces = list(self.panelizer_dict[self.object_type]['Surfaces'].keys())
        else:
            active_surfaces_temp = list(self.panelizer_dict[self.object_type]['Surfaces'].keys())
            self.active_surfaces = [srf for srf in active_surfaces_temp if
                                    srf not in self.project.management_exclude_surfaces]
        return self.active_surfaces

    # def get_strings(self, surface_name):
    #     self.active_strings = list(self.panelizer_dict[self.object_type]['Surfaces'][surface_name]['Strings'].keys())
    #     return self.active_strings


    def get_string_keys(self, surface):
        return list(self.get_dict_instance([surface])['Strings'].keys())


    def get_modules_on_string(self, surface, string_key):
        return self.get_dict_instance([surface])['Strings'][string_key]['modules']


    def get_modules(self, surface_name):
        self.active_modules = list(
            self.panelizer_dict[self.object_type]['Surfaces'][surface_name]['Modules'].keys())
        return self.active_modules

    def get_surface_capacities(self, surface_name):
        modules = self.get_modules(surface_name)
        capacities = list(
            set([self.get_dict_instance([surface_name, module])['Parameters']['param_actual_capacity_Wp'] for module in
                 modules]))
        capacities.sort()
        self.surface_capacities = capacities
        return self.surface_capacities

    def get_submodule_map(self, surface_name, module_name):
        self.active_submodule_map = \
            self.panelizer_dict[self.object_type]['Surfaces'][surface_name]['Modules'][
                module_name]['Maps']['Submodules']
        self.submodules = np.unique(self.active_submodule_map)
        return self.active_submodule_map

    def get_diode_map(self, surface_name, module_name):
        self.active_diode_map = \
            self.panelizer_dict[self.object_type]['Surfaces'][surface_name]['Modules'][
                module_name]['Maps']['Diodes']
        self.diodes = np.unique(self.active_diode_map)
        return self.active_diode_map

    def get_cells_xyz(self, surface_name, module_name):
        self.cells_xyz = \
            self.panelizer_dict[self.object_type]['Surfaces'][surface_name]['Modules'][
                module_name]['CellsXYZ']
        return np.array(self.cells_xyz)

    def get_cells_normals(self, surface_name, module_name):
        self.cells_normals = \
            self.panelizer_dict[self.object_type]['Surfaces'][surface_name]['Modules'][
                module_name]['CellsNormals']
        return self.cells_normals

    def get_cells_irrad(self, surface_name, module_name):
        self.cells_irrad = \
            self.panelizer_dict[self.object_type]['Surfaces'][surface_name]['Modules'][
                module_name]['CellsIrrad']
        return self.cells_irrad

    def get_cells_irrad_eff(self, surface_name, module_name):
        self.cells_irrad_eff = \
            self.panelizer_dict[self.object_type]['Surfaces'][surface_name]['Modules'][
                module_name]['CellsIrradEff']
        return self.cells_irrad_eff

    def get_cells_temp(self, surface_name, module_name):
        self.cells_temp = \
            self.panelizer_dict[self.object_type]['Surfaces'][surface_name]['Modules'][
                module_name]['CellsTemp']
        return self.cells_temp

    def calc_irrad_eff(self):
        # effective_irradiance.calculate_effective_irradiance()
        self.cells_irrad_eff = None
        return self.cells_irrad_eff

    def get_dict_instance(self, keys, strings_or_modules='modules'):
        if len(keys) == 0:
            return self.panelizer_dict[self.object_type]
        elif len(keys) == 1:
            return self.panelizer_dict[self.object_type]['Surfaces'][keys[0]]
        elif len(keys) == 2:
            if strings_or_modules == 'modules':
                return self.panelizer_dict[self.object_type]['Surfaces'][keys[0]]['Modules'][keys[1]]
            else:
                return self.panelizer_dict[self.object_type]['Surfaces'][keys[0]]['Strings'][keys[1]]
        elif len(keys) == 3:
            if strings_or_modules == 'modules':
                return self.panelizer_dict[self.object_type]['Surfaces'][keys[0]]['Modules'][keys[1]][keys[2]]
            else:
                return self.panelizer_dict[self.object_type]['Surfaces'][keys[0]]['Strings'][keys[1]][keys[2]]

    def solve_module_center_pts(self, surface):
        start_time = time.time()
        dbt = self.tmy_dataframe['drybulb_C'].values[self.all_hoy]
        psl = self.tmy_dataframe['atmos_Pa'].values[self.all_hoy]

        radiance_surface_key = self.get_dict_instance([surface])['Details']['radiance_surface_label']
        direct_irrad = io.load_irradiance_file(self.project, radiance_surface_key, "direct").values[
            self.all_hoy]
        diffuse_irrad = io.load_irradiance_file(self.project, radiance_surface_key, "diffuse").values[
            self.all_hoy]
        grid_pts = io.load_grid_file(self.project, radiance_surface_key)
        sensor_pts_xyz_arr = grid_pts[['X', 'Y', 'Z']].values

        # for string in self.get_strings(surface):
        #     string_dict = self.get_dict_instance([surface, string])
        #     string_details = string_dict['DETAILS']

        # surface_details = self.get_dict_instance([surface])['Details']

        # base_parameters = io.get_cec_data(surface_details['cec_key'], file_path=self.project.cec_data)
        # custom_module_data = pd.read_csv(self.project.module_cell_data, index_col='general_device_id').loc[
        #     surface_details['device_id']].to_dict()

        for module in self.get_modules(surface):
            module_dict = self.get_dict_instance([surface, module])

            # module_dict['Parameters'] = workbench.device.devices.build_parameter_dict(module_dict, custom_module_data)
            module_area, irradiance_result, power_result = method_simple_power.module_center_pt(module_dict,
                                                                                                sensor_pts_xyz_arr,
                                                                                                direct_irrad,
                                                                                                diffuse_irrad,
                                                                                                self.all_hoy,
                                                                                                self.tmy_location,
                                                                                                psl,
                                                                                                dbt)

            if "center_point" not in module_dict['Yield'].keys():
                module_dict['Yield']['center_point'] = {"pmp": {},
                                                        "irrad": {},
                                                        'area': {}}

            module_dict['Yield']['center_point']['pmp'] = dict(zip(self.all_hoy, power_result))
            module_dict['Yield']['center_point']['irrad'] = dict(zip(self.all_hoy, irradiance_result))
            module_dict['Yield']['center_point']['area'] = dict(zip(self.all_hoy, module_area))
        total_time = round(time.time() - start_time, 2)
        self.project.log(total_time, "module-center-point")

    def solve_module_cell_pts(self, surface):
        start_time = time.time()
        dbt = self.tmy_dataframe['drybulb_C'].values[self.all_hoy]
        psl = self.tmy_dataframe['atmos_Pa'].values[self.all_hoy]

        radiance_surface_key = self.get_dict_instance([surface])['Details']['radiance_surface_label']
        direct_irrad = io.load_irradiance_file(self.project, radiance_surface_key, "direct").values[
            self.all_hoy]
        diffuse_irrad = io.load_irradiance_file(self.project, radiance_surface_key, "diffuse").values[
            self.all_hoy]
        grid_pts = io.load_grid_file(self.project, radiance_surface_key)
        sensor_pts_xyz_arr = grid_pts[['X', 'Y', 'Z']].values

        # surface_details = self.get_dict_instance([surface])['Details']
        # # base_parameters = io.get_cec_data(surface_details['cec_key'], file_path=self.project.cec_data)
        # custom_module_data = pd.read_csv(self.project.module_cell_data, index_col='general_device_id').loc[
        #     surface_details['device_id']].to_dict()

        modules = self.get_modules(surface)
        for module_name in modules:
            module_dict = self.get_dict_instance([surface, module_name])
            pv_cells_xyz_arr = np.array(self.get_cells_xyz(surface, module_name))

            # module_dict['Parameters'] = workbench.device.devices.build_parameter_dict(module_dict, custom_module_data)

            module_area, irradiance_result, power_result = method_simple_power.module_cell_pt(module_dict,
                                                                                              pv_cells_xyz_arr,
                                                                                              sensor_pts_xyz_arr,
                                                                                              direct_irrad,
                                                                                              diffuse_irrad,
                                                                                              self.all_hoy,
                                                                                              self.tmy_location,
                                                                                              psl,
                                                                                              dbt)

            if "cell_point" not in module_dict['Yield'].keys():
                module_dict['Yield']['cell_point'] = {"pmp": {},
                                                      "irrad": {},
                                                      "area": {}}

            module_dict['Yield']['cell_point']['pmp'] = power_result
            module_dict['Yield']['cell_point']['irrad'] = irradiance_result
            module_dict['Yield']['cell_point']['area'] = module_area

        total_time = round(time.time() - start_time, 2)
        self.project.log(total_time, "module-cell-point")

    def sum_simple_module_results(self, surface, simple_topology):

        module_results_pmp = []
        module_results_irrad = []
        area_results = []

        for module in self.get_modules(surface):
            module_results_pmp.append(
                self.get_tabular_results([surface, module], simple_topology, rename_cols=False)[
                    'pmp'])
            module_results_irrad.append(
                self.get_tabular_results([surface, module], simple_topology, rename_cols=False)[
                    'irrad'])
            area_results.append(
                self.get_tabular_results([surface, module], simple_topology, rename_cols=False)[
                    'area'])
        pmp = pd.concat(module_results_pmp, axis=1).sum(axis=1).rename('pmp')
        irrad = pd.concat(module_results_irrad, axis=1).sum(axis=1).rename('irrad')
        area = pd.concat(area_results, axis=1).sum(axis=1).rename('area')

        return pd.concat([pmp, irrad, area], axis=1)

    def solve_all_modules_iv_curve(self, surface):

        start_time = time.time()
        dbt = self.tmy_dataframe['drybulb_C'].values[self.all_hoy]
        psl = self.tmy_dataframe['atmos_Pa'].values[self.all_hoy]

        radiance_surface_key = self.get_dict_instance([surface])['Details']['radiance_surface_label']
        direct_irrad = io.load_irradiance_file(self.project, radiance_surface_key, "direct").values[
            self.all_hoy]
        diffuse_irrad = io.load_irradiance_file(self.project, radiance_surface_key, "diffuse").values[
            self.all_hoy]
        grid_pts = io.load_grid_file(self.project, radiance_surface_key)
        sensor_pts_xyz_arr = grid_pts[['X', 'Y', 'Z']].values

        modules = self.get_modules(surface)
        for module_name in modules:
            module_dict = self.get_dict_instance([surface, module_name])
            pv_cells_xyz_arr = np.array(self.get_cells_xyz(surface, module_name))
            G_dir_ann_mod = general.collect_raw_irradiance(pv_cells_xyz_arr, sensor_pts_xyz_arr, direct_irrad)
            G_diff_ann_mod = general.collect_raw_irradiance(pv_cells_xyz_arr, sensor_pts_xyz_arr, diffuse_irrad)

            G_eff_ann_mod = method_effective_irradiance.calculate_effective_irradiance_timeseries(G_dir_ann_mod,
                                                                                                  G_diff_ann_mod,
                                                                                                  module_dict[
                                                                                                      'Details'][
                                                                                                      'panelizer_normal'],
                                                                                                  self.all_hoy,
                                                                                                  self.tmy_location,
                                                                                                  psl,
                                                                                                  dbt,
                                                                                                  module_dict['Layers'][
                                                                                                      'panelizer_front_film'])

            Imod, Vmod = method_module_iv.solve_module_iv_curve(self, G_eff_ann_mod, module_dict, self.all_hoy, dbt)
            module_dict['Curves']['Imod'] = Imod
            module_dict['Curves']['Vmod'] = Vmod


            Gmod = np.sum(G_eff_ann_mod * module_dict['Parameters']['param_one_cell_area_m2'], axis=1)
            module_dict['Yield']['initial_simulation']['irrad'] = dict(zip(self.all_hoy, np.round(Gmod, 3)))

        # return Imod, Vmod

    def string_surface(self, surface):
        stringer.building_string_map(self, surface)

    def string_building(self):
        for surface in self.get_surfaces():
            self.string_surface(surface)

    def solve_string_iv_curves(self, surface, string):
        method_topology_solver.solve_string_inverter_iv(self, surface, string)
        method_topology_solver.solve_string_inverter_mpp(self, surface, string)

    def solve_all_string_iv_curves(self, surface):
        for string_key in self.get_string_keys(surface):
            self.solve_string_iv_curves(surface, string_key)


    def solve_all_string_iv_curves_building(self):
        for surface in self.get_surfaces():
            self.string_surface(surface)

    def solve_surface_central_inverter(self, surface):
        method_topology_solver.solve_central_inverter_iv(self, surface)
        method_topology_solver.solve_central_inverter_mpp(self, surface)

    def solve_all_central_inverter_building(self):
        for surface in self.get_surfaces():
            self.solve_surface_central_inverter(surface)

    def write_key_parameters(self):
        object_dict = self.get_dict_instance([])

        object_capacity = []
        object_area = []
        object_cells_area = []
        for surface in self.get_surfaces():
            surface_dict = self.get_dict_instance([surface])

            surface_capacity = []
            surface_area = []
            surface_cells_area = []
            for string_key in self.get_string_keys(surface):
                string_dict = self.get_dict_instance([surface])['Strings'][string_key]
                modules = string_dict['modules']

                string_capacity = []
                string_area = []
                string_cells_area = []
                for module in modules:
                    module_dict = self.get_dict_instance([surface, module])
                    module_capacity = module_dict['Parameters']['param_actual_capacity_Wp']
                    module_area = module_dict['Parameters']['param_actual_module_area_m2']
                    module_cells_area = module_dict['Parameters']['param_actual_total_cell_area_m2']
                    string_capacity.append(module_capacity)
                    string_area.append(module_area)
                    string_cells_area.append(module_cells_area)

                string_dict['Details']['installed_capacity_Wp'] = np.sum(string_capacity)
                string_dict['Details']['installed_area_m2'] = np.sum(string_area)
                string_dict['Details']['installed_cell_area_m2'] = np.sum(string_cells_area)
                surface_capacity.append(np.sum(string_capacity))
                surface_area.append(np.sum(string_area))
                surface_cells_area.append(np.sum(string_cells_area))

            surface_dict['Details']['installed_capacity_Wp'] = np.sum(surface_capacity)
            surface_dict['Details']['installed_area_m2'] = np.sum(surface_area)
            surface_dict['Details']['installed_cell_area_m2'] = np.sum(surface_cells_area)
            object_capacity.append(np.sum(surface_capacity))
            object_area.append(np.sum(surface_area))
            object_cells_area.append(np.sum(surface_cells_area))

        object_dict['Details']['installed_capacity_Wp'] = np.sum(object_capacity)
        object_dict['Details']['installed_area_m2'] = np.sum(object_area)
        object_dict['Details']['installed_cell_area_m2'] = np.sum(object_cells_area)

    ######### EVERYTHING BELOW IS AN UNKNOWN AS OF 22 NOVEMBER 2023
    def calculate_module_curve(self, irradiance_hoy, temperature_hoy, submodule_map, subdiode_map):
        # TODO break apart into constituent pieces
        # TODO add subcell routine

        submodule_i = []
        submodule_v = []
        submodules = np.unique(submodule_map.flatten())
        for submodule_key in submodules:
            submodule_mask = submodule_map == submodule_key
            submodule_diode = subdiode_map[submodule_mask]
            diodes = np.unique(submodule_diode)
            submodule_irrad = irradiance_hoy[submodule_mask]
            submodule_temp = temperature_hoy[submodule_mask]
            diode_i = []
            diode_v = []
            for diode_key in diodes:
                diode_mask = submodule_diode == diode_key
                submodule_subdiode_irrad = submodule_irrad[diode_mask]
                submodule_subdiode_temp = submodule_temp[diode_mask]
                sub_diode_curves = self.cell.retrieve_curves_multiple_cells(submodule_subdiode_irrad,
                                                                            submodule_subdiode_temp)
                i, v = circuits.calc_series(sub_diode_curves,
                                            breakdown_voltage=self.cell.breakdown_voltage,
                                            diode_threshold=self.cell.diode_threshold,
                                            bypass=False)
                diode_i.append(i)
                diode_v.append(v)

            # calc series with bypass diodes
            diode_curves = np.array([diode_i, diode_v])
            i, v = circuits.calc_series(diode_curves,
                                        breakdown_voltage=self.cell.breakdown_voltage,
                                        diode_threshold=self.cell.diode_threshold,
                                        bypass=True)
            submodule_i.append(i)
            submodule_v.append(v)
        submodule_curves = np.array([submodule_i, submodule_v])
        Imod, Vmod = circuits.calc_parallel(submodule_curves)
        return Imod, Vmod

    def print_system_names(self):
        for surface in self.get_surfaces():
            print(surface)
            for string_ in self.get_strings(surface):
                print("  ", string_)
                for module in self.get_modules(surface, string_):
                    print("    ", module)

    def transfer_initial(self):
        for surface_ in self.get_surfaces():
            for string_ in self.get_strings(surface_):
                for module in self.get_modules(surface_, string_):
                    for topology in self.simulation_suite_topologies:
                        module_dict = self.get_dict_instance(([surface_, string_, module]))
                        # Imod_init = module_dict['CURVES']['Imod']
                        # Vmod_init = module_dict['CURVES']['Vmod']
                        # module_dict['CURVES'][topology]['Imod'] = copy.deepcopy(Imod_init)
                        # module_dict['CURVES'][topology]['Vmod'] = copy.deepcopy(Vmod_init)
                        module_dict['YIELD'][topology]['irrad'] = module_dict['YIELD']["initial_simulation"]['irrad']

                        # for k in ['actual_module_area_m2', 'actual_capacity_Wp']:
                        #     module_dict['YIELD'][topology][k] = module_dict['PARAMETERS'][k]

    def delete_cell_location(self):
        for surface in self.get_surfaces():
            for module in self.get_modules(surface):
                module_dict = self.get_dict_instance([surface, module])
                module_dict.pop('CellsXYZ', None)
                module_dict.pop('CellsNormals', None)



    def solve_system(self, surface, hoy):
        if self.topology == 'central_inverter':
            simulations.simulation_central_inverter(
                self, surface, hoy)
        elif self.topology == 'string_inverter':
            simulations.simulation_string_inverter(
                self, surface, hoy)
        elif self.topology == 'micro_inverter':
            simulations.simulation_micro_inverter(
                self, surface, hoy)

    def write_first_level_results(self, surface, string=None):
        if self.topology == 'string_inverter':
            self.write_string_inverter(surface)
        elif self.topology == 'central_inverter':
            self.write_central_inverter(surface)
        elif self.topology == 'micro_inverter':
            self.write_micro_inverter(surface)

    def write_micro_inverter(self, surface):

        for module in self.get_modules(surface):

            module_dict = self.get_dict_instance([surface, module])
            mpp_results_dict = topology_solver.solve_micro_inverter_mpp(self, module_dict)

            # write these results to the module dict
            for hoy in self.all_hoy:
                module_dict[
                    'Yield'][self.topology]['imp'].update({hoy: mpp_results_dict[hoy]['imp']})
                module_dict[
                    'Yield'][self.topology]['vmp'].update({hoy: mpp_results_dict[hoy]['vmp']})
                module_dict[
                    'Yield'][self.topology]['pmp'].update({hoy: mpp_results_dict[hoy]['pmp']})
                # module_dict[
                #     'YIELD'][self.topology]['isc'].update({hoy: mpp_results_dict[hoy]['isc']})
                # module_dict[
                #     'YIELD'][self.topology]['voc'].update({hoy: mpp_results_dict[hoy]['voc']})
                # module_dict[
                #     'YIELD'][self.topology]['ff'].update({hoy: mpp_results_dict[hoy]['ff']})
                module_dict[
                    'YIELD'][self.topology]['eff'].update({hoy: mpp_results_dict[hoy]['eff']})

    def write_string_inverter(self, surface):

        for string in self.get_strings(surface):
            string_dict = self.get_dict_instance([surface, string])

            string_iv_results = topology_solver.solve_string_inverter_iv(self, surface, string)

            string_dict['CURVES']['Istr'] = string_iv_results[0]
            string_dict['CURVES']['Vstr'] = string_iv_results[1]
            string_dict['YIELD'][self.topology]['irrad'] = string_iv_results[2]

            # copy over to the central inverter which needs them
            # string_dict['CURVES']["central_inverter"]['Istr'] = string_iv_results[0]
            # string_dict['CURVES']["central_inverter"]['Vstr'] = string_iv_results[1]
            string_dict['YIELD']["central_inverter"]['irrad'] = string_iv_results[2]

            mpp_results_dict = topology_solver.solve_string_inverter_mpp(self, string_dict)

            for hoy in self.all_hoy:
                string_dict[
                    'YIELD'][self.topology]['imp'].update({hoy: mpp_results_dict[hoy]['imp']})
                string_dict[
                    'YIELD'][self.topology]['vmp'].update({hoy: mpp_results_dict[hoy]['vmp']})
                string_dict[
                    'YIELD'][self.topology]['pmp'].update({hoy: mpp_results_dict[hoy]['pmp']})
                # string_dict[
                #     'YIELD'][self.topology]['isc'].update({hoy: mpp_results_dict[hoy]['isc']})
                # string_dict[
                #     'YIELD'][self.topology]['voc'].update({hoy: mpp_results_dict[hoy]['voc']})
                # string_dict[
                #     'YIELD'][self.topology]['ff'].update({hoy: mpp_results_dict[hoy]['ff']})
                string_dict[
                    'YIELD'][self.topology]['eff'].update({hoy: mpp_results_dict[hoy]['eff']})

                # get the current at maximum power point to send to the module to calculate the power at MPP on the module
                operating_imp = np.round(string_dict['YIELD'][self.topology]['imp'][hoy], 3)

                # loop through all modules in the string to calculate the power of the module using imp
                # then write this and then calculat ethe efficiency using the svaed irrad
                for module in self.get_modules(surface, string):
                    module_dict = self.get_dict_instance([surface, string, module])

                    Imod_hoy = module_dict['CURVES']['Imod'][hoy]
                    Vmod_hoy = module_dict['CURVES']['Vmod'][hoy]

                    # interp the relationship between I and V for the hour o the module using the oeprating I
                    operating_vmp = np.round(np.interp(operating_imp, np.flipud(Imod_hoy),
                                                       np.flipud(Vmod_hoy)), 5)

                    module_input_energy = self.get_dict_instance([surface, string, module])[
                        'YIELD'][self.topology]['irrad'][hoy]
                    module_mpp_power = np.round(operating_vmp * operating_imp, 3)
                    if module_input_energy == 0:
                        module_eff = 0
                    else:
                        module_eff = np.round(
                            module_mpp_power / module_input_energy, 3)

                    # write these results to the module dict
                    module_dict[
                        'YIELD'][self.topology]['imp'].update({hoy: operating_imp})
                    module_dict[
                        'YIELD'][self.topology]['vmp'].update({hoy: operating_vmp})
                    module_dict[
                        'YIELD'][self.topology]['pmp'].update({hoy: module_mpp_power})
                    module_dict[
                        'YIELD'][self.topology]['eff'].update({hoy: module_eff})

    def write_central_inverter(self, surface):

        surface_dict = self.get_dict_instance([surface])

        surface_iv_results = topology_solver.solve_central_inverter_iv(self, surface)

        surface_dict['CURVES']['Isrf'] = surface_iv_results[0]
        surface_dict['CURVES']['Vsrf'] = surface_iv_results[1]
        surface_dict['YIELD'][self.topology]['irrad'] = surface_iv_results[2]

        mpp_results_dict = topology_solver.solve_central_inverter_mpp(self, surface_dict)

        for hoy in self.all_hoy:
            surface_dict[
                'YIELD'][self.topology]['imp'].update({hoy: mpp_results_dict[hoy]['imp']})
            surface_dict[
                'YIELD'][self.topology]['vmp'].update({hoy: mpp_results_dict[hoy]['vmp']})
            surface_dict[
                'YIELD'][self.topology]['pmp'].update({hoy: mpp_results_dict[hoy]['pmp']})
            # surface_dict[
            #     'YIELD'][self.topology]['isc'].update({hoy: mpp_results_dict[hoy]['isc']})
            # surface_dict[
            #     'YIELD'][self.topology]['voc'].update({hoy: mpp_results_dict[hoy]['voc']})
            surface_dict[
                #     'YIELD'][self.topology]['ff'].update({hoy: mpp_results_dict[hoy]['ff']})
                # surface_dict[
                'YIELD'][self.topology]['eff'].update({hoy: mpp_results_dict[hoy]['eff']})

            # get the current at maximum power point to send to the module to calculate the power at MPP on the module
            operating_vmp = np.round(surface_dict['YIELD'][self.topology]['vmp'][hoy], 3)

            # loop through all of the strings in the surface to recalaute power output and effiency
            # using the operating Vmp fomr the central inverter
            for string in self.get_strings(surface):
                string_dict = self.get_dict_instance([surface, string])

                Istr_hoy = string_dict['CURVES']['Istr'][hoy]
                Vstr_hoy = string_dict['CURVES']['Vstr'][hoy]

                # interp the relationship between I and V for the hour o the module using the oeprating I
                operating_imp = np.round(np.interp(operating_vmp, np.flipud(Vstr_hoy),
                                                   np.flipud(Istr_hoy)), 5)
                string_yield_dict = string_dict['YIELD'][self.topology]
                string_input_energy = string_yield_dict['irrad'][hoy]
                string_mpp_power = np.round(operating_vmp * operating_imp, 3)
                if string_input_energy == 0:
                    string_eff = 0
                else:
                    string_eff = np.round(
                        string_mpp_power / string_input_energy, 3)

                # write these results to the module dict
                string_yield_dict['imp'].update({hoy: operating_imp})
                string_yield_dict['vmp'].update({hoy: operating_vmp})
                string_yield_dict['pmp'].update({hoy: string_mpp_power})
                string_yield_dict['eff'].update({hoy: string_eff})

                string_yield_dict.pop('ff', None)
                string_yield_dict.pop('isc', None)
                string_yield_dict.pop('voc', None)

                # loop through all modules in the string to calculate the power of the module using imp
                # then write this and then calculat ethe efficiency using the svaed irrad
                for module in self.get_modules(surface, string):
                    module_dict = self.get_dict_instance([surface, string, module])
                    Imod_hoy = module_dict['CURVES']['Imod'][hoy]
                    Vmod_hoy = module_dict['CURVES']['Vmod'][hoy]

                    # interp the relationship between I and V for the hour on the module using the operating V from
                    # the central inverter
                    module_vmp = np.round(np.interp(operating_imp, np.flipud(Vmod_hoy),
                                                    np.flipud(Imod_hoy)), 5)

                    module_yield_dict = module_dict['YIELD'][self.topology]
                    module_input_energy = module_yield_dict['irrad'][hoy]
                    module_mpp_power = np.round(module_vmp * operating_imp, 3)
                    if module_input_energy == 0:
                        module_eff = 0
                    else:
                        module_eff = np.round(
                            module_mpp_power / module_input_energy, 3)

                    # write these results to the module dict
                    module_yield_dict['imp'].update({hoy: operating_imp})
                    module_yield_dict['vmp'].update({hoy: module_vmp})
                    module_yield_dict['pmp'].update({hoy: module_mpp_power})
                    module_yield_dict['eff'].update({hoy: module_eff})

                    module_yield_dict.pop('ff', None)
                    module_yield_dict.pop('isc', None)
                    module_yield_dict.pop('voc', None)

    def write_up_string_results(self, surface, string_key):

        string_dict = self.get_dict_instance([surface])['Strings'][string_key]['YIELD'][self.topology]
        # if self.topology == 'micro_inverter':
        #     string_dict = copy.deepcopy(self.get_dict_instance([surface, string])['YIELD']['string_inverter'])

        sub_dict = self.get_dict_instance([surface, string_key])['MODULES']

        for key in string_dict.keys():
            key_result = general.gather_sublevel_results(self,
                                                         sub_dict,
                                                         self.get_modules(surface, string_key),
                                                         key)
            string_dict[key].update(key_result)

        # recalculate efficiency
        for hoy_n, hoy in enumerate(self.all_hoy):
            string_power = string_dict['pmp'][hoy]
            string_irrad = string_dict['irrad'][hoy]
            if string_irrad == 0:
                string_efficiency = 0
            else:
                string_efficiency = np.round(string_power / string_irrad, 3)
            string_dict['eff'][hoy] = string_efficiency

    def write_up_surface_results(self, surface):
        if self.topology == 'central_inverter':
            pass
        else:
            # set the surface dict to the results section
            surface_dict = self.get_dict_instance([surface])['YIELD'][self.topology]

            # get the dict at the surface level to write into
            surface_strings_dict = self.get_dict_instance([surface])['STRINGS']

            for key in surface_dict.keys():
                key_result = general.gather_sublevel_results(self,
                                                             surface_strings_dict,
                                                             self.get_strings(surface),
                                                             key)
                surface_dict[key].update(key_result)

            # recalculate efficiency
            for hoy_n, hoy in enumerate(self.all_hoy):
                surface_power = surface_dict['pmp'][hoy]
                surface_irrad = surface_dict['irrad'][hoy]
                if surface_irrad == 0:
                    surface_efficiency = 0
                else:
                    surface_efficiency = np.round(surface_power / surface_irrad, 3)
                surface_dict['eff'][hoy] = surface_efficiency

    def write_up_object_results(self):
        object_dict = self.panelizer_dict[self.object_type]['YIELD'][self.topology]

        for key in object_dict.keys():
            key_result = general.gather_sublevel_results(self,
                                                         self.panelizer_dict[self.object_type]['SURFACES'],
                                                         self.get_surfaces(),
                                                         key)
            object_dict[key].update(key_result)

        # recalculate efficieny
        for hoy_n, hoy in enumerate(self.all_hoy):
            object_power = object_dict['pmp'][hoy]
            object_irrad = object_dict['irrad'][hoy]
            if object_irrad == 0:
                object_efficiency = 0
            else:
                object_efficiency = np.round(object_power / object_irrad, 3)
            object_dict['eff'][hoy] = object_efficiency

    def get_tabular_results(self, search_list, topology, analysis_period=None, rename_cols=True):
        """_summary_

        Args:
            search_list (list): a list comrpising of options '[surface, string, module]' is the max
            topology (_type_): the electrical topology
            analysis_period (_type_): an analysis period to filter to final results df
            rename_cols (bool, optional): will expand column names and include units. Defaults to True.

        Returns:
            _type_: dataframe
        """

        result_dict = self.get_dict_instance(search_list)['Yield'][topology]

        result_series_l = []

        for key in result_dict.keys():

            result_series = pd.Series(result_dict[key], dtype='float')
            hoy_index = pd.Series(self.all_hoy) #pd.Series(np.arange(0, 8760, 1), name='HOY')

            result_df = pd.concat([hoy_index, result_series], axis=1)
            result = result_df[0].rename(key)

            if len(result) == 0:
                pass
            else:
                result_series_l.append(pd.Series(result).rename(key))

        results_df = pd.concat(result_series_l, axis=1).sort_index()

        idx_start = temporal.hoy_to_date(results_df.index[0])
        idx_end = temporal.hoy_to_date(results_df.index[-1])

        results_df.set_index(temporal.create_datetime(start=str(idx_start), end=str(idx_end)),
                             inplace=True)

        rename_dict = {"imp": "Current at MPPT (amperes)",
                       "vmp": "Voltage at MPPT (volt)",
                       "pmp": "Power at MPPT (W)",
                       "isc": "Short Circuit current (amperes)",
                       "voc": "Open Circuit voltage (volt)",
                       "ff": "Fill Factor (unitless)",
                       "irrad": "Irradiance (W)",
                       "eff": "Efficiency (%)"}
        if rename_cols == True:
            results_df = results_df.rename(columns=rename_dict)

        if analysis_period == None:
            return results_df
        else:
            if analysis_period == "annual":
                return results_df.sum(axis=0)
            else:
                return results_df.loc[analysis_period]


def solve_object_module_iv(panelizer_object, write_system=False, mp=False, display_print=False):
    timeseries = panelizer_object.all_hoy
    tmy_location = general.tmy_location(panelizer_object.tmy_file)
    dbt = panelizer_object.tmy_dataframe['drybulb_C'].values[timeseries]
    psl = panelizer_object.tmy_dataframe['atmos_Pa'].values[timeseries]

    for surface in panelizer_object.get_surfaces():
        print(f"    ...surface {surface}")

        surface_start = time.time()

        # load radiance data
        rad_surface_key = panelizer_object.get_dict_instance([surface])['DETAILS']['radiance_surface_label']
        total_ill = ipv_irrad.load_irradiance_file(panelizer_object.RADIANCE_DIR, rad_surface_key, "total",
                                                   panelizer_object.management_scenario_name).values[
            timeseries]
        direct_ill = ipv_irrad.load_irradiance_file(panelizer_object.RADIANCE_DIR, rad_surface_key, "direct",
                                                    panelizer_object.management_scenario_name).values[
            timeseries]
        diffuse_ill = np.where(total_ill < direct_ill, direct_ill * 0.01, total_ill - direct_ill)

        grid_pts = ipv_irrad.load_grid_file(panelizer_object.RADIANCE_DIR, rad_surface_key,
                                            panelizer_object.management_scenario_name)
        if display_print == True:
            print("    Completed loading grid and irradiance data.")
        # TODO refactor this workflow to iterate over all string,module combinations to increase MP performance
        for string in panelizer_object.get_strings(surface):
            if display_print == True:
                print("     -------------------")
            if display_print == True:
                print(f"    Starting string {string}")
            string_start = time.time()

            string_dict = panelizer_object.get_dict_instance([surface, string])
            string_details = string_dict['DETAILS']

            modules = panelizer_object.get_modules(surface, string)

            base_parameters = io.get_cec_data(string_details['cec_key'], file_path=panelizer_object.cec_data)
            custom_module_data = pd.read_csv(panelizer_object.module_cell_data, index_col='scenario').loc[
                string_details['module_type']].to_dict()

            module_template = string_details['module_type']
            cell_type = ipv_mm.get_cell_type(module_template[0])
            orientation = ipv_mm.get_orientation(module_template[1])
            map_file = [fp for fp in panelizer_object.map_files if f"{cell_type}_{orientation}" in fp][0]
            default_submodule_map, default_diode_map, default_subcell_map = io.read_map_excel(map_file)

            if mp == False:
                if display_print == True:
                    print(f"        Processing string modules and timeseries using a single core.")
                for module_name in modules:
                    module_start = time.time()
                    module_dict = panelizer_object.get_dict_instance([surface, string, module_name])
                    pv_cells_xyz_arr = np.array(panelizer_object.get_cells_xyz(surface, string, module_name))

                    module_i_dict, module_v_dict, module_g_dict, module_params = compile_system_single_core(module_dict,
                                                                                                            timeseries,
                                                                                                            tmy_location,
                                                                                                            dbt, psl,
                                                                                                            pv_cells_xyz_arr,
                                                                                                            grid_pts,
                                                                                                            direct_ill,
                                                                                                            diffuse_ill,
                                                                                                            base_parameters,
                                                                                                            custom_module_data,
                                                                                                            default_submodule_map,
                                                                                                            default_diode_map,
                                                                                                            default_subcell_map,
                                                                                                            cell_type)

                    module_dict['CURVES']['Imod'] = module_i_dict
                    module_dict['CURVES']['Vmod'] = module_v_dict
                    module_dict['YIELD']["initial_simulation"]['irrad'] = module_g_dict
                    module_dict['PARAMETERS'] = module_params
                    module_end = time.time()
                    if display_print == True:
                        print(
                            f"            Time elapsed for module {module_name}: {round(module_end - module_start, 2)}s")

            else:  # MP is true
                # if len(modules) >= panelizer_object.ncpu:
                if display_print == True:
                    print(f"        Processing string {string} across the module list of length {len(modules)}")

                module_start = time.time()
                # compile_mp.main_v3(panelizer_object, surface, string, tmy_location, dbt, psl, grid_pts, direct_ill, diffuse_ill)
                compile_mp.main_v2(panelizer_object, surface, string, dbt, grid_pts, direct_ill, diffuse_ill)
                module_end = time.time()
                if display_print == True:
                    print(f"            Time elapsed for all modules: {round(module_end - module_start, 2)}s")

                # simulations_mp.compile_system_mp_wrapper_module_loop(panelizer_object, surface, string, tmy_location, dbt, psl,
                #                                          grid_pts, direct_ill, diffuse_ill)
                # print(module_dict['PARAMETERS']['n_cols'])

                # else:
                #     if display_print==True:
                #         print(f"        Multiprocessing for string {string}, the timeseries within the module list of length {len(modules)}")
                #     # loop through individual modules and run the MP by chunking the timeseries
                #
                #     for module_name in modules:
                #         module_dict = panelizer_object.get_dict_instance([surface, string, module_name])
                #         pv_cells_xyz_arr = np.array(panelizer_object.get_cells_xyz(surface, string, module_name))
                #
                #         module_start = time.time()
                #         mp_results = single_module_mp.main(panelizer_object,
                #                                            string,
                #                                            module_dict,
                #                                            pv_cells_xyz_arr,
                #                                            tmy_location, dbt, psl,
                #                                            grid_pts,
                #                                            direct_ill, diffuse_ill,
                #                                            base_parameters,
                #                                            custom_module_data,
                #                                            default_submodule_map,
                #                                            default_diode_map,
                #                                            default_subcell_map,
                #                                            cell_type)
                #         module_end = time.time()
                #         if display_print==True:
                #             print(f"            Time elapsed for module {module_name}: {round(module_end - module_start, 2)}s")
                #
                #         module_dict['CURVES']['Imod'] = mp_results[0]
                #         module_dict['CURVES']['Vmod'] = mp_results[1]
                #         module_dict['YIELD']["initial_simulation"]['irrad'] = mp_results[2]
                #         module_dict['PARAMETERS'] = mp_results[3][0]

            string_end = time.time()
            if display_print == True:
                print(f"    Time elapsed for string {string}: {round(string_end - string_start, 2)}s")

        surface_end = time.time()
        if display_print == True:
            print(f"Time elapsed for surface {surface}: {round(surface_end - surface_start, 2)}s")

    # panelizer_object.delete_cell_location()

    # if write_system == True:
    #     print("Panelizer has completed writing input data into dict. File will now be saved.")
    #     utils.write_pickle(panelizer_object, panelizer_object.system_file)
    # else:
    #     print("Panelizer has completed writing input data into dict. File will not be saved. "
    #           "Rerun with write_system to True to save or use 'utils.write_pickle(panelizer_ibject, panelizer_object.system_file)'")


def compile_system_single_core(module_dict, timeseries, tmy_location, dbt, psl, pv_cells_xyz_arr, grid_pts, direct_ill,
                               diffuse_ill, base_parameters, custom_module_data, default_submodule_map,
                               default_diode_map, default_subcell_map, cell_type):
    G_eff_ann, C_temp_ann_arr = build_module_features(module_dict, timeseries, tmy_location, dbt, psl, pv_cells_xyz_arr,
                                                      grid_pts, direct_ill, diffuse_ill,
                                                      base_parameters, custom_module_data,
                                                      default_submodule_map, default_diode_map, default_subcell_map,
                                                      cell_type)

    module_i_dict = {}
    module_v_dict = {}
    module_g_dict = {}
    for hoy_n, hoy in enumerate(timeseries):
        Gmod = G_eff_ann[hoy_n]  # panelizer_object.get_cells_irrad_eff(surface, string, module)[hoy]
        Tmod = C_temp_ann_arr[hoy_n]  # panelizer_object.get_cells_temp(surface, string, module)[hoy]
        if np.sum(Gmod < module_dict['PARAMETERS']['minimum_irradiance_cell']) > 0:
            # Gmod_total = np.sum(Gmod.flatten()*module_dict['PARAMETERS']['one_cell_area_m2']) / module_dict['PARAMETERS']['actual_module_area_m2']
            # if Gmod_total < module_dict['PARAMETERS']['minimum_irradiance_module']

            # print(hoy_n, hoy, time_utils.hoy_to_date(hoy), "Zero Array", np.sum(Gmod))
            Imod, Vmod = (np.zeros(303), np.zeros(303))
        else:
            Imod, Vmod = ipv_calc.calculate_module_map_dependent(Gmod,
                                                                 Tmod,
                                                                 module_dict,
                                                                 ivcurve_pnts=500)

        module_i_dict.update({hoy: np.round(Imod, 5)})
        module_v_dict.update({hoy: np.round(Vmod, 5)})

        # Gmod is originally an array of W/m2 for each cell. Need to convert this array to W by multiply by cell area
        # then take the sum of irradiance for all the cells
        if module_dict['PARAMETERS']['N_subcells'] > 1:
            if module_dict['PARAMETERS']['orientation'] == 'portrait':
                Gmod = np.mean(Gmod, axis=0)
            else:
                Gmod = np.mean(Gmod, axis=1)
        else:
            pass
        Gmod = Gmod * module_dict['PARAMETERS']['one_cell_area_m2']
        module_g_dict.update({hoy: np.round(np.sum(Gmod), 1)})

    return module_i_dict, module_v_dict, module_g_dict, module_dict['PARAMETERS']


def compile_system_multi_core(module_dict_chunk, module_name_chunk, timeseries, tmy_location, dbt, psl,
                              pv_cells_xyz_arr_chunk, grid_pts, direct_ill, diffuse_ill, base_parameters,
                              custom_module_data,
                              default_submodule_map, default_diode_map, default_subcell_map, cell_type):
    module_results = {}

    for n, module_dict in enumerate(module_dict_chunk):
        module_name = module_name_chunk[n]
        pv_cells_xyz_arr = pv_cells_xyz_arr_chunk[n]
        Imod, Vmod, Gmod, module_parameters = compile_system_single_core(module_dict, timeseries,
                                                                         tmy_location, dbt, psl,
                                                                         pv_cells_xyz_arr, grid_pts,
                                                                         direct_ill, diffuse_ill,
                                                                         base_parameters,
                                                                         custom_module_data,
                                                                         default_submodule_map,
                                                                         default_diode_map,
                                                                         default_subcell_map,
                                                                         cell_type)

        module_results.update({module_name: [Imod, Vmod, Gmod, module_parameters]})

    return module_results


def compile_system_multi_core_v2(module_dict_chunk, module_name_chunk, timeseries, tmy_location, dbt, psl,
                                 pv_cells_xyz_arr_chunk, grid_pts, direct_ill_tup, diffuse_ill_tup, base_parameters,
                                 custom_module_data,
                                 default_submodule_map, default_diode_map, default_subcell_map, cell_type):
    direct_ill = multi.unpack_shared_tuple(direct_ill_tup)
    diffuse_ill = multi.unpack_shared_tuple(diffuse_ill_tup)

    module_results = {}

    for n, module_dict in enumerate(module_dict_chunk):
        module_name = module_name_chunk[n]
        pv_cells_xyz_arr = pv_cells_xyz_arr_chunk[n]
        Imod, Vmod, Gmod, module_parameters = compile_system_single_core(module_dict, timeseries,
                                                                         tmy_location, dbt, psl,
                                                                         pv_cells_xyz_arr, grid_pts,
                                                                         direct_ill, diffuse_ill,
                                                                         base_parameters,
                                                                         custom_module_data,
                                                                         default_submodule_map,
                                                                         default_diode_map,
                                                                         default_subcell_map,
                                                                         cell_type)

        module_results.update({module_name: [Imod, Vmod, Gmod, module_parameters]})

    return module_results


def build_module_features(module_dict, timeseries, tmy_location, dbt, psl, pv_cells_xyz_arr, grid_pts, direct_ill,
                          diffuse_ill, base_parameters, custom_module_data, default_submodule_map, default_diode_map,
                          default_subcell_map, cell_type):
    # build parameters for module
    # module_dict = panelizer_object.get_dict_instance([surface, string, module])
    if len(pv_cells_xyz_arr.shape) > 2:
        pv_cells_xyz_arr = pv_cells_xyz_arr[0]

    device_parameters = workbench.device.devices.build_parameter_dict(module_dict, custom_module_data, base_parameters)

    # get direct and diffuse irradiance
    # pv_cells_xyz_arr = np.array(panelizer_object.get_cells_xyz(surface, string, module))

    if type(grid_pts) == np.ndarray:
        sensor_pts_xyz_arr = grid_pts[:, 0:3]
    else:
        sensor_pts_xyz_arr = grid_pts[['X', 'Y', 'Z']].values

    # print(pv_cells_xyz_arr.shape,sensor_pts_xyz_arr.shape,direct_ill.T.shape)
    G_dir_ann = ipv_irrad.collect_raw_irradiance(pv_cells_xyz_arr,
                                                 sensor_pts_xyz_arr,
                                                 direct_ill)  # .values)
    G_diff_ann = ipv_irrad.collect_raw_irradiance(pv_cells_xyz_arr,
                                                  sensor_pts_xyz_arr,
                                                  diffuse_ill)  # .values)
    # calculate effective irradiance
    # tmy_location = utils.tmy_location(panelizer_object.tmy_file)
    # dbt = panelizer_object.tmy_dataframe['drybulb_C'].values
    # psl = panelizer_object.tmy_dataframe['atmos_Pa'].values

    module_normal = tuple(module_dict['CELLSNORMALS'][0])
    front_cover = module_dict['LAYERS']['front_film']
    # print("Direcr shape", G_dir_ann.shape)
    # print("Diffuse shape", G_diff_ann.shape)
    # calcualte the effectivate irradiance for the year
    G_eff_ann = ipv_irrad.calculate_effective_irradiance_timeseries(G_dir_ann,
                                                                    G_diff_ann,
                                                                    module_normal,
                                                                    timeseries,
                                                                    tmy_location,
                                                                    psl,
                                                                    dbt,
                                                                    front_cover)
    # print(len(psl))
    # print(len(dbt))
    # print(G_dir_ann.shape)
    # print(G_diff_ann.shape)
    # print(G_eff_ann.shape)
    # restructure the G_eff arrays following the general shape of the template (top right first)

    # nrows will always be 1 in cdte portrait now due to the changes in grasshopper. this may cause problem
    ncols = device_parameters['n_cols']
    nrows = device_parameters['n_rows']

    hoy_arrs = []

    for hoy_n in np.arange(0, len(timeseries)):
        G_eff_hoy_arr = np.round(np.fliplr(G_eff_ann[hoy_n].reshape(-1, nrows)).T, 2)
        hoy_arrs.append(G_eff_hoy_arr)

    G_eff_ann = np.array(hoy_arrs)

    # calculate cell temperature
    C_temp_ann_arr = ipv_calc.calculate_cell_temperature(G_eff_ann,
                                                         dbt[:, None, None],
                                                         method="ross")

    # set into dict
    # module_dict['CELLSTEMP'] = C_temp_ann_arr
    # module_dict['CELLSIRRADEFF'] = G_eff_ann

    # set new _maps
    # module_template = module_dict['DETAILS']['module_type']
    # cell_type = ipv_mm.get_cell_type(module_template[0])
    # orientation = ipv_mm.get_orientation(module_template[1])
    # map_file = [fp for fp in map_files if f"{cell_type}_{orientation}" in fp][0]
    # default_submodule_map, default_diode_map, default_subcell_map = utils.read_map_excel(map_file)

    if ipv_mm.detect_nonstandard_module(module_dict) == 'standard':
        module_dict['MAPS']['SUBMODULES'] = default_submodule_map
        module_dict['MAPS']['DIODES'] = default_diode_map
        module_dict['MAPS']['SUBCELLS'] = default_subcell_map
    else:
        remap_results = ipv_mm.remap_module_maps(cell_type,
                                                 device_parameters,
                                                 default_diode_map,
                                                 default_subcell_map)
        module_dict['MAPS']['SUBMODULES'] = remap_results[0]
        module_dict['MAPS']['DIODES'] = remap_results[1]
        module_dict['MAPS']['SUBCELLS'] = remap_results[2]
        device_parameters['N_s'] = remap_results[3]
        device_parameters['N_p'] = remap_results[4]
        device_parameters['N_diodes'] = remap_results[4]
        device_parameters['N_subcells'] = remap_results[6]

    for k, v in device_parameters.items():
        module_dict['PARAMETERS'].update({k: v})

    return G_eff_ann, C_temp_ann_arr


def compile_system_multi_core_v2(module_dict_chunk, module_name_chunk, timeseries, G_eff_ann_chunk, dbt,
                                 base_parameters, custom_module_data, default_submodule_map, default_diode_map,
                                 default_subcell_map, cell_type):
    module_results = {}

    for n, module_dict in enumerate(module_dict_chunk):
        module_name = module_name_chunk[n]
        G_eff_ann = G_eff_ann_chunk[n]
        Imod, Vmod, Gmod, module_parameters = compile_system_single_core_v2(module_dict, timeseries,
                                                                            dbt, G_eff_ann,
                                                                            base_parameters,
                                                                            custom_module_data,
                                                                            default_submodule_map,
                                                                            default_diode_map,
                                                                            default_subcell_map,
                                                                            cell_type)

        module_results.update({module_name: [Imod, Vmod, Gmod, module_parameters]})

    return module_results


def compile_system_single_core_v2(module_dict, timeseries, dbt, G_eff_ann,
                                  base_parameters, custom_module_data, default_submodule_map,
                                  default_diode_map, default_subcell_map, cell_type):
    G_eff_ann, C_temp_ann_arr = build_module_features_v2(module_dict, timeseries, dbt, G_eff_ann,
                                                         base_parameters, custom_module_data,
                                                         default_submodule_map, default_diode_map, default_subcell_map,
                                                         cell_type)

    module_i_dict = {}
    module_v_dict = {}
    module_g_dict = {}
    for hoy_n, hoy in enumerate(timeseries):
        Gmod = G_eff_ann[hoy_n]  # panelizer_object.get_cells_irrad_eff(surface, string, module)[hoy]
        Tmod = C_temp_ann_arr[hoy_n]  # panelizer_object.get_cells_temp(surface, string, module)[hoy]
        if np.sum(Gmod < module_dict['PARAMETERS']['minimum_irradiance_cell']) > 0:
            # Gmod_total = np.sum(Gmod.flatten()*module_dict['PARAMETERS']['one_cell_area_m2']) / module_dict['PARAMETERS']['actual_module_area_m2']
            # if Gmod_total < module_dict['PARAMETERS']['minimum_irradiance_module']

            # print(hoy_n, hoy, time_utils.hoy_to_date(hoy), "Zero Array", np.sum(Gmod))
            Imod, Vmod = (np.zeros(303), np.zeros(303))
        else:
            Imod, Vmod = ipv_calc.calculate_module_map_dependent(Gmod,
                                                                 Tmod,
                                                                 module_dict,
                                                                 ivcurve_pnts=500)

        module_i_dict.update({hoy: np.round(Imod, 5)})
        module_v_dict.update({hoy: np.round(Vmod, 5)})

        # Gmod is originally an array of W/m2 for each cell. Need to convert this array to W by multiply by cell area
        # then take the sum of irradiance for all the cells
        if module_dict['PARAMETERS']['N_subcells'] > 1:
            if module_dict['PARAMETERS']['orientation'] == 'portrait':
                Gmod = np.mean(Gmod, axis=0)
            else:
                Gmod = np.mean(Gmod, axis=1)
        else:
            pass
        Gmod = Gmod * module_dict['PARAMETERS']['one_cell_area_m2']
        module_g_dict.update({hoy: np.round(np.sum(Gmod), 1)})

    return module_i_dict, module_v_dict, module_g_dict, module_dict['PARAMETERS']


def build_module_features_v2(module_dict, timeseries, dbt, G_eff_ann,
                             base_parameters, custom_module_data, default_submodule_map, default_diode_map,
                             default_subcell_map, cell_type):
    device_parameters = workbench.device.devices.build_parameter_dict(module_dict, custom_module_data, base_parameters)

    # nrows will always be 1 in cdte portrait now due to the changes in grasshopper. this may cause problem
    ncols = device_parameters['n_cols']
    nrows = device_parameters['n_rows']

    hoy_arrs = []
    for hoy_n in np.arange(0, len(timeseries)):
        G_eff_hoy_arr = np.round(np.fliplr(G_eff_ann[hoy_n].reshape(-1, nrows)).T, 2)
        hoy_arrs.append(G_eff_hoy_arr)

    G_eff_ann = np.array(hoy_arrs)

    # calculate cell temperature
    C_temp_ann_arr = ipv_calc.calculate_cell_temperature(G_eff_ann,
                                                         dbt[:, None, None],
                                                         method="ross")

    if ipv_mm.detect_nonstandard_module(module_dict) == 'standard':
        module_dict['MAPS']['SUBMODULES'] = default_submodule_map
        module_dict['MAPS']['DIODES'] = default_diode_map
        module_dict['MAPS']['SUBCELLS'] = default_subcell_map
    else:
        remap_results = ipv_mm.remap_module_maps(cell_type,
                                                 device_parameters,
                                                 default_diode_map,
                                                 default_subcell_map)
        module_dict['MAPS']['SUBMODULES'] = remap_results[0]
        module_dict['MAPS']['DIODES'] = remap_results[1]
        module_dict['MAPS']['SUBCELLS'] = remap_results[2]
        device_parameters['N_s'] = remap_results[3]
        device_parameters['N_p'] = remap_results[4]
        device_parameters['N_diodes'] = remap_results[4]
        device_parameters['N_subcells'] = remap_results[6]

    for k, v in device_parameters.items():
        module_dict['PARAMETERS'].update({k: v})

    return G_eff_ann, C_temp_ann_arr
