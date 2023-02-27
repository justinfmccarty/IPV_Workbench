import copy
import time

from ipv_workbench.utilities import utils, circuits, time_utils
from ipv_workbench.solver import calculations as ipv_calc, compile_mp, single_module_mp, topology_solver
from ipv_workbench.solver import simulations
from ipv_workbench.solver import simulations_mp
from ipv_workbench.translators import mapping_irradiance as ipv_irrad
from ipv_workbench.translators import module_mapping as ipv_mm
import os
import numpy as np
import pandas as pd
import multiprocessing as mp
import glob


class PanelizedObject:
    def __init__(self, project_folder, object_name, raw_panelizer):

        self.object_name = object_name
        self.raw_panelizer = raw_panelizer
        self.project_folder = project_folder
        self.project_setup()

        self.cell = None
        self.module = None
        self.topology = None
        self.tmy_dataframe = None
        self.analyis_period = (None, None, None)  # start, end, increment
        self.all_hoy = None
        self.annual_hoy = np.arange(0, 8760, 1)
        self.ncpu = mp.cpu_count() - 1
        self.multiprocess = True
        self.simulation_suite = False
        self.simulation_suite_topologies = ['micro_inverter', 'string_inverter', 'central_inverter']
        self.analysis_location = None
        self.analysis_year = None
        self.hourly_resolution = None

    def project_setup(self):
        self.PROJECT_DIR = self.project_folder

        # object data
        self.OBJECTS_DIR = os.path.join(self.PROJECT_DIR, "objects")
        utils.directory_creator(self.OBJECTS_DIR)
        self.OBJECT_DIR = os.path.join(self.OBJECTS_DIR, self.object_name)
        utils.directory_creator(self.OBJECT_DIR)
        self.PANELIZER_DIR = os.path.join(self.OBJECT_DIR, "panelizer")
        utils.directory_creator(self.PANELIZER_DIR)

        self.set_panelizer_dict()

        self.CUMULATIVE_RESULTS_DIR = os.path.join(self.OBJECTS_DIR, "cumulative_results")
        utils.directory_creator(self.CUMULATIVE_RESULTS_DIR)
        self.CUMULATIVE_RESULTS_DENSE_DIR = os.path.join(self.OBJECTS_DIR, "cumulative_condensed")
        utils.directory_creator(self.CUMULATIVE_RESULTS_DENSE_DIR)

        self.COLD_DIR = os.path.join(self.OBJECT_DIR, "cold_storage")
        utils.directory_creator(self.COLD_DIR)
        self.RESULTS_DIR = os.path.join(self.OBJECT_DIR, "results")
        utils.directory_creator(self.RESULTS_DIR)
        self.ANNUAL_RESULT_DIR = os.path.join(self.RESULTS_DIR, "annual")
        utils.directory_creator(self.ANNUAL_RESULT_DIR)
        self.TIMESERIES_RESULT_DIR = os.path.join(self.RESULTS_DIR, "timeseries")
        utils.directory_creator(self.TIMESERIES_RESULT_DIR)

        # shared data betweens cenarios, objects
        self.SHARED_DIR = os.path.join(self.PROJECT_DIR, "shared")
        utils.directory_creator(self.SHARED_DIR)

        self.RADIANCE_DIRS = os.path.join(self.SHARED_DIR, "radiance_models")
        utils.directory_creator(self.RADIANCE_DIRS)
        self.RADIANCE_DIR = os.path.join(self.RADIANCE_DIRS, self.object_name)
        utils.directory_creator(self.RADIANCE_DIR)

        self.RESOURCES_DIR = os.path.join(self.SHARED_DIR, "resources")
        utils.directory_creator(self.RESOURCES_DIR)
        self.SUN_UP_FILE = os.path.join(self.RESOURCES_DIR, "tmy", "sun-up-hours.txt")
        sun_up, sun_hours = utils.create_sun_mask(self.SUN_UP_FILE)
        self.sunup_array = sun_hours['HOY'].values
        self.sundown_array = sun_up[sun_up['Sunny'] == False]['HOY'].values


        self.MODULE_CELL_DIR = os.path.join(self.RESOURCES_DIR, "cell_module_data")
        utils.directory_creator(self.MODULE_CELL_DIR)
        self.module_cell_data = os.path.join(self.MODULE_CELL_DIR, "cell_module_datasheet.csv")
        self.cec_data = os.path.join(self.MODULE_CELL_DIR, "cec_database_local.csv")
        self.MAPS_DIR = os.path.join(self.MODULE_CELL_DIR, "map_files")
        utils.directory_creator(self.MAPS_DIR)
        self.map_files = glob.glob(os.path.join(self.MAPS_DIR, "*.xls*"))

        self.LOADS_DIR = os.path.join(self.RESOURCES_DIR, "loads")
        utils.directory_creator(self.LOADS_DIR)

    def set_panelizer_dict(self):
        # print("Setting dict from raw panelizer file")
        if type(self.raw_panelizer) is dict:
            self.input_type = 'dict'
            self.panelizer_file = None
            self.panelizer_dict = self.raw_panelizer
        else:
            object_path = os.path.join(self.OBJECT_DIR, "panelizer", self.raw_panelizer)
            # print(object_path)
            if os.path.exists(object_path):
                self.input_type = 'file'
                self.panelizer_file = object_path
                self.file_name = self.panelizer_file.split(os.sep)[-1].split(".")[0]
                self.panelizer_dict = utils.read_pickle(object_path)
            else:
                print(object_path)
                print("Input detected as string but file path does not exist.")

        self.object_type = list(self.panelizer_dict.keys())[0]
        self.object_surfaces = list(self.panelizer_dict[self.object_type]['SURFACES'].keys())

        # print(self.object_type)
        # if self.object_type == 'BUILDING':
        #     self.object_name = f"B{self.panelizer_dict[self.object_type]['NAME']}"
        # else:
        #     self.object_name = f"O{self.panelizer_dict[self.object_type]['NAME']}"

    def set_tmy_data(self):
        self.TMY_DIR = os.path.join(self.RESOURCES_DIR, 'tmy')
        utils.directory_creator(self.TMY_DIR)
        self.wea_file = os.path.join(self.TMY_DIR, f"{self.analysis_location}_{self.analysis_year}.wea")
        self.tmy_file = os.path.join(self.TMY_DIR, f"{self.analysis_location}_{self.analysis_year}.epw")
        self.tmy_dataframe = utils.tmy_to_dataframe(self.tmy_file)

    def correct_maps(self):
        if self.input_type == 'file':
            for surface in self.get_surfaces():
                strings = self.get_strings(surface)
                for string in strings:
                    modules = self.get_modules(surface, string)
                    for module in modules:
                        mod_dict = self.get_dict_instance([surface, string, module])
                        mod_dict['MAPS']['DIODES'] = utils.flip_maps(mod_dict['MAPS']['DIODES'])
                        mod_dict['MAPS']['SUBMODULES'] = utils.flip_maps(mod_dict['MAPS']['SUBMODULES'])
                        mod_dict['MAPS']['SUBCELLS'] = utils.flip_maps(mod_dict['MAPS']['SUBCELLS'])

    def set_analysis_period(self):
        if self.hourly_resolution is None:
            hourly_resolution = 1
        else:
            hourly_resolution = self.hourly_resolution
        self.all_hoy = time_utils.build_analysis_period(self.sunup_array, hourly_resolution)

        return self.all_hoy

    # def reset_results_dict(self, surface):
    #     self.get_dict_instance([surface])['RESULTSDICT'] = {}
    #
    # def get_results_dict(self, surface):
    #     return self.get_dict_instance([surface])['RESULTSDICT']

    def get_surfaces(self):
        self.active_surfaces = list(self.panelizer_dict[self.object_type]['SURFACES'].keys())
        return self.active_surfaces

    def get_strings(self, surface_name):
        self.active_strings = list(self.panelizer_dict[self.object_type]['SURFACES'][surface_name]['STRINGS'].keys())
        return self.active_strings

    def get_modules(self, surface_name, string_name):
        self.active_modules = list(
            self.panelizer_dict[self.object_type]['SURFACES'][surface_name]['STRINGS'][string_name]['MODULES'].keys())
        return self.active_modules

    def get_submodule_map(self, surface_name, string_name, module_name):
        self.active_submodule_map = \
            self.panelizer_dict[self.object_type]['SURFACES'][surface_name]['STRINGS'][string_name]['MODULES'][
                module_name][
                'MAPS']['SUBMODULES']
        self.submodules = np.unique(self.active_submodule_map)
        return self.active_submodule_map

    def get_diode_map(self, surface_name, string_name, module_name):
        self.active_diode_map = \
            self.panelizer_dict[self.object_type]['SURFACES'][surface_name]['STRINGS'][string_name]['MODULES'][
                module_name][
                'MAPS']['DIODES']
        self.diodes = np.unique(self.active_diode_map)
        return self.active_diode_map

    def get_cells_xyz(self, surface_name, string_name, module_name):
        self.cells_xyz = \
            self.panelizer_dict[self.object_type]['SURFACES'][surface_name]['STRINGS'][string_name]['MODULES'][
                module_name][
                'CELLSXYZ']
        return np.array(self.cells_xyz)

    def get_cells_normals(self, surface_name, string_name, module_name):
        self.cells_normals = \
            self.panelizer_dict[self.object_type]['SURFACES'][surface_name]['STRINGS'][string_name]['MODULES'][
                module_name][
                'CELLSNORMALS']
        return self.cells_normals

    def get_cells_irrad(self, surface_name, string_name, module_name):
        self.cells_irrad = \
            self.panelizer_dict[self.object_type]['SURFACES'][surface_name]['STRINGS'][string_name]['MODULES'][
                module_name][
                'CELLSIRRAD']
        return self.cells_irrad

    def get_cells_irrad_eff(self, surface_name, string_name, module_name):
        self.cells_irrad_eff = \
            self.panelizer_dict[self.object_type]['SURFACES'][surface_name]['STRINGS'][string_name]['MODULES'][
                module_name][
                'CELLSIRRADEFF']
        return self.cells_irrad_eff

    def get_cells_temp(self, surface_name, string_name, module_name):
        self.cells_temp = \
            self.panelizer_dict[self.object_type]['SURFACES'][surface_name]['STRINGS'][string_name]['MODULES'][
                module_name][
                'CELLSTEMP']
        return self.cells_temp

    def calc_irrad_eff(self):
        # effective_irradiance.calculate_effective_irradiance()
        self.cells_irrad_eff = None
        return self.cells_irrad_eff

    def get_dict_instance(self, keys):
        if len(keys) == 0:
            return self.panelizer_dict[self.object_type]
        elif len(keys) == 1:
            return self.panelizer_dict[self.object_type]['SURFACES'][keys[0]]
        elif len(keys) == 2:
            return self.panelizer_dict[self.object_type]['SURFACES'][keys[0]]['STRINGS'][keys[1]]
        elif len(keys) == 3:
            return self.panelizer_dict[self.object_type]['SURFACES'][keys[0]]['STRINGS'][keys[1]]['MODULES'][keys[2]]
        elif len(keys) == 4:
            return self.panelizer_dict[self.object_type]['SURFACES'][keys[0]]['STRINGS'][keys[1]]['MODULES'][keys[2]][
                keys[3]]

    def calculate_module_curve(self, irradiance_hoy, temperature_hoy):
        # TODO break apart into constituent pieces
        # TODO add subcell routine

        submodule_i = []
        submodule_v = []

        for submodule_key in self.submodules:
            submodule_mask = self.active_submodule_map == submodule_key
            submodule_diode = self.active_diode_map[submodule_mask]
            submodule_irrad = irradiance_hoy[submodule_mask]
            submodule_temp = temperature_hoy[submodule_mask]
            diode_i = []
            diode_v = []
            for diode_key in self.diodes:
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
            for string in self.get_strings(surface):
                for module in self.get_modules(surface, string):
                    module_dict = self.get_dict_instance([surface, string, module])
                    module_dict.pop('CELLSXYZ', None)
                    module_dict.pop('CELLS_NORMALS', None)

    def write_key_parameters(self):
        object_dict = self.get_dict_instance([])

        object_capacity = []
        object_area = []
        for surface in self.get_surfaces():
            surface_dict = self.get_dict_instance([surface])

            surface_capacity = []
            surface_area = []
            for string_ in self.get_strings(surface):
                string_dict = self.get_dict_instance([surface, string_])

                string_capacity = []
                string_area = []
                for module in self.get_modules(surface, string_):
                    module_dict = self.get_dict_instance([surface, string_, module])
                    module_capacity = module_dict['PARAMETERS']['actual_capacity_Wp']
                    module_area = module_dict['PARAMETERS']['actual_module_area_m2']
                    string_capacity.append(module_capacity)
                    string_area.append(module_area)
                string_dict['DETAILS']['installed_capacity_Wp'] = np.sum(string_capacity)
                string_dict['DETAILS']['installed_area_m2'] = np.sum(string_area)
                surface_capacity.append(np.sum(string_capacity))
                surface_area.append(np.sum(string_area))

            surface_dict['DETAILS']['installed_capacity_Wp'] = np.sum(surface_capacity)
            surface_dict['DETAILS']['installed_area_m2'] = np.sum(surface_area)
            object_capacity.append(np.sum(surface_capacity))
            object_area.append(np.sum(surface_area))

        object_dict['DETAILS']['installed_capacity_Wp'] = np.sum(object_capacity)
        object_dict['DETAILS']['installed_area_m2'] = np.sum(object_area)

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
            self.write_micro_inverter(surface, string)

    def write_micro_inverter(self, surface, string):

        for module in self.get_modules(surface, string):

            module_dict = self.get_dict_instance([surface, string, module])
            mpp_results_dict = topology_solver.solve_micro_inverter_mpp(self, module_dict)

            # write these results to the module dict
            for hoy in self.all_hoy:
                module_dict[
                    'YIELD'][self.topology]['imp'].update({hoy: mpp_results_dict[hoy]['imp']})
                module_dict[
                    'YIELD'][self.topology]['vmp'].update({hoy: mpp_results_dict[hoy]['vmp']})
                module_dict[
                    'YIELD'][self.topology]['pmp'].update({hoy: mpp_results_dict[hoy]['pmp']})
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

    def write_up_string_results(self, surface, string):

        string_dict = self.get_dict_instance([surface, string])['YIELD'][self.topology]
        # if self.topology == 'micro_inverter':
        #     string_dict = copy.deepcopy(self.get_dict_instance([surface, string])['YIELD']['string_inverter'])

        sub_dict = self.get_dict_instance([surface, string])['MODULES']

        for key in string_dict.keys():
            key_result = utils.gather_sublevel_results(self,
                                                       sub_dict,
                                                       self.get_modules(surface, string),
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
                key_result = utils.gather_sublevel_results(self,
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
            key_result = utils.gather_sublevel_results(self,
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
        result_dict = self.get_dict_instance(search_list)['YIELD'][topology]

        result_series_l = []

        for key in result_dict.keys():

            result_series = pd.Series(result_dict[key])
            hoy_index = pd.Series(np.arange(0, 8760, 1), name='HOY')
            result_df = pd.concat([hoy_index, result_series], axis=1)
            result = result_df[0].rename(key)

            if len(result) == 0:
                pass
            else:
                result_series_l.append(pd.Series(result).rename(key))

        results_df = pd.concat(result_series_l, axis=1).sort_index()

        idx_start = time_utils.hoy_to_date(results_df.index[0])
        idx_end = time_utils.hoy_to_date(results_df.index[-1])

        results_df.set_index(time_utils.create_datetime(start=str(idx_start), end=str(idx_end)), inplace=True)

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
    tmy_location = utils.tmy_location(panelizer_object.tmy_file)
    dbt = panelizer_object.tmy_dataframe['drybulb_C'].values[timeseries]
    psl = panelizer_object.tmy_dataframe['atmos_Pa'].values[timeseries]

    for surface in panelizer_object.get_surfaces():
        print(f"    ...surface {surface}")

        surface_start = time.time()

        # load radiance data
        rad_surface_key = panelizer_object.get_dict_instance([surface])['DETAILS']['radiance_surface_label']
        total_ill = ipv_irrad.load_irradiance_file(panelizer_object.RADIANCE_DIR, rad_surface_key, "total").values[
            timeseries]
        direct_ill = ipv_irrad.load_irradiance_file(panelizer_object.RADIANCE_DIR, rad_surface_key, "direct").values[
            timeseries]
        diffuse_ill = np.where(total_ill < direct_ill, direct_ill * 0.01, total_ill - direct_ill)

        grid_pts = ipv_irrad.load_grid_file(panelizer_object.RADIANCE_DIR, rad_surface_key)

        for string in panelizer_object.get_strings(surface):
            if display_print == True:
                print("     -------------------")
            if display_print == True:
                print(f"    Starting string {string}")
            string_start = time.time()

            string_dict = panelizer_object.get_dict_instance([surface, string])
            string_details = string_dict['DETAILS']

            modules = panelizer_object.get_modules(surface, string)

            base_parameters = utils.get_cec_data(string_details['cec_key'], file_path=panelizer_object.cec_data)
            custom_module_data = pd.read_csv(panelizer_object.module_cell_data, index_col='scenario').loc[
                string_details['module_type']].to_dict()

            module_template = string_details['module_type']
            cell_type = ipv_mm.get_cell_type(module_template[0])
            orientation = ipv_mm.get_orientation(module_template[1])
            map_file = [fp for fp in panelizer_object.map_files if f"{cell_type}_{orientation}" in fp][0]
            default_submodule_map, default_diode_map, default_subcell_map = utils.read_map_excel(map_file)

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
                compile_mp.main(panelizer_object, surface, string, tmy_location, dbt, psl, grid_pts, direct_ill,
                                diffuse_ill)
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

    # module_dict['DETAILS'].update({'para_keys':np.array(list(module_dict['PARAMETERS'].keys()))})
    # module_dict['DETAILS'].update({'para_values':np.array(list(module_dict['PARAMETERS'].values()))})

    module_dict['PARAMETERS']['minimum_irradiance_cell'] = 5

    module_i_dict = {}
    module_v_dict = {}
    module_g_dict = {}
    for hoy_n, hoy in enumerate(timeseries):
        Gmod = G_eff_ann[hoy_n]  # panelizer_object.get_cells_irrad_eff(surface, string, module)[hoy]
        Tmod = C_temp_ann_arr[hoy_n]  # panelizer_object.get_cells_temp(surface, string, module)[hoy]
        if np.sum(Gmod < module_dict['PARAMETERS']['minimum_irradiance_cell']) > 0:

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
        Gmod = Gmod * module_dict['PARAMETERS']['one_subcell_area_m2']
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


def build_module_features(module_dict, timeseries, tmy_location, dbt, psl, pv_cells_xyz_arr, grid_pts, direct_ill,
                          diffuse_ill, base_parameters, custom_module_data, default_submodule_map, default_diode_map,
                          default_subcell_map, cell_type):
    # build parameters for module
    # module_dict = panelizer_object.get_dict_instance([surface, string, module])
    if len(pv_cells_xyz_arr.shape) > 2:
        pv_cells_xyz_arr = pv_cells_xyz_arr[0]

    module_details = module_dict['DETAILS']

    for k, v in custom_module_data.items():
        base_parameters[k] = v

    base_parameters['N_subcells'] = int(max(base_parameters['Nsubcell_col'], base_parameters['Nsubcell_row']))
    for k, v in base_parameters.items():
        if type(v) is str:
            try:
                base_parameters[k] = float(v)
            except ValueError:
                pass

    base_parameters['n_cols'] = module_details['n_cols']
    base_parameters['n_rows'] = module_details['n_rows']
    ideal_cell_count = base_parameters['n_cols_ideal'] * base_parameters['n_rows_ideal']
    base_parameters['total_cells'] = base_parameters['n_cols'] * base_parameters['n_rows']
    watts_cell = base_parameters['Wp'] / ideal_cell_count
    base_parameters['actual_capacity_Wp'] = watts_cell * base_parameters['total_cells']

    m2_cell = base_parameters['module_area'] / ideal_cell_count
    base_parameters['actual_module_area_m2'] = m2_cell * base_parameters['total_cells']

    base_parameters['actual_cell_area_m2'] = base_parameters['cell_area']
    base_parameters['Wp_m2_cell'] = base_parameters['Wp'] / base_parameters['actual_cell_area_m2']
    base_parameters['Wp_m2_module'] = base_parameters['Wp'] / base_parameters['actual_module_area_m2']
    base_parameters['one_subcell_area_m2'] = ((base_parameters['cell_width'] * base_parameters['cell_height']) /
                                              base_parameters['N_subcells']) * 0.000001
    base_parameters['one_cell_area_m2'] = (base_parameters['cell_width'] * base_parameters['cell_height']) * 0.000001
    base_parameters['minimum_irradiance_cell'] = 5

    # assign subcell counts if present
    actual_cols = base_parameters['n_cols']
    actual_rows = base_parameters['n_rows']
    ideal_subcell_col = base_parameters['Nsubcell_col']
    ideal_subcell_row = base_parameters['Nsubcell_row']
    if ideal_subcell_col > ideal_subcell_row:
        # print("Subcells detected for columns.")
        base_parameters['Nsubcell_col'] = actual_cols
        base_parameters['N_subcells'] = actual_cols
    elif ideal_subcell_col < ideal_subcell_row:
        # print("Subcells detected for rows.")
        base_parameters['Nsubcell_row'] = actual_rows
        base_parameters['N_subcells'] = actual_rows
    else:
        pass

    base_parameters['cell_area'] = (base_parameters['cell_area'] / (
            base_parameters['n_rows_ideal'] * base_parameters['n_cols_ideal'])) * base_parameters['total_cells']

    # module_dict['PARAMETERS'] = base_parameters.to_dict()
    base_parameters = base_parameters.to_dict()

    # get direct and diffuse irradiance
    # pv_cells_xyz_arr = np.array(panelizer_object.get_cells_xyz(surface, string, module))
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

    evaluated_normal_vector = tuple(module_dict['CELLSNORMALS'][0])
    front_cover = module_dict['LAYERS']['front_film']
    # print("Direcr shape", G_dir_ann.shape)
    # print("Diffuse shape", G_diff_ann.shape)
    # calcualte the effectivate irradiance for the year
    G_eff_ann = ipv_irrad.calculate_effective_irradiance_timeseries(G_dir_ann,
                                                                    G_diff_ann,
                                                                    evaluated_normal_vector,
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
    ncols = base_parameters['n_cols']
    nrows = base_parameters['n_rows']

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
                                                 base_parameters,
                                                 default_diode_map,
                                                 default_subcell_map)
        module_dict['MAPS']['SUBMODULES'] = remap_results[0]
        module_dict['MAPS']['DIODES'] = remap_results[1]
        module_dict['MAPS']['SUBCELLS'] = remap_results[2]
        base_parameters['N_s'] = remap_results[3]
        base_parameters['N_p'] = remap_results[4]
        base_parameters['N_diodes'] = remap_results[4]
        base_parameters['N_subcells'] = remap_results[6]

    for k, v in base_parameters.items():
        module_dict['PARAMETERS'].update({k: v})

    return G_eff_ann, C_temp_ann_arr
