from ipv_workbench.utilities import utils, circuits, time_utils
from ipv_workbench.simulator import simulations
from ipv_workbench.simulator import simulations_mp
import os
import numpy as np
import pandas as pd
import multiprocessing as mp
import glob

class PanelizedObject:
    def __init__(self, panelizer_input, project_folder):
        if type(panelizer_input) is dict:
            self.input_type = 'dict'
            self.panelizer_file = None
            self.panelizer_dict = panelizer_input
        elif type(panelizer_input) is str:
            if os.path.exists(panelizer_input):
                self.input_type = 'file'
                self.panelizer_file = panelizer_input
                self.panelizer_dict = utils.read_pickle(panelizer_input)
            else:
                print("Input detected as string but file path does not exist.")

        self.object_type = list(self.panelizer_dict.keys())[0]
        self.object_surfaces = list(self.panelizer_dict[self.object_type]['SURFACES'].keys())
        if self.object_type=='BUILDING':
            self.object_name = f"B{self.panelizer_dict[self.object_type]['NAME']}"
        else:
            self.object_name = f"O{self.panelizer_dict[self.object_type]['NAME']}"
        self.cell = None
        self.module = None
        self.topology = None
        self.tmy_dataframe = None
        self.analyis_period = (None, None, None)  # start, end, increment
        self.all_hoy = None
        self.ncpu = mp.cpu_count() - 2
        self.multiprocess = True
        self.simulation_suite = False
        self.simulation_suite_topologies = ['micro_inverter', 'string_inverter', 'central_inverter']
        # self.correct_maps()
        self.PROJECT_DIR = project_folder
        self.PANELIZER_DIR = os.path.join(self.PROJECT_DIR,"panelizer",self.object_name)
        self.RESOURCES_DIR = os.path.join(self.PROJECT_DIR,"resources")
        self.RADIANCE_DIR = os.path.join(self.PROJECT_DIR,"radiance_models",self.object_name)
        self.RESULTS_DIR = os.path.join(self.PROJECT_DIR,"results",self.object_name)

        self.tmy_file = glob.glob(os.path.join(self.RESOURCES_DIR,"tmy", "*.epw"))[0]
        self.map_files = glob.glob(os.path.join(self.RESOURCES_DIR, "map_files", "*.xls*"))
        self.tmy_dataframe = utils.tmy_to_dataframe(self.tmy_file)

        self.CEC_DATA = os.path.join(self.RESOURCES_DIR, "cec_database_local.csv")
        self.MODULE_CELL_DATA = os.path.join(self.RESOURCES_DIR, "cell_module_datasheet.csv")

    def correct_maps(self):
        if self.input_type == 'file':
            for surface in self.get_surfaces():
                strings = self.get_strings(surface)
                for string in strings:
                    modules = self.get_modules(surface, string)
                    for module in modules:
                        mod_dict = self.get_dict_instance([surface,string,module])
                        mod_dict['MAPS']['DIODES'] = utils.flip_maps(mod_dict['MAPS']['DIODES'])
                        mod_dict['MAPS']['SUBMODULES'] = utils.flip_maps(mod_dict['MAPS']['SUBMODULES'])
                        mod_dict['MAPS']['SUBCELLS'] = utils.flip_maps(mod_dict['MAPS']['SUBCELLS'])

    def set_analysis_period(self, start, end, increment):
        self.analyis_period = (start, end, increment)
        self.all_hoy = np.arange(self.analyis_period[0], self.analyis_period[1], self.analyis_period[2])

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
        return self.cells_xyz

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
                                            breakdown_voltage=self.cell.cell_params['breakdown_voltage'],
                                            diode_threshold=self.cell.cell_params['diode_threshold'],
                                            bypass=False)
                diode_i.append(i)
                diode_v.append(v)

            # calc series with bypass diodes
            diode_curves = np.array([diode_i, diode_v])
            i, v = circuits.calc_series(diode_curves,
                                        breakdown_voltage=self.cell.cell_params['breakdown_voltage'],
                                        diode_threshold=self.cell.cell_params['diode_threshold'],
                                        bypass=True)
            submodule_i.append(i)
            submodule_v.append(v)
        submodule_curves = np.array([submodule_i, submodule_v])
        Imod, Vmod = circuits.calc_parallel(submodule_curves)
        return Imod, Vmod

    def simulate_system_mp(self, surface):
        if self.topology == 'central_inverter':
            simulations_mp.simulation_central_inverter(
                self, surface)
        elif self.topology == 'string_inverter':
            simulations_mp.simulation_string_inverter(
                self, surface)
        elif self.topology == 'micro_inverter':
            simulations_mp.simulation_micro_inverter(
                self, surface)

    def simulate_system(self, surface, hoy):
        if self.topology == 'central_inverter':
            simulations.simulation_central_inverter(
                self, surface, hoy)
        elif self.topology == 'string_inverter':
            simulations.simulation_string_inverter(
                self, surface, hoy)
        elif self.topology == 'micro_inverter':
            simulations.simulation_micro_inverter(
                self, surface, hoy)

    def write_first_level_results(self, surface, hoy):
        if self.topology == 'string_inverter':
            self.write_string_inverter(surface, hoy)
        elif self.topology == 'central_inverter':
            self.write_central_inverter(surface, hoy)
        elif self.topology == 'micro_inverter':
            self.write_micro_inverter(surface, hoy)

    def write_micro_inverter(self, surface, hoy):

        for string in self.get_strings(surface):

            for module in self.get_modules(surface, string):

                Imod_hoy = self.get_dict_instance(
                    [surface, string, module])['CURVES'][self.topology]['Imod'][hoy]
                Vmod_hoy = self.get_dict_instance(
                    [surface, string, module])['CURVES'][self.topology]['Vmod'][hoy]

                simulation_results_string = simulations.calcMPP_IscVocFF(Imod_hoy,
                                                                         Vmod_hoy)
                temp_results_dict = utils.generate_empty_results_dict(
                    target='STRING')
                temp_results_dict.update(simulation_results_string)

                Gmod_hoy = self.get_dict_instance([surface, string, module])['YIELD'][self.topology]['irrad'][hoy]
                temp_results_dict.update({'irrad': Gmod_hoy})

                module_input_energy = temp_results_dict['irrad']
                if module_input_energy == 0:
                    module_eff = 0
                else:
                    module_eff = np.round(temp_results_dict['pmp'] / module_input_energy, 3)

                # write these results to the module dict
                self.get_dict_instance([surface, string, module])[
                    'YIELD'][self.topology]['imp'].update({hoy: temp_results_dict['imp']})
                self.get_dict_instance([surface, string, module])[
                    'YIELD'][self.topology]['vmp'].update({hoy: temp_results_dict['vmp']})
                self.get_dict_instance([surface, string, module])[
                    'YIELD'][self.topology]['pmp'].update({hoy: temp_results_dict['pmp']})
                self.get_dict_instance([surface, string, module])[
                    'YIELD'][self.topology]['isc'].update({hoy: temp_results_dict['isc']})
                self.get_dict_instance([surface, string, module])[
                    'YIELD'][self.topology]['voc'].update({hoy: temp_results_dict['voc']})
                self.get_dict_instance([surface, string, module])[
                    'YIELD'][self.topology]['ff'].update({hoy: temp_results_dict['ff']})
                self.get_dict_instance([surface, string, module])[
                    'YIELD'][self.topology]['eff'].update({hoy: module_eff})

    def write_string_inverter(self, surface, hoy):

        # TODO the initial write can be done during the simulation

        # get the strings
        strings = self.get_strings(surface)

        # break out the results dict into individual lists
        # i_list = results_dict[hoy][0]
        # v_list = results_dict[hoy][1]
        # g_list = results_dict[hoy][2]
        i_list = [self.get_dict_instance([surface, string])['CURVES'][self.topology]['Istr'][hoy] for string in
                  self.get_strings(surface)]
        v_list = [self.get_dict_instance([surface, string])['CURVES'][self.topology]['Vstr'][hoy] for string in
                  self.get_strings(surface)]
        g_list = [self.get_dict_instance([surface, string])['YIELD'][self.topology]['irrad'][hoy] for string in
                  self.get_strings(surface)]

        # this loops through each "string" in the results
        for n in range(0, len(i_list)):

            # get the result keys from the string level
            string_result_keys = list(self.get_dict_instance([surface, strings[n]])[
                                          'YIELD'][self.topology].keys())

            # calculate mpp and related ->  [Imp, Vmp, Pmp, Isc, Voc, FF]
            # store in a temp directory
            simulation_results_string = simulations.calcMPP_IscVocFF(np.array(i_list[n]),
                                                                     np.array(v_list[n]))
            temp_results_dict = utils.generate_empty_results_dict(
                target='STRING')
            temp_results_dict.update(simulation_results_string)
            temp_results_dict.update({'irrad': g_list[n]})

            # store everything in a dict
            if g_list[n] == 0:
                temp_results_dict.update({'eff': 0})
            else:
                efficiency = temp_results_dict['pmp'] / temp_results_dict['irrad']
                temp_results_dict.update({'eff': efficiency})

            # iterate over the result keys and write then to the panelizer object
            for key in string_result_keys:
                # write to the panelizer object at the string level
                self.get_dict_instance([surface, strings[n]])[
                    'YIELD'][self.topology][key].update({hoy: np.round(temp_results_dict[key], 3)})

            # get the current at maximum power point to send to the module to calculate the power at MPP on the module
            operating_imp = np.round(temp_results_dict['imp'], 3)

            # loop through all modules in the string to calculate the power of the module using imp
            # then write this and then calculat ethe efficiency using the svaed irrad
            for module in self.get_modules(surface, strings[n]):

                Imod_hoy = self.get_dict_instance(
                    [surface, strings[n], module])['CURVES'][self.topology]['Imod'][hoy]
                Vmod_hoy = self.get_dict_instance(
                    [surface, strings[n], module])['CURVES'][self.topology]['Vmod'][hoy]

                # interp the relationship between I and V for the hour o the module using the oeprating I
                operating_vmp = np.round(np.interp(operating_imp, np.flipud(Imod_hoy),
                                                   np.flipud(Vmod_hoy)), 5)
                module_input_energy = self.get_dict_instance([surface, strings[n], module])[
                    'YIELD'][self.topology]['irrad'][hoy]
                module_mpp_power = np.round(operating_vmp * operating_imp, 3)
                if module_input_energy == 0:
                    module_eff = 0
                else:
                    module_eff = np.round(
                        module_mpp_power / module_input_energy, 3)

                # write these results to the module dict
                self.get_dict_instance([surface, strings[n], module])[
                    'YIELD'][self.topology]['imp'].update({hoy: operating_imp})
                self.get_dict_instance([surface, strings[n], module])[
                    'YIELD'][self.topology]['vmp'].update({hoy: operating_vmp})
                self.get_dict_instance([surface, strings[n], module])[
                    'YIELD'][self.topology]['pmp'].update({hoy: module_mpp_power})
                self.get_dict_instance([surface, strings[n], module])[
                    'YIELD'][self.topology]['eff'].update({hoy: module_eff})

    def write_central_inverter(self, surface, hoy):
        # po.get_dict_instance([surface])['RESULTSDICT']
        # TODO the initial write can be done during the simulation

        # get the strings
        strings = self.get_strings(surface)

        # break out the results dict into individual lists
        i_list = self.get_dict_instance([surface])['CURVES'][self.topology]['Isrf'][hoy]  # results_dict[hoy][0]
        v_list = self.get_dict_instance([surface])['CURVES'][self.topology]['Vsrf'][hoy]  # results_dict[hoy][1]
        g_list = self.get_dict_instance([surface])['YIELD'][self.topology]['irrad'][hoy]

        # this loop isn't necssary because central inverter yields one IV per surface
        # future versions though might require it if the surface is split using a multi-MPPT approach
        for n in range(0, len(i_list)):

            # get the result keys
            ## TODO when concatenating the writer this is a filter option based on topology
            surface_result_keys = list(self.get_dict_instance([surface])[
                                           'YIELD'][self.topology].keys())

            # calculate mpp and related ->  [Imp, Vmp, Pmp, Isc, Voc, FF]
            # store in a temp directory
            simulation_results = simulations.calcMPP_IscVocFF(np.array(i_list[n]),
                                                              np.array(v_list[n]))
            ## TODO when concatenating the writer this is a filter option based on topology
            temp_results_dict = utils.generate_empty_results_dict(
                target='SURFACE')
            temp_results_dict.update(simulation_results)
            temp_results_dict.update({'irrad': g_list[n]})

            # store everything in a dict
            if g_list[n] == 0:
                temp_results_dict.update({'eff': 0})
            else:
                efficiency = temp_results_dict['pmp'] / temp_results_dict['irrad']
                temp_results_dict.update({'eff': np.round(efficiency, 3)})

            # iterate over the result keys and write then to the panelizer object
            for key in surface_result_keys:
                # write to the panelizer object at the string level
                self.get_dict_instance([surface])[
                    'YIELD'][self.topology][key].update({hoy: np.round(temp_results_dict[key], 3)})

            # get the current at maximum power point to send to the module to calculate the power at MPP on the module
            operating_vmp = np.round(temp_results_dict['vmp'], 5)

            # loop through all of the strings in the surface to recalaute power output and effiency
            # using the operating Vmp fomr the central inverter
            for string in strings:
                Istr_hoy = self.get_dict_instance(
                    [surface, string])['CURVES'][self.topology]['Istr'][hoy]
                Vstr_hoy = self.get_dict_instance(
                    [surface, string])['CURVES'][self.topology]['Vstr'][hoy]

                # interp the relationship between I and V for the hour o the module using the oeprating I
                operating_imp = np.round(np.interp(operating_vmp, np.flipud(Vstr_hoy),
                                                   np.flipud(Istr_hoy)), 5)
                string_input_energy = self.get_dict_instance([surface, string])[
                    'YIELD'][self.topology]['irrad'][hoy]
                string_mpp_power = np.round(operating_vmp * operating_imp, 3)
                if string_input_energy == 0:
                    string_eff = 0
                else:
                    string_eff = np.round(
                        string_mpp_power / string_input_energy, 3)

                # write these results to the module dict
                self.get_dict_instance([surface, string])[
                    'YIELD'][self.topology]['imp'].update({hoy: operating_imp})
                self.get_dict_instance([surface, string])[
                    'YIELD'][self.topology]['vmp'].update({hoy: operating_vmp})
                self.get_dict_instance([surface, string])[
                    'YIELD'][self.topology]['pmp'].update({hoy: string_mpp_power})
                self.get_dict_instance([surface, string])[
                    'YIELD'][self.topology]['eff'].update({hoy: string_eff})

                # loop through all modules in the string to calculate the power of the module using imp
                # then write this and then calculat ethe efficiency using the svaed irrad
                for module in self.get_modules(surface, string):
                    Imod_hoy = self.get_dict_instance(
                        [surface, string, module])['CURVES'][self.topology]['Imod'][hoy]
                    Vmod_hoy = self.get_dict_instance(
                        [surface, string, module])['CURVES'][self.topology]['Vmod'][hoy]

                    # interp the relationship between I and V for the hour on the module using the operating V from
                    # the central inverter
                    module_vmp = np.round(np.interp(operating_imp, np.flipud(Vmod_hoy),
                                                    np.flipud(Imod_hoy)), 5)
                    module_input_energy = self.get_dict_instance([surface, string, module])[
                        'YIELD'][self.topology]['irrad'][hoy]
                    module_mpp_power = np.round(module_vmp * operating_imp, 3)
                    if module_input_energy == 0:
                        module_eff = 0
                    else:
                        module_eff = np.round(
                            module_mpp_power / module_input_energy, 3)

                    # write these results to the module dict
                    self.get_dict_instance([surface, string, module])[
                        'YIELD'][self.topology]['imp'].update({hoy: operating_imp})
                    self.get_dict_instance([surface, string, module])[
                        'YIELD'][self.topology]['vmp'].update({hoy: module_vmp})
                    self.get_dict_instance([surface, string, module])[
                        'YIELD'][self.topology]['pmp'].update({hoy: module_mpp_power})
                    self.get_dict_instance([surface, string, module])[
                        'YIELD'][self.topology]['eff'].update({hoy: module_eff})

    def write_up_string_results(self, surface, string):
        string_dict = self.get_dict_instance([surface, string])['YIELD'][self.topology]

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
            result = result_dict[key]
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
            return results_df.loc[analysis_period]
