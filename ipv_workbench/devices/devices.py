from ipv_workbench.utilities import utils
import os
from ipv_workbench.solver import simulations as sim
from ipv_workbench.visualize import plots as ipv_plots
import numpy as np


class Cell:
    def __init__(self, parameter_dict):
        self.parameters_dict = parameter_dict
        self.assign_parameters()
        self.iv_library = None
        self.iv_library_conditions = None
        self.iv_library_resolution = None
        self.vectorized_retrieve_curve = np.vectorize(self.retrieve_curve, excluded='self')

    def assign_parameters(self):
        # TODO map datatypes to value assignment
        for k, v in self.parameters_dict.items():
            setattr(self, k, v)
        self.cell_area = self.cell_width * self.cell_height

    def assign_iv_library(self, profile_path, write_library=True):
        conditions = utils.create_conditions_map()
        if os.path.exists(profile_path):
            self.iv_library = utils.read_json(profile_path)
            self.iv_library_conditions = conditions
            self.split_library_keys()
        else:
            simulation_results = {}
            for n in range(0, len(conditions)):
                irrad = conditions[n][0]
                temp = conditions[n][1]
                key = f"{irrad},{temp}"
                simulation_results[key] = {"i": [],
                                           "v": []}
                results = sim.solve_iv_curve(self.parameters_dict,
                                             irrad,
                                             temp,
                                             self.curve_resolution)
                results = np.array(results)
                simulation_results[key]['i'] = list(results[0, :])
                simulation_results[key]['v'] = list(results[1, :])
            if write_library:
                utils.write_json(simulation_results, profile_path)

            self.iv_library = simulation_results
            self.iv_library_conditions = conditions
            self.split_library_keys()

    def find_matching_key(self, search_irrad, search_temp):
        # search the preestablished conditions libraries for the closest match
        matching_irrad_idx = self.iv_library_irrad_conditions.searchsorted(search_irrad)
        matching_temp_idx = self.iv_library_temp_conditions.searchsorted(search_temp)

        # if the match is greater than the highest number in the list search sorted adds an extra number ot the index
        # so it needs to be clipped to the length of the conditions library minus 1 because of zero indexing
        matching_irrad_idx = np.clip(matching_irrad_idx, 0, len(self.iv_library_irrad_conditions) - 1)
        matching_temp_idx = np.clip(matching_temp_idx, 0, len(self.iv_library_temp_conditions) - 1)

        # extract frm the library based on the matching idx
        matching_irrad = self.iv_library_irrad_conditions[matching_irrad_idx]
        matching_temp = self.iv_library_temp_conditions[matching_temp_idx]

        # return the formatted key
        return f"{int(matching_irrad)},{matching_temp}"

    def retrieve_curve(self, search_irrad, search_temp):
        iv_key = self.find_matching_key(search_irrad, search_temp)
        return iv_key, self.iv_library[iv_key]['i'], self.iv_library[iv_key]['v']

    def retrieve_curves_multiple_cells(self, irradiance_arr, temperature_arr):
        i_list = []
        v_list = []
        for params in list(zip(irradiance_arr, temperature_arr)):
            k, i, v = self.retrieve_curve(params[0], params[1])
            i_list.append(i)
            v_list.append(v)

        iv_curves = np.array([i_list, v_list])
        return iv_curves

    def split_library_keys(self):
        irrad_conditions = []
        temp_conditions = []
        for k in self.iv_library.keys():
            k_split = k.split(",")
            irrad_conditions.append(k_split[0])
            temp_conditions.append(k_split[1])

        self.iv_library_irrad_conditions = np.sort(np.array(list(set(irrad_conditions)), dtype=float))
        self.iv_library_temp_conditions = np.sort(np.array(list(set(temp_conditions)), dtype=float))

    def cell_plot_iv(self, irrad_list, temp_list, title,
                     bypass=True, mpp=True, colors=None, linewidths=None, linestyles=None,
                     figsize=None, out_file=None):
        if isinstance(irrad_list, list):
            irrad_length = len(irrad_list)
        else:
            irrad_length = 1

        if isinstance(temp_list, list):
            temp_length = len(temp_list)
        else:
            temp_length = 1

        list_length = max(irrad_length, temp_length)

        if not isinstance(irrad_list, list):
            irrad_list = [irrad_list] * list_length

        if not isinstance(temp_list, list):
            temp_list = [temp_list] * list_length

        labels = []
        i_arrs = []
        v_arrs = []

        for irrad, temp in zip(irrad_list, temp_list):
            key, i, v = self.retrieve_curve(irrad, temp)
            labels.append(key)
            i_arrs.append(i)
            v_arrs.append(v)

        if colors is None:
            colors = ['grey'] * len(labels)
        if linestyles is None:
            linestyles = ['solid'] * len(labels)
        if linewidths is None:
            linewidths = [0.75] * len(labels)
        if figsize is None:
            figsize = (5, 3)

        ipv_plots.plot_curves(i_arrs, v_arrs,
                              self.parameters_dict,
                              labels=labels,
                              title=title,
                              linewidth=linewidths,
                              colors=colors,
                              linestyle=linestyles,
                              fs=figsize,
                              bypass=bypass,
                              mpp=mpp,
                              reverse=True,
                              save=out_file
                              )


class Module:
    def __init__(self):
        self.module_description = None

        self.front_film = None
        self.front_cover = None
        self.encapsulant = None
        self.cell = None
        self.rear_cover = None
        self.frame = None

        self.mounting_system = None
        self.air_gap = None
        self.interconnection_type = None

        self.rows_cell = None
        self.row_gutter = None
        self.columns_cell = None
        self.column_gutter = None

        # maps
        self.cell_boolean_map = None
        self.series_map = None
        self.diode_map = None
        self.xyz_relative = None
        self.xyz_absolute = None
        self.normals = None
        self.direct_irradiance = None
        self.diffuse_irradiance = None
        self.effective_irradiance = None

    def create_diode_map(self, map_type, n_size=None, n_size_direction=None):
        if map_type == 'row':
            pass
        elif map_type == 'column':
            pass
        elif map_type == 'chessboard':
            pass
        elif map_type == 'n_size':
            pass
        elif map_type == 'random':
            pass
        else:
            # use n_size method with 3 diodes
            pass

    def print_cell_description(self):
        # this is more of a test than anything
        print(self.cell.cell_description)


def build_parameter_dict(module_dict, custom_module_data, base_parameters):
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

    base_parameters['nom_eff'] = (base_parameters['Wp'] / base_parameters['module_area']) / 1000
    base_parameters['n_cols'] = module_details['n_cols']
    base_parameters['n_rows'] = module_details['n_rows']
    ideal_cell_count = base_parameters['n_cols_ideal'] * base_parameters['n_rows_ideal']
    base_parameters['total_cells'] = base_parameters['n_cols'] * base_parameters['n_rows']
    watts_cell = base_parameters['Wp'] / ideal_cell_count
    base_parameters['cell_peak_Wp'] = watts_cell
    base_parameters['actual_capacity_Wp'] = watts_cell * base_parameters['total_cells']
    m2_cell = base_parameters['module_area'] / ideal_cell_count
    base_parameters['actual_module_area_m2'] = m2_cell * base_parameters['total_cells']
    base_parameters['Wp_m2_module'] = base_parameters['Wp'] / base_parameters['actual_module_area_m2']
    base_parameters['one_subcell_area_m2'] = ((base_parameters['cell_width'] * base_parameters['cell_height']) /
                                              base_parameters['N_subcells']) * 0.000001
    base_parameters['one_cell_area_m2'] = (base_parameters['cell_width'] * base_parameters['cell_height']) * 0.000001
    base_parameters['actual_cell_area_m2'] = base_parameters['one_cell_area_m2'] * base_parameters['total_cells']
    base_parameters['Wp_m2_cell'] = base_parameters['Wp'] / base_parameters['actual_cell_area_m2']

    # base_parameters['minimum_irradiance_cell'] = 100

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
    return base_parameters
