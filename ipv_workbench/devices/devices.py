from ipv_workbench.utilities import utils
import os
from ipv_workbench.simulator import simulations as sim
from ipv_workbench.visualize import plots as ipv_plots
import numpy as np


class Cell:
    def __init__(self, parameter_file):
        self.parameter_file = parameter_file
        self.parameters_dict = utils.read_parameter_file(self.parameter_file)
        self.assign_parameters()
        self.iv_library = None
        self.iv_library_conditions = None
        self.iv_library_resolution = None

    def assign_parameters(self):
        # TODO map datatypes to value assignment
        for k, v in self.parameters_dict.items():
            setattr(self, k, v)

    def assign_iv_library(self, profile_path, write_library=True):
        conditions = utils.create_conditions_map()
        if os.path.exists(profile_path):
            self.iv_library = utils.read_compressed_json(profile_path)
            self.iv_library_conditions = conditions
        else:
            simulation_results = {}
            for n in range(0, len(conditions)):
                irrad = conditions[n][0]
                temp = conditions[n][1]
                key = f"{irrad},{temp}"
                simulation_results[key] = {"i": [],
                                           "v": []}
                results = sim.simulate_cell_curve(self.parameters_dict,
                                                  irrad,
                                                  temp,
                                                  self.curve_resolution)
                results = np.array(results)
                simulation_results[key]['i'] = list(results[0, :])
                simulation_results[key]['v'] = list(results[1, :])
                if write_library:
                    utils.write_compressed_json(simulation_results, profile_path)

                self.iv_library = simulation_results
                self.iv_library_conditions = conditions

    def retrieve_curve(self, irrad, temp):
        iv_key = utils.find_matching_key(self.iv_library.keys(), self.iv_library_conditions, irrad, temp)
        return iv_key, self.iv_library[iv_key]['i'], self.iv_library[iv_key]['v']


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
            figsize = (5,3)

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
        self.cell = None
        self.front_film = None
        self.front_cover = None
        self.rear_cover = None
        self.encapsulant = None
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
        if map_type=='row':
            pass
        elif map_type=='column':
            pass
        elif map_type=='chessboard':
            pass
        elif map_type=='n_size':
            pass
        elif map_type=='random':
            pass
        else:
            # use n_size method with 3 diodes
            pass

    def print_cell_description(self):
        # this is more of a test than anything
        print(self.cell.cell_description)




