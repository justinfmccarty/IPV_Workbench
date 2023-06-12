import pandas as pd
import sys
import os
import random
import numpy as np

import workbench.utilities.io
from workbench.old_solver import calculations
from workbench.utilities import general


def read_sample_irradiance(stream):
    return pd.read_csv(os.path.join(os.path.dirname(__file__),
                                    f"irradiance_profiles_{stream}.csv"))


def get_sample_profiles(number, stream, method='random'):
    df = read_sample_irradiance(stream)
    if method == 'random':
        randomlist = []
        for i in range(0, number):
            col = str(random.randint(0, 99))
            randomlist.append(df[col].tolist())
    elif method == 'top25':
        randomlist = []
        for i in range(0, number):
            col = str(random.randint(0, 24))
            randomlist.append(df[col].tolist())
    elif method == 'middle50':
        randomlist = []
        for i in range(0, number):
            col = str(random.randint(25, 74))
            randomlist.append(df[col].tolist())
    elif method == 'lowest25':
        randomlist = []
        for i in range(0, number):
            col = str(random.randint(75, 99))
            randomlist.append(df[col].tolist())
    else:
        print("Defaulting to random")
        randomlist = []
        for i in range(0, number):
            col = str(random.randint(0, 99))
            randomlist.append(df[col].tolist())

    return randomlist


def generate_sample_module_maps(n_rows=6, n_cols=20, module_template='a1', ndiodes=3):
    # defaults will return a standard 120 cell landscape orientation square cut module
    # returns two arrays (1. is the submodule map, 2. is the diode mask)
    if module_template == 'a1':
        submodules_arr = np.zeros(n_rows * n_cols, dtype=int).reshape(n_rows, n_cols)

        diodes_arr = submodules_arr.copy()
        diodes_arr[0:2, :] = 0
        diodes_arr[2:4, :] = 1
        diodes_arr[4:6, :] = 2

    elif module_template == 'a2':
        submodules_arr = np.zeros(n_rows * n_cols, dtype=int).reshape(n_rows, n_cols)
        submodules_arr[:, int(n_cols / 2):] = 1

        diodes_arr = submodules_arr.copy()
        diodes_arr[0:2, :int(n_cols / 2)] = 0
        diodes_arr[2:4, :int(n_cols / 2)] = 1
        diodes_arr[4:6, :int(n_cols / 2)] = 2
        diodes_arr[0:2, int(n_cols / 2):] = 0
        diodes_arr[2:4, int(n_cols / 2):] = 1
        diodes_arr[4:6, int(n_cols / 2):] = 2
    # elif module_template=='b1':
    #     submodules_arr = np.zeros(nrows*ncols,dtype=int).reshape(nrows,ncols)
    # elif module_template=='b2':
    #     submodules_arr = np.zeros(nrows*ncols,dtype=int).reshape(nrows,ncols)

    return submodules_arr, diodes_arr


def generate_sample_module_irrad(stream, n_rows=6, n_cols=20):
    irrad_df = get_sample_profiles(n_rows * n_cols, stream)
    irrad_arr = np.zeros((n_rows, n_cols), dtype=np.ndarray)
    for n, i in enumerate(np.ndindex(irrad_arr.shape)):
        irrad_arr[i[0], i[1]] = np.array(irrad_df[n])

    return irrad_arr


def calculate_cell_temperature_array(G_eff_arr, dbt_arr):
    temp_C_arr = np.zeros(G_eff_arr.shape, dtype=np.ndarray)
    for n, i in enumerate(np.ndindex(G_eff_arr.shape)):
        temperatures = calculations.calculate_cell_temperature(G_eff_arr[i[0], i[1]],
                                                               dbt_arr,
                                                               None)
        temp_C_arr[i[0], i[1]] = np.array(temperatures)
    return temp_C_arr


def generate_sample_building_dict(n_surfaces, n_strings, n_modules):
    base_dict = {'BUILDING':
                     {"DETAILS": {},
                      'YIELD': {'central_inverter': utils.generate_empty_results_dict(target='OBJECT'),
                                'string_inverter': utils.generate_empty_results_dict(target='OBJECT'),
                                'micro_inverter': utils.generate_empty_results_dict(target='OBJECT')
                                },
                      "CURVES": {'central_inverter': {"Isys": {}, "Vsys": {}},
                                 'string_inverter': {"Isys": {}, "Vsys": {}},
                                 'micro_inverter': {"Isys": {}, "Vsys": {}}
                                 },
                      "SURFACES": {}
                      }
                 }

    for n_srf in range(0, n_surfaces):
        srf_key = f"Srf{n_srf}"
        srf_dict = {srf_key: {"DETAILS": {},
                              'YIELD': {
                                  'central_inverter': utils.generate_empty_results_dict(target='SURFACE'),
                                  'string_inverter': utils.generate_empty_results_dict(target='SURFACE'),
                                  'micro_inverter': utils.generate_empty_results_dict(target='SURFACE'),
                              },
                              "CURVES": {'central_inverter': {"Isrf": {}, "Vsrf": {}},
                                         'string_inverter': {"Isrf": {}, "Vsrf": {}},
                                         'micro_inverter': {"Isrf": {}, "Vsrf": {}}
                                         },
                              'RESULTSDICT': {},
                              "STRINGS": {}}}
        for n_str in range(0, n_strings):
            str_key = f"0:0:{n_str}"
            srf_dict[srf_key]["STRINGS"][str_key] = {"DETAILS": {},
                                                     "MODULES": {}}
            for n_mod in range(0, n_modules):
                mod_key = f"0:0:{n_str}" + f":{n_mod}"
                srf_dict[srf_key]["STRINGS"][str_key]['MODULES'][mod_key] = generate_sample_module_dict()
                srf_dict[srf_key]["STRINGS"][str_key]['YIELD'] = {
                    'central_inverter': utils.generate_empty_results_dict(target='STRING'),
                    'string_inverter': utils.generate_empty_results_dict(target='STRING'),
                    'micro_inverter': utils.generate_empty_results_dict(target='STRING'),
                }
                srf_dict[srf_key]["STRINGS"][str_key]["CURVES"] = {'central_inverter': {"Istr": {}, "Vstr": {}},
                                                                   'string_inverter': {"Istr": {}, "Vstr": {}},
                                                                   'micro_inverter': {"Istr": {}, "Vstr": {}}}

        base_dict['BUILDING']['SURFACES'].update(srf_dict)
    return base_dict


def generate_sample_module_dict(n_rows=6, n_cols=20, module_template='a1'):
    tmy_file = os.path.join(os.path.dirname(__file__),
                            "zurich_2007_2021.epw")
    tmy_df = workbench.utilities.io.read_epw(tmy_file)
    submodule_map, diode_pathways_map = generate_sample_module_maps(
        n_rows, n_cols, module_template)
    module_irrad_direct = generate_sample_module_irrad('direct', n_rows, n_cols)
    module_irrad_diffuse = generate_sample_module_irrad('diffuse', n_rows, n_cols)
    module_irrad_eff = calculations.calculate_effective_irradiance(module_irrad_direct, module_irrad_diffuse)
    module_temps = calculate_cell_temperature_array(module_irrad_eff, tmy_df.drybulb_C.to_numpy())
    sample_module = {
        "DETAILS": {},
        "LAYERS": {},
        "MAPS": {
            "SUBMODULES": submodule_map,
            "DIODES": diode_pathways_map
        },
        "CELLSXYZ": [],
        "CELLSNORMALS": [],
        "CELLSIRRADDIRECT": module_irrad_direct,
        "CELLSIRRADDIFFUSE": module_irrad_diffuse,
        "CELLSIRRADEFF": module_irrad_eff,
        "CELLSTEMP": module_temps,
        "YIELD": {'initial_simulation': utils.generate_empty_results_dict(target='MODULE'),
                  'central_inverter': utils.generate_empty_results_dict(target='MODULE'),
                  'string_inverter': utils.generate_empty_results_dict(target='MODULE'),
                  'micro_inverter': utils.generate_empty_results_dict(target='MODULE')},
        "CURVES": {'initial_simulation': {"Imod": {}, "Vmod": {}},
                   'central_inverter': {"Imod": {}, "Vmod": {}},
                   'string_inverter': {"Imod": {}, "Vmod": {}},
                   'micro_inverter': {"Imod": {}, "Vmod": {}}
                   }
    }
    return sample_module
