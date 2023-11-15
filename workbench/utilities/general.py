import math
import sys
import numpy as np
import pandas as pd
from operator import itemgetter
import os
import pvlib
import subprocess
from scipy.spatial.distance import cdist
from workbench.utilities import io


def collect_raw_irradiance(pv_cells_xyz_arr, sensor_pts_xyz_arr, sensor_pts_irradiance_arr, method="closest"):
    # TODO add rectangular sampling based on cell dimensions
    # print("PV Cells", pv_cells_xyz_arr.shape)
    # print("Sensor XYZ", sensor_pts_xyz_arr.shape)
    # print("Sensor irrad", sensor_pts_irradiance_arr.shape)
    cdist_arr = cdist(pv_cells_xyz_arr, sensor_pts_xyz_arr)

    if method == 'closest':
        first = cdist_arr.argsort()[:, 0]
        irrad_cell_mean = sensor_pts_irradiance_arr.T[first]
    elif method == 'mean':
        first = cdist_arr.argsort()[:, 0]
        second = cdist_arr.argsort()[:, 1]
        third = cdist_arr.argsort()[:, 2]
        irrad_cell_mean = (sensor_pts_irradiance_arr.T[first] + sensor_pts_irradiance_arr.T[second] +
                           sensor_pts_irradiance_arr.T[third]) / 3
    else:
        print("The arg 'method' must be specified as either 'closest' or 'mean' (mean of nearest 3 points). "
              "Defaulting to closest.")
        first = cdist_arr.argsort()[:, 0]
        irrad_cell_mean = sensor_pts_irradiance_arr.T[first]




    return irrad_cell_mean.T


def flatten_list(lst):
    return [j for i in lst for j in i]


def flip_maps(arr):
    """
    The panelizer from grasshopper writes the cells in a different order
    than the maps are created. This corrects them so if they are flattened
     they ar indexed the same. It only does this if the input for the object
      was a file path and the map is a list (not a numpy array). This is
      controlled in the panelizer object function itselfZ.
    :param arr: the map (diode or submodule)
    :return:
    """
    if type(arr) is list:
        return np.fliplr(np.array(arr).T)
    else:
        return arr


# TODO build param file generator
# def generate_parameter_file():


def create_conditions_map(temp_min=-45, temp_max=45, temp_step=0.25, irrad_min=0, irrad_max=1000, irrad_step=5):
    """
    Defines a list of tuples that are all combinations of a range of temperatures and irradiance levels.
    :param temp_min: minimum temperature in list (˚C) Default -45
    :param temp_max: maximum temperature in list (˚C) Default 45
    :param temp_step: step in ˚C to set the resolution of the list at
    :param irrad_min: minimum irradiance in list (W/m2) Default 0
    :param irrad_max: maximum irradiance in list (W/m2) Default 1000
    :param irrad_step: step in W/m2 to set the resolution of the list at
    :return: combinations: a list of tuples where the first index is irradiance and the second is temp
    """
    irrad_space = np.arange(irrad_min, irrad_max + irrad_step, irrad_step)
    temp_space = np.arange(temp_min, temp_max + temp_step, temp_step)

    combinations = []
    for irrad in irrad_space:
        for temp in temp_space:
            combinations.append((irrad, temp))
    return np.array(combinations)


def chunk_list(lst, n):
    """
    yield successive n-sized chunks of a list
    :param lst: input list to chunk
    :param n: size of chunks
    :return: list of chunked lists
    """
    new_list = []
    for i in range(0, len(lst), n):
        new_list.append(lst[i:i + n])
    return new_list


def find_unique_condition(conditions_list):
    unique_conditions = list(set(conditions_list))
    unique_conditions.sort()
    return unique_conditions


def find_step(conditions, variable):
    if variable == 'irradiance':
        conditions = conditions[:, 0]
    elif variable == 'temperature':
        conditions = conditions[:, 1]
    else:
        return ValueError("Must enter 'irradiance' or 'temperature' as variable")

    unique_conditions = find_unique_condition(conditions)
    return unique_conditions[1] - unique_conditions[0]


def round_nearest(x, a):
    return round(x / a) * a


def find_nearest(search_value, cell, search_var):
    nearest_value = round_nearest(search_value,
                                  find_step(cell.iv_library_conditions,
                                            variable=search_var))
    return nearest_value


def find_matching_key(all_keys, conditions, search_irrad, search_temp):
    nearest_irrad = float(round_nearest(search_irrad, find_step(conditions, variable='irradiance')))
    nearest_temp = int(round_nearest(search_temp, find_step(conditions, variable='temperature')))

    matching_irrad = [k for k in all_keys if nearest_irrad == int(k.split(",")[0])]
    matching_temp = [k for k in matching_irrad if nearest_temp == float(k.split(",")[1])]
    if len(matching_temp) == 0:
        return ValueError("Input 'search_irrad' and/or 'search_temp' did not return a result"
                          "Try to modify the search so irrad is 0 to 1000 and temp -45 to 45")
    else:
        return matching_temp[0]


def archive_expand_ndarray_2d_3d_slow(ndarray_input):
    empty_arr = np.zeros((ndarray_input.shape[0],
                          ndarray_input.shape[1],
                          ndarray_input[0][0].shape[0]))

    for n, i in enumerate(np.ndindex(empty_arr.shape)):
        empty_arr[i[0], i[1], i[2]] = np.array(ndarray_input[i[0], i[1]][i[2]])
    return empty_arr


def expand_ndarray_2d_3d(ndarray_input):
    return np.array([[np.array(arr_) for arr_ in arr] for arr in ndarray_input])


def archive_extract_curves_multiple_cells(irradiance_arr, temperature_arr, cell):
    i_list = []
    v_list = []
    for params in list(zip(irradiance_arr, temperature_arr)):
        # k, i, v = cell.retrieve_curve(params[0], params[1])
        k, i, v = cell.retrieve_curve(int(find_nearest(params[0], cell, "irradiance")),
                                      float(find_nearest(params[1], cell, "temperature"))
                                      )
        i_list.append(i)
        v_list.append(v)

    iv_curves = np.array([i_list, v_list])
    return iv_curves


def generate_empty_results_dict(target):
    empty_dict = {'imp': {},
                  'vmp': {},
                  'pmp': {},
                  'isc': {},
                  'voc': {},
                  'ff': {},
                  'irrad': {},
                  'eff': {}}

    if target == 'SRTING':

        empty_dict.pop('isc', None)
        empty_dict.pop('voc', None)

    elif target == 'SURFACE':

        empty_dict.pop('isc', None)
        empty_dict.pop('voc', None)

    elif target == 'OBJECT':

        empty_dict.pop('imp', None)
        empty_dict.pop('vmp', None)
        empty_dict.pop('isc', None)
        empty_dict.pop('voc', None)
        empty_dict.pop('ff', None)

    return empty_dict


def gather_sublevel_results(panelizer_object, larger_dict, sublevel_iterable, result_key):
    dict_list = []
    for sub in sublevel_iterable:
        sub_dict = larger_dict[sub]['YIELD'][panelizer_object.topology]
        dict_list.append(sub_dict[result_key])

    return {k: sum(map(itemgetter(k), dict_list)) for k in dict_list[0]}


def mask_nd(x, m):
    '''
    Mask a 2D array and preserve the
    dimension on the resulting array
    ----------
    x: np.array
       2D array on which to apply a mask
    m: np.array
        2D boolean mask
    Returns
    -------
    List of arrays. Each array contains the
    elements from the rows in x once masked.
    If no elements in a row are selected the
    corresponding array will be empty
    https://stackoverflow.com/questions/53918392/mask-2d-array-preserving-shape
    '''
    take = m.sum(axis=1)
    return np.array([arr for arr in np.split(x[m], np.cumsum(take)[:-1]) if len(arr) > 0])


def create_voltage_range(sde_args, kwargs, curve_pts=1000):
    v_oc = pvlib.singlediode.bishop88_v_from_i(0.0, *sde_args, **kwargs)
    evaluated_voltages = np.linspace(0.95 * kwargs['breakdown_voltage'], v_oc * 1.05, curve_pts)
    return evaluated_voltages


def tmy_location(tmy_file):
    with open(tmy_file, "r") as fp:
        tmy_header = fp.readlines()

    tmy_first_line = tmy_header[0].split(",")
    tmy_first_line = [l.strip('\n') for l in tmy_first_line]
    lat = round(float(tmy_first_line[6]), 3)
    lon = round(float(tmy_first_line[7]), 3)
    utc = int(float(tmy_first_line[8]))
    elevation = int(float(tmy_first_line[9][0:3]))
    return {"lat": lat,
            "lon": lon,
            "utc": utc,
            "elevation": elevation}


def create_sun_mask(file_path_sun_up_hours):
    sun = pd.read_csv(file_path_sun_up_hours, names=['HOY'])
    # if len(sun[sun['HOY'] == range(1416, 1440)]) > 0:
    #     print('Leap days detected')

    sun_hours = np.floor(sun).astype(int)
    empty = pd.DataFrame(data={'HOY': list(range(0, 8760))})

    def eval_sun_up(HOY, sun_up_list):
        if HOY in sun_up_list:
            return True
        else:
            return False

    sun_up = empty.apply(lambda x: eval_sun_up(x['HOY'], sun_hours['HOY'].tolist()), axis=1)
    sun_up = pd.DataFrame(sun_up).reset_index().rename(columns={"index": "HOY", 0: "Sunny"})
    return sun_up, sun_hours


def build_full_ill(ill_df, wea_file=None):
    """

    :param ill_df: either the .ill file read as a dataframe or the path to it
    :return:
    """
    if type(ill_df) is str:
        ill_df = io.read_ill(ill_df)

    if wea_file is None:
        pass
    else:
        wea, header = io.read_wea(wea_file)
        ill_df.set_index(wea.index, inplace=True, drop=True)

    return ill_df


def unpack_mp_results(mp_results, panelizer_object, surface, string, modules, timeseries):
    results_dict = {}
    for r in mp_results:  # size is based on n_cpu (just reorganizing from the pool)
        results_dict.update(r)

    for module in modules:
        module_dict = panelizer_object.get_dict_instance([surface, string, module])

        module_results_dict = results_dict[module]
        Imod = module_results_dict[0]
        Vmod = module_results_dict[1]
        Gmod = module_results_dict[2]

        module_dict.update({'PARAMETERS': module_results_dict[3]})
        for hoy in timeseries:
            module_dict['CURVES']['Imod'].update({hoy: np.round(Imod[hoy], 5)})
            module_dict['CURVES']['Vmod'].update({hoy: np.round(Vmod[hoy], 5)})
            module_dict['YIELD']["initial_simulation"]['irrad'].update({hoy: np.round(Gmod[hoy], 1)})


def clean_grasshopper_key(key):
    return key.replace("{", "").replace("}", "").replace(";", "_")




def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


def log_run(file_path, write_string):
    # file_path = os.path.join(folder,"runtime.txt")
    if os.path.exists(file_path):
        with open(file_path, "a") as fp:
            fp.write(write_string)
    else:
        with open(file_path, "w+") as fp:
            fp.write(write_string)


def get_object_capacity(object_hourly_results):
    capacity_cols = []
    for col in object_hourly_results.columns:
        if "_kwp" in col:
            if "kwh" in col:
                pass
            else:
                capacity_cols.append(col)

    return object_hourly_results[capacity_cols].iloc[0].sum()


def get_object_surface_area(object_hourly_results):
    area_cols = []
    for col in object_hourly_results.columns:
        if "surface_area" in col:
            area_cols.append(col)

    return object_hourly_results[area_cols].iloc[0].sum()


def count_lines(text_file):
    with open(text_file, "r") as fp:
        line_count = len(fp.readlines())
    return line_count


def run_command(command):
    """
    :param command:
    :return:
    """
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    return stdout, stderr, process.returncode


def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True