import numpy as np
import gzip
import json
import pickle
import pandas as pd
from operator import itemgetter
import os
import pvlib
import copy

def directory_creator(dir_path):
    if os.path.exists(dir_path):
        pass
    else:
        os.mkdir(dir_path)

def ts_8760(year=2022):
    index = pd.date_range(start=f"01-01-{year} 00:00", end=f"12-31-{year} 23:00", freq="h")
    return index


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


def read_parameter_file(parameter_file):
    """
    :param parameter_file: a file path for the specific text file used for parameters
    :return: cell_parameters: a dict containing the cell parameters in the file
    """
    with open(parameter_file, "r") as fp:
        data = fp.readlines()
        keys = []
        items = []
        for text_line in data:
            if text_line[0] == r"#":
                pass
            else:

                key = str(text_line).split(":")[0]
                keys.append(key)
                item = str(text_line).split(":")[1]
                try:
                    items.append(float(item.strip().strip(" ").strip("'")))
                except ValueError:
                    items.append(str(item.strip().strip("'")))
    cell_parameters = dict(zip(keys, items))
    return cell_parameters


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


def read_json(file_path):
    if ".gz" in file_path:
        with gzip.open(file_path, 'r') as fp:
            json_bytes = fp.read()

        json_str = json_bytes.decode('utf-8')
        data = json.loads(json_str)
    else:
        with open(file_path, 'r') as fp:
            data = json.load(fp)
    return data


def write_json(input_dict, out_path):
    if ".gz" in out_path:
        json_str = json.dumps(input_dict)
        json_bytes = json_str.encode('utf-8')

        with gzip.open(out_path, 'w') as fp:
            fp.write(json_bytes)
    else:
        with open(out_path, 'w') as fp:
            json.dump(input_dict, fp)


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


def read_pickle(file_path, read_method='rb'):
    with open(file_path, read_method) as fp:
        cucumber = pickle.load(fp)
    return cucumber

def write_pickle(cucumber, file_path, write_method="wb"):
    with open(file_path, write_method) as fp:
        pickle.dump(cucumber, fp, protocol=pickle.HIGHEST_PROTOCOL)
    return file_path



def tmy_to_dataframe(path_data):
    tmy_labels = [
        'year', 'month', 'day', 'hour', 'minute', 'datasource', 'drybulb_C',
        'dewpoint_C', 'relhum_percent', 'atmos_Pa', 'exthorrad_Whm2',
        'extdirrad_Whm2', 'horirsky_Whm2', 'glohorrad_Whm2', 'dirnorrad_Whm2',
        'difhorrad_Whm2', 'glohorillum_lux', 'dirnorillum_lux',
        'difhorillum_lux', 'zenlum_lux', 'winddir_deg', 'windspd_ms',
        'totskycvr_tenths', 'opaqskycvr_tenths', 'visibility_km',
        'ceiling_hgt_m', 'presweathobs', 'presweathcodes', 'precip_wtr_mm',
        'aerosol_opt_thousandths', 'snowdepth_cm', 'days_last_snow', 'Albedo',
        'liq_precip_depth_mm', 'liq_precip_rate_Hour'
    ]

    df = pd.read_csv(path_data,
                     skiprows=8,
                     header=None,
                     index_col=False,
                     usecols=list(range(0, 35)),
                     names=tmy_labels)  # .drop('datasource', axis=1)

    df['hour'] = df['hour'].astype(int)
    if df['hour'][0] == 1:
        # print('TMY file hours reduced from 1-24h to 0-23h')
        df['hour'] = df['hour'] - 1
    else:
        pass
        # print('TMY file hours maintained at 0-23hr')
    df['minute'] = 0
    return df


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


def read_map_excel(file_path):
    submodule_map = pd.read_excel(file_path, header=None, sheet_name='submodule').to_numpy()#.tolist()
    subdiode_map = pd.read_excel(file_path, header=None, sheet_name='subdiode').to_numpy()#.tolist()
    subcell_map = pd.read_excel(file_path, header=None, sheet_name='subcell').to_numpy()#.tolist()
    return submodule_map, subdiode_map, subcell_map


def tmy_location(tmy_file):
    with open(tmy_file, "r") as fp:
        tmy_header = fp.readlines()

    tmy_first_line = tmy_header[0].split(",")
    lat = round(float(tmy_first_line[6]), 3)
    lon = round(float(tmy_first_line[7]), 3)
    utc = int(float(tmy_first_line[8]))
    elevation = int(tmy_first_line[9][0:3])
    return {"lat": lat,
            "lon": lon,
            "utc": utc,
            "elevation": elevation}


def create_sun_mask(file_path_sun_up_hours):
    sun = pd.read_csv(file_path_sun_up_hours, names=['HOY'])
    if len(sun[sun['HOY'] == range(1416, 1440)]) > 0:
        print('Leap days detected')

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

def build_full_ill(file_path_sun_up_hours, ill_df):
    if type(ill_df) is str:
        ill_df = pd.read_csv(ill_df, delimiter=' ', header=None).iloc[:, 1:].T.reset_index(drop=True)
    sun_up, sun_hours = create_sun_mask(file_path_sun_up_hours)
    sun_ill = pd.concat([sun_hours, ill_df], axis=1)
    irrad_df = pd.merge(sun_up, sun_ill, how="left", on="HOY").fillna(0)
    del irrad_df['Sunny']
    del irrad_df['HOY']
    return irrad_df

def get_cec_data(cec_key=None, file_path=None):
    if file_path is None:
        mod_df = pvlib.pvsystem.retrieve_sam("CECMod")
    else:
        mod_df = pd.read_csv(file_path, index_col='Unnamed: 0')
    if cec_key is None:
        return mod_df
    else:
        return mod_df[cec_key]



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
            # if panelizer_object.simulation_suite == False:
            #     module_dict['CURVES'][panelizer_object.topology][
            #         'Imod'].update({hoy: np.round(Imod[hoy], 5)})
            #     module_dict['CURVES'][panelizer_object.topology][
            #         'Vmod'].update({hoy: np.round(Vmod[hoy], 5)})
            #     module_dict['YIELD'][panelizer_object.topology][
            #         'irrad'].update({hoy: np.round(Gmod[hoy], 1)})
            # else:
            module_dict['CURVES']["initial_simulation"][
                'Imod'].update({hoy: np.round(Imod[hoy], 5)})
            module_dict['CURVES']["initial_simulation"][
                'Vmod'].update({hoy: np.round(Vmod[hoy], 5)})
            module_dict['YIELD']["initial_simulation"][
                'irrad'].update({hoy: np.round(Gmod[hoy], 1)})

        # if panelizer_object.simulation_suite == True:
        # for topology in panelizer_object.simulation_suite_topologies:
        #     module_dict['CURVES'][topology] = \
        #         copy.deepcopy(
        #             module_dict['CURVES']['initial_simulation'])
        #     module_dict['YIELD'][topology] = \
        #         copy.deepcopy(
        #             module_dict['YIELD']['initial_simulation'])

