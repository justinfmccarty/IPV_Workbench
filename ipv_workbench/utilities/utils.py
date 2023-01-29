import numpy as np
import gzip
import json
import pickle
import pandas as pd


def ts_8760(year=2022):
    index = pd.date_range(start=f"01-01-{year} 00:00",end=f"12-31-{year} 23:00",freq="h")
    return index

def flatten_list(lst):
    return [j for i in lst for j in i]

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

def find_mpp(iv_arr):
    power_arr = iv_arr[0, :] * iv_arr[1, :]
    idx = np.argmax(power_arr)
    return power_arr, iv_arr[0, :][idx], iv_arr[1, :][idx]


def apply_bypass_diode(Vsub, module_params):
    return np.clip(Vsub, a_min=module_params['diode_threshold'], a_max=None)


def read_pickle(file_path, read_method='rb'):
    with open(file_path, read_method) as fp:
        loaded_file = pickle.load(fp)
    return loaded_file

def calc_short_circuit(iv_curves):
    substring_Isc = [np.interp(0,
                            sub[:,1], # V curve valeus
                            sub[:,0]) # I curve values
                    for sub in iv_curves]
    return np.array(substring_Isc)
def calc_current_max(iv_curves, cell):
    substring_Imax = [np.interp(cell.breakdown_voltage,
                            sub[:,1], # V curve valeus
                            sub[:,0]) # I curve values
                    for sub in iv_curves]
    return np.array(substring_Imax)