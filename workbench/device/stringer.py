import numpy as np
from workbench.utilities import general, constants


def split_large_string_group(building, surface, initial_string_group, inverter_capacity, vector):
    # surface = general.reverse_grasshopper_key(surface_dict['Details']['radiance_surface_label'])
    # initial_string_group = surface_dict['Strings'][string_key]
    total_capacity = initial_string_group['total_capacity_kWp']
    total_modules = initial_string_group['count']
    modules = initial_string_group['modules']
    # general_direction = surface_dict['Details']['general_angle']
    module_capacity_Wp = initial_string_group['module_capacity_Wp']

    n_strings = np.ceil(total_capacity / inverter_capacity)
    n_per = np.ceil(total_modules / n_strings)

    x_list = []
    y_list = []
    z_list = []

    string_dicts = {}

    # get the point values
    for module in modules:
        module_dict = building.get_dict_instance([surface, module])
        mod_x, mod_y, mod_z = module_dict['Parameters']['panelizer_center_pt_inplane']
        x_list.append(mod_x)
        y_list.append(mod_y)
        z_list.append(mod_z)
        if vector == 'x':
            key = mod_x
        else:
            key = mod_y

        if key in string_dicts.keys():
            string_dicts[key].append(module)
        else:
            string_dicts[key] = [module]

    string_dicts = dict(sorted(string_dicts.items()))
    return create_lists_from_dict(string_dicts, n_strings, n_per)


def create_lists_from_dict(input_dict, num_lists, max_list_size):
    num_lists = int(num_lists)
    max_list_size = int(max_list_size)
    result_lists = [[] for _ in range(num_lists)]

    current_list_index = 0

    for key, values in sorted(input_dict.items()):
        for item in values:
            result_lists[current_list_index].append(item)

            if len(result_lists[current_list_index]) == max_list_size:
                current_list_index = (current_list_index + 1) % num_lists

    return result_lists


def vector_control_stringer(building, surface_dict, initial_string_group, inverter_capacity):
    general_direction = surface_dict['Details']['general_angle']
    surface = general.reverse_grasshopper_key(surface_dict['Details']['radiance_surface_label'])
    if (general_direction == 'east') or (general_direction == 'west'):
        # organize the strings based on the module center point x values
        vector = "x"
    else:
        # organize the strings based on the module center point y values
        vector = "y"

    return split_large_string_group(building, surface, initial_string_group, inverter_capacity, vector)


def split_up_initial_strings(building, surface_dict, string_key):
    initial_string_group = surface_dict['Strings'][string_key]
    total_capacity = initial_string_group['total_capacity_kWp']
    print(total_capacity)
    if total_capacity < 0.5:  # kW
        # one string using a 0.5kW inverter
        new_string_groups = [initial_string_group['modules']]
    elif (total_capacity >= 0.5) & (total_capacity < 20):  # kW
        # split into multiple strings using the 2.5kW inverters
        inverter_capacity = 2.5
        new_string_groups = vector_control_stringer(building, surface_dict, initial_string_group, inverter_capacity)
    elif (total_capacity >= 20) & (total_capacity < 60):  # kW
        # split into multiple strings using the 10kW inverters
        inverter_capacity = 10
        new_string_groups = vector_control_stringer(building, surface_dict, initial_string_group, inverter_capacity)

    elif (total_capacity >= 60) & (total_capacity < 200):  # kW
        # split into multiple strings using the 20kW inverters
        inverter_capacity = 20
        new_string_groups = vector_control_stringer(building, surface_dict, initial_string_group, inverter_capacity)
    else:
        # split into multiple strings using the 500kW inverters
        inverter_capacity = 500
        new_string_groups = vector_control_stringer(building, surface_dict, initial_string_group, inverter_capacity)

    return new_string_groups


def building_string_map(building, surface):
    surface_dict = building.get_dict_instance([surface])

    capacities = building.get_surface_capacities(surface)
    string_keys = ["s" + str(n).zfill(3) for n in range(0, len(capacities))]
    # string_keys = list(constants.alphabet[0:len(capacities)].strip())

    string_map = {}

    for string_key, capacity in zip(string_keys, capacities):
        string_map[string_key] = {'modules': [],
                                  'total_capacity_kWp': 0,
                                  'count': 0}

        for module in building.get_modules(surface):
            module_dict = building.get_dict_instance([surface, module])
            if module_dict['Parameters']['param_actual_capacity_Wp'] == capacity:
                string_map[string_key]['modules'].append(module)

            else:
                pass

        string_map[string_key]['count'] = len(string_map[string_key]['modules'])
        string_map[string_key]['total_capacity_kWp'] = capacity * string_map[string_key]['count'] / 1000
        string_map[string_key]['module_capacity_Wp'] = capacity

    surface_dict['Strings'] = string_map

    new_string_set = []
    new_module_capacity_set = []

    for string_key in surface_dict['Strings'].keys():
        new_string_list = split_up_initial_strings(building, surface_dict, string_key)
        new_string_set.append(new_string_list)
        module_capacity = surface_dict['Strings'][string_key]['module_capacity_Wp']
        new_module_capacity_set.append([module_capacity] * len(new_string_list))

    new_string_set = general.flatten_list(new_string_set)
    new_module_capacity_set = general.flatten_list(new_module_capacity_set)
    # new_string_keys = list(constants.alphabet[0: len(new_string_set)].strip())
    new_string_keys = ["s" + str(n).zfill(3) for n in range(0, len(new_string_set))]
    new_string_map = {}
    for new_string_key, new_string_module_list, new_capacity in zip(
            new_string_keys, new_string_set, new_module_capacity_set
    ):
        new_string_map[new_string_key] = {
            "modules": [],
            "total_capacity_kWp": 0,
            "count": 0,
            'Curves': {'Istr':{},
                       'Vstr':{}},
            'Yield': general.generate_empty_results_dict(target='STRING'),
            'Details': {}
        }

        for module in new_string_module_list:
            new_string_map[new_string_key]["modules"].append(module)

        new_string_map[new_string_key]["count"] = len(
            new_string_map[new_string_key]["modules"]
        )
        new_string_map[new_string_key]["total_capacity_kWp"] = (
                new_capacity * new_string_map[new_string_key]["count"] / 1000
        )
        new_string_map[new_string_key]["module_capacity_Wp"] = new_capacity

    surface_dict['Strings'] = new_string_map
    return new_string_map
