import configparser
import pathlib

def format_config_item(section, key, value):

    config_dtypes = {'management': {'project_name': 'str',
                                    'parent_dir': 'path',
                                    'latitude': 'float',
                                    'longitude': 'float',
                                    'elevation': 'float',
                                    'utc': 'int',
                                    'host_name': 'str',
                                    'raw_host_file': 'path',
                                    'exclude_surfaces': 'list-str',
                                    'project_data': 'path',
                                    'scenario_name': 'str',
                                    'base_epw': 'path',
                                    'tmy_name': 'str',
                                    'timezone': 'str'},
                     'analysis': {'device_id': 'str',
                                  'analysis_period': 'str',
                                  'active_surface': 'str',
                                  'n_workers': 'int'},
                     'irradiance': {'grid_x_mm': 'float',
                                    'grid_y_mm': 'float',
                                    'radiance_param_rflux': 'str',
                                    'radiance_param_rcontrib': 'str',
                                    'n_workers': 'int',
                                    'store_radiance': 'bool',
                                    'use_accelerad': 'bool'},
                     }
    bool_map = {'true':True,'false':False}
    if (value == 'None') or (value is None):
        value_out = None
    else:
        format_type = config_dtypes[section][key]
        if format_type == 'path':
            if key == 'base_epw':
                if value == 'None':
                    value_out = None
                else:
                    value_out = pathlib.Path(value)
            else:
                value_out = pathlib.Path(value)
        elif format_type == 'str':
            value_out = str(value)
        elif format_type == 'int':
            value_out = int(value)
        elif format_type == 'float':
            value_out = float(value)
        elif format_type == 'bool':
            value_out = bool_map[str(value).lower()]
        elif format_type == 'list-str':
            value_out = [str(v) for v in value.split(",")]
        elif format_type == 'list-int':
            value_out = [int(v) for v in value.split(",")]
        elif format_type == 'list-float':
            value_out = [float(v) for v in value.split(",")]
        elif format_type == 'list-bool':
            value_out = [bool_map[str(v).lower()] for v in value.split(",")]
        elif format_type == 'tuple-float':
            value_out = tuple([float(v) for v in value.split(",")])
        else:
            value_out = str(value)

    return value_out


def edit_cfg_file(config_path, section, key, new_value):
    parser = configparser.ConfigParser()
    parser.read(config_path)
    # value_out = format_config_item(section, key, new_value)
    parser.set(section, key, str(new_value))

    # Writing our configuration file to 'example.ini'
    with open(config_path, 'w') as configfile:
        parser.write(configfile)
