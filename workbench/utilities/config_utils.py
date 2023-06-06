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
                     'irradiance': {'grid_x_mm': 'float',
                                    'grid_y_mm': 'float',
                                    'radiance_parameters': 'str'},
                     }

    if value == 'None':
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
            value_out = bool(value)
        elif format_type == 'list-str':
            value_out = [str(v) for v in value.split(",")]
        elif format_type == 'list-int':
            value_out = [int(v) for v in value.split(",")]
        elif format_type == 'list-float':
            value_out = [float(v) for v in value.split(",")]
        elif format_type == 'list-bool':
            value_out = [bool(v) for v in value.split(",")]
        elif format_type == 'tuple-float':
            value_out = tuple([float(v) for v in value.split(",")])
        else:
            value_out = str(value)

    return value_out


def edit_cfg_file(config_path, section, key, new_value):
    parser = configparser.ConfigParser()
    parser.read(config_path)
    parser.set(section, key, str(new_value))

    # Writing our configuration file to 'example.ini'
    with open(config_path, 'w') as configfile:
        parser.write(configfile)
