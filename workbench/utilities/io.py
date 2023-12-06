import gzip
import json
import lzma
import os
import pathlib
import pickle
import shutil
import glob
import pvlib
import pandas as pd
import calendar
from workbench.utilities import temporal, general


def get_cec_data(cec_key=None, file_path=None):
    if file_path is None:
        mod_df = pvlib.pvsystem.retrieve_sam("CECMod")
    else:
        mod_df = pd.read_csv(file_path, index_col='Unnamed: 0')
    if cec_key is None:
        return mod_df
    else:
        return mod_df[cec_key]


def load_irradiance_file(project, radiance_surface_key, component):
    input_surface_dir = os.path.join(project.management_parent_dir, project.management_project_name, "inputs", "hosts", project.management_host_name,
                                     project.management_scenario_name, "radiance", f"{radiance_surface_key}")
    wea_filepath = os.path.join(input_surface_dir, "model", f"{project.management_scenario_name}.wea")
    output_surface_dir = os.path.join(project.management_parent_dir, project.management_project_name, "outputs", project.management_host_name,
                                      project.management_scenario_name, "irradiance", f"{radiance_surface_key}")
    results_filepath = os.path.join(output_surface_dir, f"{component}.lz4")
    ill_df = general.build_full_ill(results_filepath, wea_file=wea_filepath)
    ill_df.sort_index(inplace=True)
    return ill_df


def read_wea(wea_file, year=2030):
    with open(wea_file, "r") as fp:
        header = fp.readlines()[0:6]

    # get wea data
    df = pd.read_csv(wea_file, skiprows=6, header=None, sep=" ")
    df.rename(columns={0: "month", 1: "day", 2: "hour"}, inplace=True)
    df['year'] = year

    # create datetime index
    df.set_index(pd.to_datetime(df[["year", "month", "day", "hour"]]), inplace=True)
    return df, header

def find_ill_skip(fp):
    break_line = None
    with open(fp, "r") as fp_:
        for n, line in enumerate(fp_.readlines()):
            if "FORMAT=ascii" in line:
                break_line = n
                break
    return break_line + 1

def read_ill(filepath):
    """

    :param filepath: the ill filepath
    :return: a pandas dataframe where each column is a snesor point and the rows coordinate to the timeseries analysed
    """
    # this works on honeybee files
    # return pd.read_csv(filepath, delimiter=' ', header=None, dtype='float32').iloc[:, 1:].T.reset_index(drop=True)
    if pathlib.Path(filepath).suffix == ".ill":
        skiprows_n = find_ill_skip(filepath)
        df = pd.read_csv(filepath, header=None, skiprows=skiprows_n, delimiter=' ', dtype='float')
        # df = df[range(1, len(df.columns))].round(2)
        df = df.round(2)
    else:
        df = pd.read_feather(filepath)
        # df = df[range(1, len(df.columns))].round(2)
        df = df.round(2)
    return df


def load_grid_file(project, radiance_surface_key):
    input_surface_dir = os.path.join(project.management_parent_dir, project.management_project_name, "inputs", "hosts", project.management_host_name,
                                     project.management_scenario_name, "radiance", f"{radiance_surface_key}")

    pts_path = glob.glob(os.path.join(input_surface_dir, "model", "grid", "*.pts"))[0]
    return load_sensor_points(pts_path)


def load_sensor_points(sensor_file):
    return pd.read_csv(sensor_file, sep=' ', header=None, dtype='float64', names=["X", "Y", "Z", "X_v", "Y_v", "Z_v"])


def directory_creator(dir_path):
    if os.path.exists(dir_path):
        pass
    else:
        os.makedirs(dir_path)


def copy_file(fsrc, fdst):
    if not os.path.exists(fdst):
        shutil.copyfile(fsrc, fdst)
    else:
        print(f"Destination file already exists, copy aborted./n{fdst}")


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


def read_pickle(file_path, read_method='rb'):
    extension = file_path.split(".")[-1]
    if extension == 'xz':
        with lzma.open(file_path, read_method) as fp:
            cucumber = pickle.load(fp)
    else:
        with open(file_path, read_method) as fp:
            cucumber = pickle.load(fp)
    return cucumber


def write_pickle(cucumber, file_path, write_method="wb", compress=False):
    if compress == False:
        with open(file_path, write_method) as fp:
            pickle.dump(cucumber, fp, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with lzma.open(file_path, write_method) as fp:
            pickle.dump(cucumber, fp)
    return file_path


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


def create_log_file(destination_path):
    first_line = "date,scenario,device_id,module_cover,host,simulation_type,n_negatives,n_workers,n_points,rad_par,runtime [sec]\n"
    with open(destination_path, 'w') as fp:
        fp.writelines([first_line])


def read_map_excel(file_path):
    submodule_map = pd.read_excel(file_path, header=None, sheet_name='submodule').to_numpy()  # .tolist()
    subdiode_map = pd.read_excel(file_path, header=None, sheet_name='subdiode').to_numpy()  # .tolist()
    subcell_map = pd.read_excel(file_path, header=None, sheet_name='subcell').to_numpy()  # .tolist()
    return submodule_map, subdiode_map, subcell_map


def read_epw(path_data, create_timeseries=True):
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
                     # usecols=list(range(0, 35)),
                     names=tmy_labels).drop('datasource', axis=1)

    if calendar.isleap(df['year'].tolist()[0]):
        df['year'] += 1

    df['hour'] = df['hour'].astype(int)
    if df['hour'][0] == 1:
        # print('TMY file hours reduced from 1-24h to 0-23h')
        df['hour'] = df['hour'] - 1
    else:
        pass
        # print('TMY file hours maintained at 0-23hr')
    df['minute'] = 0

    if create_timeseries == False:
        pass
    else:
        df.set_index(temporal.ts_8760(df['year'].tolist()[0]), inplace=True)

    return df