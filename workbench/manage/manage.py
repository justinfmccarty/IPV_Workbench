import glob
import os
import pandas as pd
import pathlib
import shutil
import pvlib
import configparser
from workbench.utilities import general, config_utils, temporal, io


def initiate_project(parent_directory, project_name, project_epw):
    """


    :param parent_directory: the directory within which to create a folder to hold the project inputs and outputs
    :param project_name: the name of the project which will be used to create the folder in the parent_directory
    :param project_epw: a project-specific EPW file located somewhere within your file directory
    :return: file path to the newly created configuration file
    """
    project_directory = os.path.join(parent_directory, project_name)
    io.directory_creator(project_directory)
    default_config = os.path.join(pathlib.Path(__file__).parent, 'default.config')
    new_config = os.path.join(project_directory, f"{project_name}.config")
    if os.path.exists(new_config):
        pass
    else:
        io.copy_file(default_config, new_config)

    config_utils.edit_cfg_file(new_config, 'management', 'parent_dir', parent_directory)
    config_utils.edit_cfg_file(new_config, 'management', 'project_name', project_name)
    config_utils.edit_cfg_file(new_config, 'management', 'base_epw', project_epw)
    config_utils.edit_cfg_file(new_config, 'management', 'tmy_name', pathlib.Path(project_epw).name)

    tmy_location = general.tmy_location(project_epw)
    config_utils.edit_cfg_file(new_config, 'management', 'latitude', tmy_location['lat'])
    config_utils.edit_cfg_file(new_config, 'management', 'longitude', tmy_location['lon'])
    config_utils.edit_cfg_file(new_config, 'management', 'elevation', tmy_location['elevation'])
    config_utils.edit_cfg_file(new_config, 'management', 'utc', tmy_location['utc'])

    timezone = temporal.get_timezone(tmy_location['lat'], tmy_location['lon'])
    config_utils.edit_cfg_file(new_config, 'management', 'timezone', timezone)

    return new_config


class Project:
    def __init__(self, config_file_path):
        self.config_file_path = config_file_path
        self.config = configparser.ConfigParser()
        self.config.read(config_file_path)
        for section in self.config.sections():
            for k in list(self.config[section].keys()):
                value_in = self.config[section][k]
                formatted_value = config_utils.format_config_item(section, k, value_in)
                self.__setattr__(f"{section}_{k}", formatted_value)
        if self.irradiance_use_accelerad == True:
            self.irradiance_n_workers = 1
        else:
            if self.irradiance_n_workers == 0:
                self.irradiance_n_workers = os.cpu_count() - 1

    def project_setup(self):
        library_root = pathlib.Path(__file__).parent.parent

        self.PROJECT_DIR = os.path.join(self.management_parent_dir, self.management_project_name)

        # inputs
        self.INPUTS_DIR = os.path.join(self.PROJECT_DIR, "inputs")
        io.directory_creator(self.INPUTS_DIR)
        ## host data
        self.HOSTS_DIR = os.path.join(self.INPUTS_DIR, "hosts")
        io.directory_creator(self.HOSTS_DIR)
        self.HOST_DIR = os.path.join(self.HOSTS_DIR, self.management_host_name)
        io.directory_creator(self.HOST_DIR)
        self.SCEN_DIR = os.path.join(self.HOST_DIR, self.management_scenario_name)
        io.directory_creator(self.SCEN_DIR)
        self.GEOMETRY_DIR = os.path.join(self.SCEN_DIR, "geometry")
        io.directory_creator(self.GEOMETRY_DIR)
        self.PANELIZER_DIR = os.path.join(self.SCEN_DIR, "panelizer")
        io.directory_creator(self.PANELIZER_DIR)
        self.STRINGER_DIR = os.path.join(self.SCEN_DIR, "stringer")
        io.directory_creator(self.STRINGER_DIR)

        ### radiance scene
        self.RADIANCE_DIR = os.path.join(self.SCEN_DIR, "radiance")
        io.directory_creator(self.RADIANCE_DIR)

        ### loads data
        self.HOST_LOADS_DIR = os.path.join(self.SCEN_DIR, "loads")
        io.directory_creator(self.HOST_LOADS_DIR)
        self.HOST_STATIONARY_LOADS = os.path.join(self.HOST_LOADS_DIR, "stationary.csv")
        self.HOST_MOBILITY_LOADS = os.path.join(self.HOST_LOADS_DIR, "mobility.csv")


        ## shared data between scenarios, objects
        self.SHARED_DIR = os.path.join(self.INPUTS_DIR, "shared")
        io.directory_creator(self.SHARED_DIR)
        ### log file
        self.LOG_FILE = os.path.join(self.PROJECT_DIR, 'log_file.txt')
        if os.path.exists(self.LOG_FILE):
            pass
        else:
            io.create_log_file(self.LOG_FILE)

        ### tmy data
        self.TMY_DIR = os.path.join(self.SHARED_DIR, 'tmy')
        io.directory_creator(self.TMY_DIR)
        self.TMY_FILE = os.path.join(self.TMY_DIR, f"{self.management_scenario_name}.epw")
        if self.management_base_epw is None:
            default_epw_file = os.path.join(library_root, "template_files", "zurich_2001_2021.epw")
            io.copy_file(default_epw_file, self.TMY_FILE)
            self.management_base_epw = default_epw_file
            self.edit_cfg_file('management', 'base_epw', self.management_base_epw)
        else:
            io.copy_file(self.management_base_epw, self.TMY_FILE)
        self.SUN_UP_FILE = os.path.join(self.TMY_DIR, "sun-up-hours.txt")

        if os.path.exists(self.SUN_UP_FILE):
            pass
        else:
            # print("Universal sun file missing. Creating from project location.")
            self.create_sun_file()
        sun_up, sun_hours = general.create_sun_mask(self.SUN_UP_FILE)
        self.sunup_array = sun_hours['HOY'].values
        self.sundown_array = sun_up[sun_up['Sunny'] == False]['HOY'].values


        ### device data

        self.MODULE_CELL_DIR = os.path.join(self.SHARED_DIR, "cell_module_data")
        io.directory_creator(self.MODULE_CELL_DIR)
        self.module_cell_data = os.path.join(self.MODULE_CELL_DIR, "cactus_typical_devices.csv")
        if os.path.exists(self.module_cell_data):
            pass
        else:
            default_module_data = os.path.join(library_root, "device", "default_devices", "cactus_typical_devices.csv")
            io.copy_file(default_module_data, self.module_cell_data)
        self.cec_data = os.path.join(self.MODULE_CELL_DIR, "cec_database_local.csv")
        if os.path.exists(self.cec_data):
            pass
        else:
            default_module_data = os.path.join(library_root, "device", "default_devices", "cec_database_local.csv")
            io.copy_file(default_module_data, self.cec_data)

        self.MAPS_DIR = os.path.join(self.MODULE_CELL_DIR, "map_files")
        io.directory_creator(self.MAPS_DIR)

        default_maps = os.path.join(library_root, "device", "default_devices", "map_files")
        self.MAP_FILES = glob.glob(os.path.join(default_maps, "*.xls*"))

        for map_file in self.MAP_FILES:
            fname = pathlib.Path(map_file).name
            dest_map = os.path.join(self.MAPS_DIR, fname)
            if os.path.exists(dest_map):
                pass
            else:
                io.copy_file(map_file, dest_map)

        # outputs
        self.OUTPUTS_DIR = os.path.join(self.PROJECT_DIR, "outputs")
        io.directory_creator(self.OUTPUTS_DIR)

        self.HOST_RESULTS = os.path.join(self.OUTPUTS_DIR, self.management_host_name)
        io.directory_creator(self.HOST_RESULTS)
        self.SCEN_RESULTS = os.path.join(self.HOST_RESULTS, self.management_scenario_name)
        io.directory_creator(self.SCEN_RESULTS)
        self.POWER_RESULTS_DIR = os.path.join(self.SCEN_RESULTS, "power")
        io.directory_creator(self.POWER_RESULTS_DIR)
        self.LCA_RESULTS_DIR = os.path.join(self.SCEN_RESULTS, "lcia")
        io.directory_creator(self.LCA_RESULTS_DIR)
        self.LCCA_RESULTS_DIR = os.path.join(self.SCEN_RESULTS, "lcca")
        io.directory_creator(self.LCCA_RESULTS_DIR)
        self.DEVICES_RESULTS_DIR = os.path.join(self.SCEN_RESULTS, "device")
        io.directory_creator(self.DEVICES_RESULTS_DIR)
        self.IRRADIANCE_RESULTS_DIR = os.path.join(self.SCEN_RESULTS, "irradiance")
        io.directory_creator(self.IRRADIANCE_RESULTS_DIR)

        # self.CUMULATIVE_RESULTS_DIR = os.path.join(self.HOSTS_DIR, "cumulative_results")
        # general.directory_creator(self.CUMULATIVE_RESULTS_DIR)
        # self.CUMULATIVE_RESULTS_DENSE_DIR = os.path.join(self.HOSTS_DIR, "cumulative_condensed")
        # general.directory_creator(self.CUMULATIVE_RESULTS_DENSE_DIR)

        self.COLD_DIR = os.path.join(self.HOST_DIR, "compressed")

        io.directory_creator(self.COLD_DIR)
        if self.management_scenario_name == None:
            scenario_name = 'base'
        else:
            scenario_name = self.management_scenario_name

        # self.RESULTS_DIR = os.path.join(self.HOST_DIR, "scenario_results", scenario_name)
        # general.directory_creator(self.RESULTS_DIR)
        # self.ANNUAL_RESULT_DIR = os.path.join(self.RESULTS_DIR, "annual")
        # general.directory_creator(self.ANNUAL_RESULT_DIR)
        # self.TIMESERIES_RESULT_DIR = os.path.join(self.RESULTS_DIR, "timeseries")
        # general.directory_creator(self.TIMESERIES_RESULT_DIR)

        # template data
        ## geometry writers
        for file_ext in ["3dm", "gh"]:
            src = os.path.join(library_root, "template_files", f"0_cactus_geometry_template.{file_ext}")
            dst = os.path.join(self.GEOMETRY_DIR, f"0_cactus_geometry_template.{file_ext}")
            io.copy_file(src, dst)

        # ## radiance writer
        # src = os.path.join(library_root, "template_files", f"0_workbench_radiance_writer_template.gh")
        # dst = os.path.join(self.GEOMETRY_DIR, f"0_workbench_radiance_writer_template.gh")
        # general.copy_file(src, dst)

        ## panelizer
        src = os.path.join(library_root, "template_files", f"1_cactus_panelizer_template.gh")
        dst = os.path.join(self.GEOMETRY_DIR, f"1_cactus_panelizer_template.gh")
        io.copy_file(src, dst)

        ## radiance files
        self.skyglow_template = os.path.join(library_root, "template_files", "skyglow.rad")

        print(f"The project is initialized. We have created a base host object named '{self.management_host_name}'.\n"
              "You will need to either move or create the geometry and panelizer files into the appropriate directories.\n"
              "The geometry files should follow the convention defined in the output of the template grasshopper and rhino files.\n"
              "The panelizer files are those that have been created using the grasshopper utility. The Panelizer is\n"
              "not ready for a pure python implementation as of yet.")
    def get_irradiance_results(self):

        active_surface = f"surface_{self.analysis_active_surface}"
        io.directory_creator(os.path.join(self.IRRADIANCE_RESULTS_DIR, active_surface))
        self.DIRECT_IRRAD_FILE = os.path.join(self.IRRADIANCE_RESULTS_DIR, active_surface, "direct.lz4")
        self.DIFFUSE_IRRAD_FILE = os.path.join(self.IRRADIANCE_RESULTS_DIR, active_surface, "diffuse.lz4")

    def log(self, runtime, simulation_type, front_cover=None):
        not_relevant = 'NA'
        if front_cover is None:
            front_cover = not_relevant
        if simulation_type=='irradiance':
            n_negatives = 0
            rad_par = self.irradiance_radiance_param_rcontrib + " " + self.irradiance_radiance_param_rflux
            n_workers = self.irradiance_n_workers
            rad_surface_dir = os.path.join(self.RADIANCE_DIR, f"surface_{self.analysis_active_surface}")
            grid_file = glob.glob(os.path.join(rad_surface_dir, "model", "grid", "*.pts"))[0]
            n_points = int(grid_file.split("_")[-1].split("s")[0])
        else:
            n_negatives = not_relevant
            rad_par = not_relevant
            n_workers = self.analysis_n_workers
            n_points = not_relevant
        entry = f"{temporal.current_time()},{self.management_scenario_name},{self.analysis_device_id}," \
                f"{front_cover},{self.management_host_name}_{self.analysis_active_surface},{simulation_type}," \
                f"{n_negatives},{n_workers},{n_points},{rad_par},{runtime}\n"
        with open(self.LOG_FILE, "a") as fp:
            fp.write(entry)

    def update_cfg(self):
        self.config = configparser.ConfigParser()
        self.config.read(self.config_file_path)
        for section in self.config.sections():
            for k in list(self.config[section].keys()):
                value_in = self.config[section][k]
                formatted_value = config_utils.format_config_item(section, k, value_in)
                self.__setattr__(f"{section}_{k}", formatted_value)

    def edit_cfg_file(self, section, key, new_value):
        self.config.set(section, key, str(new_value))
        with open(self.config_file_path, 'w') as configfile:
            self.config.write(configfile)
        self.update_cfg()

    def create_sun_file(self):
        solpos = pvlib.solarposition.get_solarposition(
            pd.date_range(start=f"01-01-2022 00:30", end=f"12-31-2022 23:30", freq="h", tz=self.management_timezone),
            self.management_latitude,
            self.management_longitude,
            self.management_elevation).reset_index()

        solpos = [f"{h}\n" for h in list(solpos[solpos['elevation'] > 0].index.values + 0.5)]
        with open(self.SUN_UP_FILE, 'w') as fp:
            fp.writelines(solpos)

    def find_sun_file(self):
        srf_dir = glob.glob(os.path.join(self.RADIANCE_DIR, "surface_*"))[0]
        results_dir = glob.glob(os.path.join(srf_dir, "*results*"))[0]
        if "scenario" in results_dir.split(os.sep)[-1]:
            scen_dirs = glob.glob(os.path.join(results_dir, "*"))
            scen_dir_one = scen_dirs[0]
            src_sun_file = os.path.join(scen_dir_one, "results", "annual_irradiance", "results", "total",
                                        "sun-up-hours.txt")
            os.makedirs(os.path.dirname(self.SUN_UP_FILE), exist_ok=False)
            shutil.copy2(src_sun_file, self.SUN_UP_FILE)
        else:
            src_sun_file = os.path.join(results_dir, "annual_irradiance", "results", "total",
                                        "sun-up-hours.txt")
            os.makedirs(os.path.dirname(self.SUN_UP_FILE), exist_ok=False)
            shutil.copy2(src_sun_file, self.SUN_UP_FILE)
