import copy
import os
import time
import pandas as pd
from ipv_workbench.devices import devices
from ipv_workbench.translators import panelizer
from ipv_workbench.utilities import utils
from ipv_workbench.translators import results_writers as ipv_results
import numpy as np
from tqdm import tqdm


def run_building(project_folder, cell_technology, orientation, front_cover, building, year, scenario, log_file,
                 all_topologies, hourly_resolution):
    start_time = time.time()
    # set up the simulation
    raw_panelizer_file = f"{cell_technology}{orientation}_{front_cover}_{building}_raw.pickle"

    panelizer_object = panelizer.PanelizedObject(project_folder, building, raw_panelizer_file)
    panelizer_object.analysis_location = 'zurich'
    panelizer_object.analysis_year = year
    panelizer_object.set_tmy_data()

    # setting the analysis_period
    panelizer_object.hourly_resolution = hourly_resolution
    panelizer_object.set_analysis_period()



    custom_device_data = pd.read_csv(panelizer_object.module_cell_data, index_col='scenario').loc[
        f"{cell_technology}{orientation}"].to_dict()
    panelizer_object.cell = devices.Cell(custom_device_data)

    # skip results if they exist DEBUGGING ONLY
    result_file_building = os.path.join(panelizer_object.RESULTS_DIR, 'timeseries',
                                        f"{scenario}_central_inverter_building_level_results_hourly.csv")
    if os.path.exists(result_file_building):
        print("Result exists, skipping iteration")
        return None

    # run the major simulation
    print("     Starting module solver...")
    solver_start = time.time()
    panelizer.solve_object_module_iv(panelizer_object, mp=True)
    print(f"     Module solver finished in {round(time.time() - solver_start, 1)} seconds")

    # transfer necessary data between levels
    panelizer_object.transfer_initial()

    # solve and write results of stringing
    for surface in panelizer_object.get_surfaces():
        for topology in panelizer_object.simulation_suite_topologies:
            panelizer_object.topology = topology
            if topology == 'micro_inverter':
                for string in panelizer_object.get_strings(surface):
                    panelizer_object.write_first_level_results(surface, string)
            else:
                panelizer_object.write_first_level_results(surface)

    for topology in panelizer_object.simulation_suite_topologies:
        # print(f"    Starting {topology}")
        panelizer_object.topology = topology
        for surface in panelizer_object.get_surfaces():

            if panelizer_object.topology == 'micro_inverter':
                for string in panelizer_object.get_strings(surface):
                    panelizer_object.write_up_string_results(surface, string)

            panelizer_object.write_up_surface_results(surface)

        panelizer_object.write_up_object_results()

    panelizer_object.write_key_parameters()

    for topology in all_topologies:
        # write results
        ipv_results.write_building_results_timeseries(panelizer_object, scenario, topology)
        # building_results_files[topology].append(building_results_file)

        # pmp_results.update(
        #     {topology: np.sum(np.fromiter(panelizer_object.get_dict_instance([])['YIELD'][topology]['pmp'].values(), dtype=float))})
        # irrad_results.update(
        #     {topology: np.sum(np.fromiter(panelizer_object.get_dict_instance([])['YIELD'][topology]['irrad'].values(), dtype=float))})
    end_time = time.time()
    run_time = end_time - start_time
    log_string = f"{year},{cell_technology},{orientation},{front_cover},{building},{np.round(run_time, 3)}\n"
    utils.log_run(log_file, log_string)
    print(f"    {building}, {scenario} completed in {round(run_time,1)} seconds.")
    pickel_start = time.time()
    print("     Storing object")
    utils.write_pickle(panelizer_object,
                       os.path.join(panelizer_object.COLD_DIR,f"{scenario}_{building}.xz"),
                       compress=True)
    pickle_end = time.time() - pickel_start
    print(f"        {round(pickle_end,1)} seconds to compress pickle")
    print(f"    Completed {building} {scenario}")
    return panelizer_object.get_dict_instance([])['DETAILS']


def main():
    project_folder = r"C:\Users\Justin\Desktop\bano_project_folder"
    year_list = [2020, 2050, 2080]
    building_list = ["B1391", "B1389", "B1390", "B1360", "B1392", "B1393", "B1394", "B2494"]
    all_topologies = ['micro_inverter', 'string_inverter', 'central_inverter']
    log_file = os.path.join(project_folder, 'shared', 'resources', 'log_file.txt')
    hourly_resolution = 2  #run every N hours (interpolate between the results at the very end)
    for year in year_list:
        for front_cover in ["solar_glass", "light_grey", "basic_white"]:
            for orientation in ["P", "L"]:
                for cell_technology in ["A", "B", "C", "D", "E"]:

                    # this is key (setting the scenario)
                    scenario = f"{cell_technology}{orientation}_{front_cover}_{year}"
                    print(r"-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-")

                    object_detail_dicts = []

                    for building in building_list:
                        print(f"Starting {scenario}, {building}")
                        object_details = run_building(project_folder, cell_technology, orientation, front_cover,
                                                      building, year, scenario, log_file, all_topologies, hourly_resolution)
                        object_detail_dicts.append(object_details)

                    for topology in all_topologies:
                        building_results_files = []
                        for building in building_list:
                            building_result_file = os.path.join(project_folder, 'objects', building, 'results',
                                                                'timeseries',
                                                                f"{scenario}_{topology}_building_level_results_hourly.csv")
                            building_results_files.append(building_result_file)
                        bldg_results = [pd.read_csv(fp, index_col="index") for fp in building_results_files]
                        cumulative_df = ipv_results.write_cumulative_scenario_results(project_folder,
                                                                                      scenario,
                                                                                      topology,
                                                                                      bldg_results)

                        ipv_results.write_condensed_result(project_folder,
                                                           object_detail_dicts,
                                                           cumulative_df,
                                                           scenario,
                                                           topology)


if __name__ == "__main__":
    main()
