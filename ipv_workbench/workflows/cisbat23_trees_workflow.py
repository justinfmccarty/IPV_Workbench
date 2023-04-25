import time
import os
import numpy as np
import pandas as pd
from ipv_workbench.utilities import utils
from ipv_workbench.translators import panelizer
from ipv_workbench.translators import results_writers as ipv_results
from ipv_workbench.devices import devices


def format_raw_filename(cell_technology, orientation, front_cover, object_name):
    return f"{cell_technology}{orientation}_{front_cover}_{object_name}_raw.pickle"


def run_configuration(project_folder, surface, building, direction, cell_technology, orientation, context,
                      front_cover='solar_glass', year=2020, resolution=1):
    scenario = f"{cell_technology}{orientation}_{front_cover}_{year}"
    config_string = f"{direction}-{context},{cell_technology},{orientation},{front_cover},{building}"
    print(f"    Starting {config_string}")
    raw_panelizer_file = format_raw_filename(cell_technology, orientation, front_cover, building)

    all_topologies = ['micro_inverter', 'string_inverter', 'central_inverter']
    log_file = os.path.join(project_folder, 'shared', 'resources', 'log_file.txt')

    panelizer_object = panelizer.PanelizedObject(project_folder,
                                                 building,
                                                 raw_panelizer_file,
                                                 exclude_surfaces=["{8733;0;0}"],
                                                 project_data=r"/Users/jmccarty/Nextcloud/Projects/12_CISBAT23_trees/panelizer_models/shared_data",
                                                 contextual_scenario=context)
    custom_device_data = pd.read_csv(panelizer_object.module_cell_data,
                                     index_col='scenario').loc[f"{cell_technology}{orientation}"].to_dict()

    panelizer_object.cell = devices.Cell(custom_device_data)
    panelizer_object.analysis_location = 'zurich'
    panelizer_object.analysis_year = year
    panelizer_object.set_tmy_data()

    # setting the analysis_period
    panelizer_object.hourly_resolution = resolution
    panelizer_object.set_analysis_period()

    # skip results if they exist DEBUGGING ONLY
    result_file_building = os.path.join(panelizer_object.RESULTS_DIR, 'timeseries',
                                        f"{scenario}_central_inverter_building_level_results_hourly.csv")
    if os.path.exists(result_file_building):
        print("Result exists, skipping iteration")
        # log_string = f"{config_string},0\n"
        # utils.log_run(log_file, log_string)
        return None

    # run the major simulation
    print("     Starting module solver...")
    solver_start = time.time()
    panelizer.solve_object_module_iv(panelizer_object, display_print=False, mp=True)
    print(f"     Module IV solver finished in {round(time.time() - solver_start, 1)} seconds")

    # save the data
    # transfer necessary data between levels
    panelizer_object.transfer_initial()

    # solve and write results of stringing
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

    # write IV Curve results
    for topology in all_topologies:
        # write results
        ipv_results.write_building_results_timeseries(panelizer_object, scenario, topology)

    end_time = time.time()
    run_time = end_time - solver_start
    log_string = f"{config_string},{np.round(run_time, 3)}\n"
    utils.log_run(log_file, log_string)
    print(f"    {building}, {scenario} completed in {round(run_time, 1)} seconds.")


def main():
    building = 'B8733'
    surface = '{8733;0;8}'

    all_directions = ['east', 'south', 'west']
    all_cell_tech = ['A', 'B', 'C', 'D']
    all_orientations = ['P', 'L']
    all_context = ['all', 'close', 'near', 'none']

    for direction in all_directions:
        project_folder = os.path.join(r"/Users/jmccarty/Nextcloud/Projects/12_CISBAT23_trees/panelizer_models",
                                      direction)
        for cell_tech in all_cell_tech:
            for orientation in all_orientations:
                for context in all_context:
                    run_configuration(project_folder, surface, building, direction, cell_tech, orientation,
                                      context, front_cover='solar_glass', year=2020, resolution=1)

if __name__ == "__main__":
    main()