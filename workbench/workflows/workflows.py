import time
import os
import shutil

from workbench.manage import results_writers
from workbench.simulations import method_2phase
from workbench.utilities import general


def run_irradiance(host):
    # Before initializing a workflow always update the config file
    # in case a manual edit was made
    host.project.update_cfg()

    # get surfaces for loop
    surfaces = host.get_surfaces()

    # start loop
    for surface in surfaces:
        # clean curly brackets
        surface = general.clean_grasshopper_key(surface)
        print(f"Starting Radiance workflow for surface {surface}.")
        start_time = time.time()
        # change surface
        host.project.edit_cfg_file('analysis', 'active_surface', surface)

        # run 2 phase
        method_2phase.run_2phase_dds(host.project)
        total_time = round(time.time() - start_time, 2)
        print(f"Completed in {total_time} seconds.")

        start_time = time.time()
        print(f"Converting .ill files to feather for surface {surface}.")
        # save compressed files
        method_2phase.save_irradiance_results(host.project)
        total_time = round(time.time() - start_time, 2)
        print(f"Completed in {total_time} seconds.")

        # delete raw output files if specified in config
        if bool(host.project.irradiance_store_radiance) == True:
            pass
        else:
            print(f"Deleting intermediate Radiance simulation data for {surface}.\n"
                  f"The input model will be kept.")
            radiance_project_dir = host.project.RADIANCE_DIR
            rad_surface_dir = os.path.join(radiance_project_dir, f"surface_{surface}")
            output_dir = os.path.join(rad_surface_dir, "outputs")
            shutil.rmtree(output_dir)

        print("-----------------------")


def run_module_point(host, point_resolution):
    if point_resolution == 'center_point':
        pass
    elif point_resolution == 'cell_point':
        pass
    else:
        print(
            "The arg 'point_resolution must be specific as either 'cell_point' or 'center_point'. Defaulting to 'center_point'.")
        point_resolution = 'center_point'

    # Before initializing a workflow always update the config file
    # in case a manual edit was made
    host.project.update_cfg()

    # get surfaces for loop
    surfaces = host.get_surfaces()

    # start loop
    for surface in surfaces:
        # clean curly brackets
        surface_c = general.clean_grasshopper_key(surface)
        print(f"Starting module {point_resolution} analysis workflow for surface {surface_c}.")
        start_time = time.time()
        # change surface
        host.project.edit_cfg_file('analysis', 'active_surface', surface_c)
        # run method
        if point_resolution == 'center_point':
            print(surface)
            host.solve_module_center_pts(surface)
        elif point_resolution == 'cell_point':
            host.solve_module_cell_pts(surface)
        total_time = round(time.time() - start_time, 2)
        print(f"Completed in {total_time} seconds.")
        print("-----------------------")

    print(f"Cumulating and saving results for host object.")
    start_time = time.time()
    results_writers.write_building_results_simple_timeseries(host, point_resolution)
    total_time = round(time.time() - start_time, 2)
    print(f"Completed in {total_time} seconds.")
