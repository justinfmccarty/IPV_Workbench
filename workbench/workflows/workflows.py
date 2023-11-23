import time
import os
import shutil

import numpy as np

from workbench.manage import results_writers
from workbench.simulations import method_2phase
from workbench.utilities import general


def run_irradiance(host, overwrite=True):
    # Before initializing a workflow always update the config file
    # in case a manual edit was made
    host.project.update_cfg()

    # get surfaces for loop
    surfaces = host.get_surfaces()

    # start loop
    for surface in surfaces:
        # clean curly brackets
        surface = general.clean_grasshopper_key(surface)
        # change surface
        host.project.edit_cfg_file('analysis', 'active_surface', surface)
        # set the irradiance results to an active variable in the project manager
        host.project.get_irradiance_results()

        # check to see if existing files exist and should be overwritten
        if overwrite==True:
            # no need to check for the files because they will be overwritten anyway
            pass
        else:
            # if the results files exist then skip this surface in the loop
            if (os.path.exists(host.project.DIFFUSE_IRRAD_FILE)) & (os.path.exists(host.project.DIRECT_IRRAD_FILE)):
                print(f"Existing results files detected and the 'overwrite' arg is set to False, skipping surface {surface}.")
                continue
            else:
                pass

        print(f"Starting Radiance workflow for surface {surface}.")
        start_time = time.time()
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


def run_module_point(host_object, point_resolution):
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
    host_object.project.update_cfg()

    # get surfaces for loop
    surfaces = host_object.get_surfaces()

    # start loop
    for surface in surfaces:
        # clean curly brackets
        surface_c = general.clean_grasshopper_key(surface)
        print(f"Starting module {point_resolution} analysis workflow for surface {surface_c}.")
        start_time = time.time()
        # change surface
        host_object.project.edit_cfg_file('analysis', 'active_surface', surface_c)
        # run method
        if point_resolution == 'center_point':
            print(surface)
            host_object.solve_module_center_pts(surface)
        elif point_resolution == 'cell_point':
            host_object.solve_module_cell_pts(surface)
        total_time = round(time.time() - start_time, 2)
        print(f"Completed in {total_time} seconds.")
        print("-----------------------")

    print(f"Cumulating and saving results for host object.")
    start_time = time.time()
    results_writers.write_building_results_simple_timeseries(host_object, point_resolution)
    total_time = round(time.time() - start_time, 2)
    print(f"Completed in {total_time} seconds.")


def run_module_iv_solver(host_object):
    # Before initializing a workflow always update the config file
    # in case a manual edit was made
    host_object.project.update_cfg()

    # get surfaces for loop
    surfaces = host_object.get_surfaces()

    # start loop
    for surface in surfaces:
        # clean curly brackets
        surface_c = general.clean_grasshopper_key(surface)

        pv_cells_xyz_arr = np.array(host_object.get_cells_xyz(surface, module_name))

        G_dir_ann = ipv_irrad.collect_raw_irradiance(pv_cells_xyz_arr,
                                                     sensor_pts_xyz_arr,
                                                     direct_ill)  # .values)
        G_diff_ann = ipv_irrad.collect_raw_irradiance(pv_cells_xyz_arr,
                                                      sensor_pts_xyz_arr,
                                                      diffuse_ill)  # .values)