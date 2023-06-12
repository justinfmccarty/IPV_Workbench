from pvlib import pvsystem, singlediode
import numpy as np
import multiprocessing as mp
from tqdm import notebook
from workbench.host import module_mapping as ipv_mm
from workbench.old_solver import calculations
from workbench.host import host
from workbench.utilities import circuits, general, temporal
import time
import copy
import pandas as pd



def main(panelizer_object, string, module_dict, pv_cells_xyz_arr, tmy_location, dbt, psl, grid_pts, direct_ill, diffuse_ill, base_parameters,
         custom_module_data, default_submodule_map, default_diode_map, default_subcell_map, cell_type):

    ncpu = panelizer_object.ncpu

    timeseries_chunks = np.array_split(panelizer_object.all_hoy, ncpu)
    if len(timeseries_chunks[0]) < 150:
        ncpu = 4
        timeseries_chunks = np.array_split(panelizer_object.all_hoy, ncpu)
        # this fix is brokedn due to the output of the function not agreegin with the expected output from the MP function
        # print("Length of one timeseries chunk is below hard coded threshold of 150. "
        #       "Switching to a single process.")
        # return panelizer.compile_system_single_core(module_dict, panelizer_object.all_hoy,
        #                                            tmy_location, dbt, psl,
        #                                            pv_cells_xyz_arr, grid_pts,
        #                                            direct_ill, diffuse_ill,
        #                                            base_parameters,
        #                                            custom_module_data,
        #                                            default_submodule_map,
        #                                            default_diode_map,
        #                                            default_subcell_map,
        #                                            cell_type)
    direct_ill_chunks = np.array_split(direct_ill, ncpu)
    diffuse_ill_chunks = np.array_split(diffuse_ill, ncpu)
    dbt_chunks = np.array_split(dbt, ncpu)
    psl_chunks = np.array_split(psl, ncpu)

    with mp.Pool(processes=ncpu) as pool:
        # print("    Pool Opened")

        args = list(zip([module_dict] * ncpu,
                        timeseries_chunks,
                        [tmy_location] * ncpu,
                        dbt_chunks,
                        psl_chunks,
                        [pv_cells_xyz_arr] * ncpu,
                        [grid_pts] * ncpu,
                        direct_ill_chunks,
                        diffuse_ill_chunks,
                        [base_parameters] * ncpu,
                        [custom_module_data] * ncpu,
                        [default_submodule_map] * ncpu,
                        [default_diode_map] * ncpu,
                        [default_subcell_map] * ncpu,
                        [cell_type] * ncpu))
        # module_dict, surface, string, module, cell_area, cell_params, hoy_chunk
        mp_results = pool.starmap(panelizer.compile_system_single_core, args)
        # print("    Result Gathered")
        temporal.sleep(0.5)
        pool.close()
        # print("    Pool closed")
        pool.join()
        # print("    Pool joined")

    # module_i_dict = {}
    # module_v_dict = {}
    # module_g_dict = {}
    #
    # for r in mp_results:
    #     r[0]


    module_i_dict = {}
    module_v_dict = {}
    module_g_dict = {}
    module_param_dict = []

    for r_ in mp_results:
        module_i_dict.update(r_[0])
        module_v_dict.update(r_[1])
        module_g_dict.update(r_[2])
        module_param_dict.append(r_[3])


    return module_i_dict, module_v_dict, module_g_dict, module_param_dict

if __name__=="__main__":
    main(panelizer_object, string, module_dict, pv_cells_xyz_arr, tmy_location, dbt, psl, grid_pts, direct_ill, diffuse_ill, base_parameters,
         custom_module_data, default_submodule_map, default_diode_map, default_subcell_map, cell_type)