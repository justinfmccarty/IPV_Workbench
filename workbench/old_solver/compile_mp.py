import numpy as np
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager

import workbench.utilities.io
from workbench.host import module_mapping as ipv_mm
from workbench.utilities import general
from workbench.irradiance import method_effective_irradiance as ipv_irrad
import pandas as pd


def main(panelizer_object, surface, string, tmy_location, dbt, psl, grid_pts, direct_ill, diffuse_ill):
    timeseries = panelizer_object.all_hoy
    ncpu = panelizer_object.ncpu
    modules = panelizer_object.get_modules(surface, string)

    if len(modules) < ncpu:
        ncpu = len(modules)
    module_dict_list = [panelizer_object.get_dict_instance([surface, string, module_name]) for module_name in modules]
    # pv_cells_xyz_arr_list = [np.array(panelizer_object.get_cells_xyz(surface, string, module_name)) for module_name in modules]
    # pv_cells_xyz_arr_chunks = []

    module_dict_chunks = np.array_split(module_dict_list, ncpu)
    module_name_chunks = np.array_split(modules, ncpu)
    pv_cells_xyz_arr_chunks = []
    for module_name_chunk in module_name_chunks:
        pv_cells_xyz_arr_chunks.append(
            [panelizer_object.get_cells_xyz(surface, string, module_name) for module_name in module_name_chunk])

    string_dict = panelizer_object.get_dict_instance([surface, string])
    string_details = string_dict['DETAILS']
    base_parameters = utils.get_cec_data(string_details['cec_key'], file_path=panelizer_object.cec_data)
    custom_module_data = pd.read_csv(panelizer_object.module_cell_data, index_col='scenario').loc[
        string_details['module_type']].to_dict()

    module_template = string_dict['DETAILS']['module_type']
    cell_type = ipv_mm.get_cell_type(module_template[0])
    orientation = ipv_mm.get_orientation(module_template[1])
    map_file = [fp for fp in panelizer_object.map_files if f"{cell_type}_{orientation}" in fp][0]
    default_submodule_map, default_diode_map, default_subcell_map = workbench.utilities.io.read_map_excel(map_file)

    with mp.Pool(processes=ncpu) as pool:
        # print("    Pool Opened")

        args = list(zip(module_dict_chunks,
                        module_name_chunks,
                        [timeseries] * ncpu,
                        [tmy_location] * ncpu,
                        [dbt] * ncpu,
                        [psl] * ncpu,
                        pv_cells_xyz_arr_chunks,
                        [grid_pts] * ncpu,
                        [direct_ill] * ncpu,
                        [diffuse_ill] * ncpu,
                        [base_parameters] * ncpu,
                        [custom_module_data] * ncpu,
                        [default_submodule_map] * ncpu,
                        [default_diode_map] * ncpu,
                        [default_subcell_map] * ncpu,
                        [cell_type] * ncpu))
        # module_dict, surface, string, module, cell_area, cell_params, hoy_chunk

        mp_results = pool.starmap(panelizer.compile_system_multi_core, args)
        # print("    Result Gathered")
        # time.sleep(1)
        pool.close()
        # print("    Pool closed")
        pool.join()
        # print("    Pool joined")
    utils.unpack_mp_results(mp_results, panelizer_object, surface, string, modules, timeseries)


def main_v2(panelizer_object, surface, string, dbt, grid_pts, direct_ill, diffuse_ill):
    timeseries = panelizer_object.all_hoy
    ncpu = panelizer_object.ncpu
    modules = panelizer_object.get_modules(surface, string)
    sensor_pts_xyz_arr = grid_pts[['X', 'Y', 'Z']].values

    if len(modules) < ncpu:
        ncpu = len(modules)
    module_dict_list = [panelizer_object.get_dict_instance([surface, string, module_name]) for module_name in modules]

    # pv_cells_xyz_arr_list = [np.array(panelizer_object.get_cells_xyz(surface, string, module_name)) for module_name in modules]
    # pv_cells_xyz_arr_chunks = []

    module_dict_chunks = np.array_split(module_dict_list, ncpu)
    module_name_chunks = np.array_split(modules, ncpu)
    g_eff_ann_chunks = []
    print(f"    {string}-starting effective irradiance chunking")
    for module_name_chunk in module_name_chunks:
        # print(f"   {module_name_chunk}/{len(module_name_chunks)}")
        g_eff_chunk = []
        for module_name in module_name_chunk:
            g_eff_ann = ipv_irrad.get_effective_module_irradiance(panelizer_object, surface, string, module_name,
                                                                  sensor_pts_xyz_arr, direct_ill, diffuse_ill)
            g_eff_chunk.append(g_eff_ann)

        g_eff_ann_chunks.append(g_eff_chunk)
    print("    Completed effective irradiance chunking")

    string_dict = panelizer_object.get_dict_instance([surface, string])
    string_details = string_dict['DETAILS']
    base_parameters = utils.get_cec_data(string_details['cec_key'], file_path=panelizer_object.cec_data)
    custom_module_data = pd.read_csv(panelizer_object.module_cell_data, index_col='scenario').loc[
        string_details['module_type']].to_dict()

    module_template = string_dict['DETAILS']['module_type']
    cell_type = ipv_mm.get_cell_type(module_template[0])
    orientation = ipv_mm.get_orientation(module_template[1])
    map_file = [fp for fp in panelizer_object.map_files if f"{cell_type}_{orientation}" in fp][0]
    default_submodule_map, default_diode_map, default_subcell_map = workbench.utilities.io.read_map_excel(map_file)

    print("    Opening pools")
    with mp.Pool(processes=ncpu) as pool:
        # print("    Pool Opened")

        args = list(zip(module_dict_chunks,
                        module_name_chunks,
                        [timeseries] * ncpu,
                        g_eff_ann_chunks,
                        [dbt] * ncpu,
                        [base_parameters] * ncpu,
                        [custom_module_data] * ncpu,
                        [default_submodule_map] * ncpu,
                        [default_diode_map] * ncpu,
                        [default_subcell_map] * ncpu,
                        [cell_type] * ncpu))
        # module_dict, surface, string, module, cell_area, cell_params, hoy_chunk

        mp_results = pool.starmap(panelizer.compile_system_multi_core_v2, args)
        # print("    Result Gathered")
        # time.sleep(1)
        pool.close()
        # print("    Pool closed")
        pool.join()
        # print("    Pool joined")
    utils.unpack_mp_results(mp_results, panelizer_object, surface, string, modules, timeseries)
    print("    Pools closed. Results unpacked")


def main_v3(panelizer_object, surface, string, tmy_location, dbt, psl, grid_pts, direct_ill, diffuse_ill):
    timeseries = panelizer_object.all_hoy
    ncpu = panelizer_object.ncpu
    modules = panelizer_object.get_modules(surface, string)

    if len(modules) < ncpu:
        ncpu = len(modules)
    module_dict_list = [panelizer_object.get_dict_instance([surface, string, module_name]) for module_name in modules]
    # pv_cells_xyz_arr_list = [np.array(panelizer_object.get_cells_xyz(surface, string, module_name)) for module_name in modules]
    # pv_cells_xyz_arr_chunks = []

    module_dict_chunks = np.array_split(module_dict_list, ncpu)
    module_name_chunks = np.array_split(modules, ncpu)
    pv_cells_xyz_arr_chunks = []
    for module_name_chunk in module_name_chunks:
        pv_cells_xyz_arr_chunks.append(
            [panelizer_object.get_cells_xyz(surface, string, module_name) for module_name in module_name_chunk])

    string_dict = panelizer_object.get_dict_instance([surface, string])
    string_details = string_dict['DETAILS']
    base_parameters = utils.get_cec_data(string_details['cec_key'], file_path=panelizer_object.cec_data)
    custom_module_data = pd.read_csv(panelizer_object.module_cell_data, index_col='scenario').loc[
        string_details['module_type']].to_dict()

    module_template = string_dict['DETAILS']['module_type']
    cell_type = ipv_mm.get_cell_type(module_template[0])
    orientation = ipv_mm.get_orientation(module_template[1])
    map_file = [fp for fp in panelizer_object.map_files if f"{cell_type}_{orientation}" in fp][0]
    default_submodule_map, default_diode_map, default_subcell_map = workbench.utilities.io.read_map_excel(map_file)

    # name the memory
    shared_names = [f"{string}_shm_direct", f"{string}_shm_diffuse"] #f"{string}_shm_grid",
    shared_arrays = [direct_ill, diffuse_ill] #grid_pts.to_numpy(),
    shared_shapes = [direct_ill.shape, diffuse_ill.shape]
    shared_memory_tups = list(zip(shared_arrays, shared_names, shared_shapes))

    # create the shared memory objects
    for shared_tup in shared_memory_tups:
        utils.create_read_only_arrays(shared_tup[0], shared_tup[1])
        # utils.create_shared_memory_nparray(shared_tup[0], shared_tup[1])

    shared_direct_arrays = [utils.access_array(shared_names[0], shared_shapes[0], shared_arrays[0].dtype) for n in
                            range(0, ncpu)]
    shared_diffuse_arrays = [utils.access_array(shared_names[1], shared_shapes[1], shared_arrays[1].dtype) for n in
                            range(0, ncpu)]



    # initiate the pool
    with mp.Pool(processes=ncpu) as pool, SharedMemoryManager() as smm:
        shared_mem_direct = smm.SharedMemory(size=direct_ill.nbytes)
        direct_ill_smm_arr = utils.create_np_array_from_shared_mem(shared_mem_direct, direct_ill.dtype, direct_ill.shape)
        direct_ill_smm_arr[:] = direct_ill  # load the data into shared memory

        shared_mem_diffuse = smm.SharedMemory(size=diffuse_ill.nbytes)
        diffuse_ill_smm_arr = utils.create_np_array_from_shared_mem(shared_mem_diffuse, diffuse_ill.dtype, diffuse_ill.shape)
        diffuse_ill_smm_arr[:] = diffuse_ill  # load the data into shared memory

        shared_names = [shared_mem_direct.name, shared_mem_diffuse.name]

        args = list(zip(module_dict_chunks,
                        module_name_chunks,
                        [timeseries] * ncpu,
                        [tmy_location] * ncpu,
                        [dbt] * ncpu,
                        [psl] * ncpu,
                        pv_cells_xyz_arr_chunks,
                        [grid_pts] * ncpu, # grid pts
                        (shared_mem_direct, direct_ill.dtype, direct_ill.shape), # shared_direct_arrays, # [shared_names[0]] * ncpu, # direct
                        (shared_mem_diffuse, diffuse_ill.dtype, diffuse_ill.shape), # shared_diffuse_arrays, # [shared_names[1]] * ncpu, # diffuse
                        [base_parameters] * ncpu,
                        [custom_module_data] * ncpu,
                        [default_submodule_map] * ncpu,
                        [default_diode_map] * ncpu,
                        [default_subcell_map] * ncpu,
                        [cell_type] * ncpu))
        # module_dict, surface, string, module, cell_area, cell_params, hoy_chunk

        mp_results = pool.starmap(panelizer.compile_system_multi_core, args)
        # print("    Result Gathered")
        # time.sleep(1)
        pool.close()
        # print("    Pool closed")
        pool.join()
        # print("    Pool joined")

        del direct_ill_smm_arr
        del diffuse_ill_smm_arr

        # # close the shared memory
        for shared_name in shared_names:
            utils.release_shared(shared_name)

    # unpack the pooled results
    utils.unpack_mp_results(mp_results, panelizer_object, surface, string, modules, timeseries)



#
# if __name__=="__main__":
#     main(panelizer_object, surface, string, tmy_location, dbt, psl, grid_pts, direct_ill, diffuse_ill)
