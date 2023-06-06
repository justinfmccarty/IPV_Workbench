from pvlib import pvsystem, singlediode
import numpy as np
import multiprocess as mp
from tqdm import notebook
from workbench.host import module_mapping as ipv_mm
from workbench.solver import calculations
from workbench.host import host
from workbench.utilities import circuits, utils, time_utils
import time
import copy
import pandas as pd









# def archive_mp_module_simulation(module_dict_chunk, module_name_chunk, cell_area, cell_params, timeseries):
#     module_results = {}
#
#     for n, module_dict in enumerate(module_dict_chunk):
#         module = module_name_chunk[n]
#         Imod, Vmod, Gmod = timeseries_module_simulation(module_dict, cell_area, cell_params, timeseries)
#         module_results.update({module: [Imod, Vmod, Gmod]})
#
#     return module_results
#
# def archive_timeseries_module_simulation(module_dict, cell_area, cell_params, timeseries):
#     modules_i_dict = {}
#     modules_v_dict = {}
#     modules_g_dict = {}
#
#     active_submodule_map = module_dict['MAPS']['SUBMODULES']
#     active_diode_map = module_dict['MAPS']['DIODES']
#     active_subcell_map = module_dict['MAPS']['SUBCELLS']
#     submodules = np.unique(active_submodule_map)
#     diodes = np.unique(active_diode_map)
#     subcells = np.unique(active_subcell_map)
#
#     module_irrad = module_dict['CELLSIRRADEFF']
#     whole_module_irrad = utils.expand_ndarray_2d_3d(module_irrad)
#
#     module_temp = module_dict['CELLSTEMP']
#     whole_module_temp = utils.expand_ndarray_2d_3d(module_temp)
#
#     for hoy in timeseries:
#         Imod, Vmod, Gmod = simulation_module_yield(whole_module_irrad, whole_module_temp, cell_area, cell_params, hoy,
#                                                    active_submodule_map, active_diode_map, active_subcell_map, submodules, diodes, subcells)
#         modules_i_dict.update({hoy: Imod})
#         modules_v_dict.update({hoy: Vmod})
#         modules_g_dict.update({hoy: Gmod})
#
#     return modules_i_dict, modules_v_dict, modules_g_dict
#
#
# def archive_simulation_module_yield(full_irrad, full_temp, cell_area, cell_params, hoy, active_submodule_map, active_diode_map, active_subcell_map,
#                             submodules, diodes, subcells):
#     irrad_hoy = full_irrad[:, :, hoy]
#     temp_hoy = full_temp[:, :, hoy]
#
#     Gmod = np.sum(irrad_hoy * cell_area)
#     if np.sum(irrad_hoy < cell_params['minimum_irradiance_cell']) > 0:
#         Imod, Vmod = (np.zeros(303), np.zeros(303))
#     else:
#         Imod, Vmod = calculations.calculate_module_curve(irrad_hoy, temp_hoy, cell_params, active_submodule_map)
#
#     return Imod, Vmod, Gmod
#
#
#
# def archive_run_mp_simulation(panelizer_object, surface, string):
#     # print(string)
#     timeseries = panelizer_object.all_hoy
#     ncpu = panelizer_object.ncpu
#     modules = panelizer_object.get_modules(surface, string)
#     module_dict_list = [panelizer_object.get_dict_instance([surface, string, module]) for module in modules]
#     module_dict_chunks = np.array_split(module_dict_list, ncpu)
#     module_name_chunks = np.array_split(modules, ncpu)
#
#     cell_area = panelizer_object.cell.cell_area
#     cell_params = panelizer_object.cell.parameters_dict
#
#     time_start = time.time()
#
#     with mp.Pool(processes=ncpu) as pool:
#         # print("    Pool Opened")
#         print("    -----------")
#         time.sleep(.05)
#         args = list(zip(module_dict_chunks,
#                         module_name_chunks,
#                         [cell_area] * ncpu,
#                         [cell_params] * ncpu,
#                         [timeseries] * ncpu, ))
#         # module_dict, surface, string, module, cell_area, cell_params, hoy_chunk
#         mp_results = pool.starmap(mp_module_simulation, args)
#         # print("    Result Gathered")
#         # time.sleep(1)
#         pool.close()
#         # print("    Pool closed")
#         pool.join()
#         # print("    Pool joined")
#     utils.unpack_mp_results(mp_results, panelizer_object, surface, string, modules, timeseries)
#     time_end = time.time()
#     print(f"Time elapsed for string {string}: {round(time_end - time_start, 2)}s")
#
#     # compile results list into one dict 'module',['Imod'['hoy'],'Vmod'['hoy'],'Gmod'['hoy']]
#
# if __name__=="__main__":
#     main(panelizer_object, surface, string)