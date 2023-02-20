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


def main():
    project_folder = "/Users/jmccarty/Desktop/bano_simulations"
    year_list = [2020, 2050, 2080]
    building_list = ["B1391"] #["B1360", "B1389", "B1390", "B1391", "B1392", "B1394", "B2494"]
    all_topologies = ['micro_inverter', 'string_inverter', 'central_inverter']
    log_file = os.path.join(project_folder,'shared','resources','log_file.txt')
    for year in year_list:
        for cell_technology in ["A", "B", "C", "D", "E"]:
            for orientation in ["P", "L"]:
                for front_cover in ["solar_glass", "light_grey", "basic_white"]:
                    scenario = f"{cell_technology}{orientation}_{front_cover}_{year}"
                    print(r"-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-")
                    print(scenario)
                    building_results_files = {all_topologies[0]:[],
                                              all_topologies[1]:[],
                                              all_topologies[2]:[]}
                    object_detail_dicts = []
                    pmp_results = {}
                    irrad_results = {}
                    for building in tqdm(building_list):
                        start_time = time.time()

                        # set up the simulation

                        raw_panelizer_file = f"{cell_technology}{orientation}_{front_cover}_{building}_raw.pickle"
                        panelizer_object = panelizer.PanelizedObject(project_folder, building, raw_panelizer_file)
                        panelizer_object.analysis_location = 'zurich'
                        panelizer_object.analysis_year = year
                        panelizer_object.set_tmy_data()
                        all_hoy = panelizer_object.set_analysis_period(0, 4000, 1)
                        custom_device_data = pd.read_csv(panelizer_object.module_cell_data, index_col='scenario').loc[
                            f"{cell_technology}{orientation}"].to_dict()
                        panelizer_object.cell = devices.Cell(custom_device_data)

                        # run the major simulation
                        panelizer.solve_object_module_iv(panelizer_object, mp=True)

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

                        # write up results and set new operating points from higher inverters
                        extra_keys = ["actual_module_area_m2", 'actual_capacity_Wp']

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
                        # print("!!!! compressing_pickle")
                        # pickle_start = time.time()
                        # # seems to be about 2.25GB stored in 30min for a full building scenario
                        # utils.write_pickle(panelizer_object,
                        #                    os.path.join(panelizer_object.COLD_DIR,f"{scenario}_completed.pickle.xz"),
                        #                    compress=True)
                        # print(round(time.time() - pickle_start))
                        object_detail_dicts.append(copy.deepcopy(panelizer_object.get_dict_instance([])['DETAILS']))

                        for topology in all_topologies:
                            # write results
                            building_results_file = ipv_results.write_building_results_timeseries(panelizer_object,
                                                                                                  scenario, topology)
                            building_results_files[topology].append(building_results_file)

                            pmp_results.update(
                                {topology: np.sum(np.fromiter(panelizer_object.get_dict_instance([])['YIELD'][topology]['pmp'].values(), dtype=float))})
                            irrad_results.update(
                                {topology: np.sum(np.fromiter(panelizer_object.get_dict_instance([])['YIELD'][topology]['irrad'].values(), dtype=float))})
                        end_time = time.time()
                        run_time = end_time - start_time
                        log_string = f"{year},{cell_technology},{orientation},{front_cover},{building},{np.round(run_time,3)}\n"
                        utils.log_run(log_file, log_string)
                    for topology in all_topologies:
                        bldg_results = [pd.read_csv(fp, index_col="index") for fp in building_results_files[topology]]
                        cumulative_df = ipv_results.write_cumulative_scenario_results(project_folder,
                                                                                      all_hoy,
                                                                                      scenario,
                                                                                      topology,
                                                                                      bldg_results)


                        ipv_results.write_condensed_result(project_folder,
                                                           object_detail_dicts,
                                                           pmp_results[topology],
                                                           irrad_results[topology],
                                                           cumulative_df,
                                                           scenario,
                                                           topology)
if __name__ == "__main__":
    main()