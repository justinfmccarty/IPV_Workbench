import os
from pathlib import Path
import numpy as np
import pandas as pd
from ipv_workbench.utilities import utils, time_utils
import openpyxl

def write_building_results_timeseries(po, scenario, topology):
    year = scenario.split("_")[-1]

    # load demand profile
    electricity_load = pd.read_csv(os.path.join(po.RESOURCES_DIR, "loads", "annual_building_demand_time_period.csv"))[
        f"grid_demand_kwh_{year}"]
    electricity_load_timeseries = electricity_load.values

    # set result file
    building_target_file = os.path.join(po.RESULTS_DIR, 'timeseries',
                                        f"{scenario}_{topology}_building_level_results_hourly.csv")
    utils.directory_creator(Path(building_target_file).parent)
    # building level results
    results_dict = {}

    # overall_results
    object_dict = po.get_dict_instance([])
    object_capacity_w = object_dict['DETAILS']['installed_capacity_Wp']
    object_capacity_kw = object_capacity_w / 1000
    object_area = object_dict['DETAILS']['installed_area_m2']

    results_dict.update(
        {f"index": po.get_tabular_results([], topology=topology, analysis_period=None, rename_cols=False)[['pmp']].index})

    generation_total_wh = po.get_tabular_results([], topology=topology, analysis_period=None, rename_cols=False)[
        ['pmp']].values.flatten()
    generation_total_kwh = generation_total_wh / 1000
    results_dict.update({f"electricity_gen_bulk_building_kwh": generation_total_kwh})

    # add electricity demand (although it is overwritten at the end
    results_dict.update({"electricity_demand_building_kwh": electricity_load_timeseries})


    object_specific_yield = generation_total_kwh / object_capacity_kw
    results_dict.update({f"electricity_specific_yield_building_kwh_kwp": object_specific_yield})

    object_generation_intensity = generation_total_kwh / object_area
    results_dict.update({f"electricity_gen_intensity_building_kwh": object_generation_intensity})



    # surface_level_results
    surface_irrad = []
    for surface in po.get_surfaces():
        surface_dict = po.get_dict_instance([surface])
        surface_clean = utils.clean_grasshopper_key(surface)
        general_dir = surface_dict['DETAILS']['general_angle']

        # generation
        generation_surface_wh = \
        po.get_tabular_results([surface], topology=topology, analysis_period=None, rename_cols=False)[
            ['pmp']].values.flatten()
        generation_surface_kwh = generation_surface_wh / 1000
        results_dict.update({f"electricity_gen_bulk_{surface_clean}_{general_dir}_kwh": generation_surface_kwh})

        # capacity
        surface_capacity_w = surface_dict['DETAILS']['installed_capacity_Wp']
        surface_capacity_kw = surface_capacity_w / 1000
        results_dict.update({f"surface_capacity_{surface_clean}_{general_dir}_kwp": [surface_capacity_kw] * len(generation_surface_kwh)})

        # area
        surface_area = surface_dict['DETAILS']['installed_area_m2']
        results_dict.update({f"surface_area_{surface_clean}_{general_dir}_m2": [surface_area] * len(generation_surface_kwh)})

        surface_specific_yield = generation_surface_kwh / surface_capacity_kw
        results_dict.update(
            {f"electricity_specific_yield_{surface_clean}_{general_dir}_kwh_kwp": surface_specific_yield})

        # gen intensity
        surface_generation_intensity = generation_surface_kwh / surface_area
        results_dict.update(
            {f"electricity_gen_intensity_{surface_clean}_{general_dir}_kwh_m2": surface_generation_intensity})

        # efficiency
        # surface_efficiency = np.fromiter(surface_dict['YIELD'][topology]['eff'].values(), dtype=float)

        surface_effiency_series = pd.Series(surface_dict['YIELD'][topology]['eff'])
        hoy_index = pd.Series(np.arange(0, 8760, 1), name='HOY')
        surface_effiency_annual_df = pd.concat([hoy_index, surface_effiency_series], axis=1)
        surface_efficiency = surface_effiency_annual_df[0].values
        results_dict.update({f"efficiency_{surface_clean}_{general_dir}_yield_irrad": surface_efficiency})

        # print(surface_efficiency.max())
        # surface_irrad_kwh = np.fromiter(surface_dict['YIELD'][topology]['irrad'].values(), dtype=float) / 1000

        surface_irrad_wh_series = pd.Series(surface_dict['YIELD'][topology]['irrad'])
        hoy_index = pd.Series(np.arange(0, 8760, 1), name='HOY')
        surface_irrad_wh_annual_df = pd.concat([hoy_index, surface_irrad_wh_series], axis=1)
        surface_irrad_kwh = surface_irrad_wh_annual_df[0].values / 1000

        surface_irrad.append(surface_irrad_kwh)
        results_dict.update({f"irrad_bulk_{surface_clean}_{general_dir}_kWh": surface_irrad_kwh})

        # rad intensity
        surface_irrad_intensity_kwh = surface_irrad_kwh / surface_area
        results_dict.update({f"irrad_intensity_{surface_clean}_{general_dir}_kWh_m2": surface_irrad_intensity_kwh})

        # self sufficiency, consumption
        self_suff, self_cons = utils.calc_self_sufficiency_consumption(electricity_load_timeseries,
                                                                       generation_surface_kwh)
        results_dict.update({f"self_sufficiency_{surface_clean}_{general_dir}_percent": self_suff})
        results_dict.update({f"self_consumption_{surface_clean}_{general_dir}_percent": self_cons})

        # index datetime
        results_dict.update({"index": time_utils.hoy_to_date(np.arange(0,8760,1))})

    results_dict.update({f"irrad_whole_building_kwh": np.sum(surface_irrad,axis=0)})
    df = pd.DataFrame(results_dict).set_index("index").round(3)

    sunup_df = df.iloc[po.sunup_array].interpolate()
    sundown_df = df.iloc[po.sundown_array].fillna(0)

    final_df = pd.concat([sunup_df, sundown_df]).sort_index()
    final_df["electricity_demand_building_kwh"] = electricity_load_timeseries
    # self sufficiency, consumption
    self_suff, self_cons = utils.calc_self_sufficiency_consumption(electricity_load_timeseries, final_df["electricity_gen_bulk_building_kwh"].values)
    final_df["self_sufficiency_building_percent"] = self_suff
    final_df["self_consumption_building_percent"] = self_cons
    final_df.to_csv(building_target_file)

    return building_target_file


def write_cumulative_scenario_results(project_folder, scenario, topology, bldg_results_list):
    year = scenario.split("_")[-1]

    # load demand profile
    electricity_load = pd.read_csv(os.path.join(project_folder,'shared', 'resources', "loads", "annual_building_demand_time_period.csv"))[
        f"grid_demand_kwh_{year}"]
    electricity_load_timeseries = electricity_load.values

    # set result file
    cumulative_target_file = os.path.join(project_folder, 'objects', "cumulative_results",
                                          f"{scenario}_{topology}_cumulative_hourly.csv")

    utils.directory_creator(Path(cumulative_target_file).parent)

    # bldg_df_cols = bldg_results_list[0].columns

    cumulative_dict = {}

    # bulk_generation
    col = "electricity_gen_bulk_building_kwh"
    cumulative_yield = np.sum([df[col] for df in bldg_results_list], axis=0)
    cumulative_dict.update({f"electricity_gen_cumulative_kwh": cumulative_yield})

    # bulk_irrad
    bldg_irrad = []
    for bldg_df in bldg_results_list:
        irrad_cols = [col for col in bldg_df.columns if "irrad_bulk_" in col]
        bldg_irrad.append(np.sum(bldg_df[irrad_cols],axis=1))

    cumulative_irrad = np.sum(bldg_irrad, axis=0)
    cumulative_dict.update({f"irradiance_cumulative_kwh": cumulative_irrad})

    # self_sufficiency
    self_suff, self_cons = utils.calc_self_sufficiency_consumption(electricity_load_timeseries, cumulative_yield)
    cumulative_dict.update({f"self_sufficiency_cumulative_percent": self_suff})
    cumulative_dict.update({f"self_consumption_cumulative_percent": self_cons})

    cumulative_dict.update({f"electricity_demand_kwh": electricity_load_timeseries})

    for direction in [("_r_", "roof_tops"), ("_west_", "west_facade"), ("_east_", "east_facade"),
                      ("_north_", "north_facade"), ("_south_", "south_facade")]:
        # build col lists
        bulk_gen_buildings = []
        capacity_buildings = []
        area_buildings = []
        irrad_buildings = []
        for bldg_df in bldg_results_list:
            gen_cols = [col for col in bldg_df.columns if (f"{direction[0]}" in col) & ("gen_bulk" in col)]
            bulk_gen_buildings.append(np.sum(bldg_df[gen_cols].values, axis=1))

            capacity_cols = [col for col in bldg_df.columns if (f"{direction[0]}" in col) & ("surface_capacity_" in col)]
            capacity_buildings.append(np.sum(bldg_df[capacity_cols].values, axis=1))

            area_cols = [col for col in bldg_df.columns if (f"{direction[0]}" in col) & ("surface_area_" in col)]
            area_buildings.append(np.sum(bldg_df[area_cols].values, axis=1))

            irrad_cols = [col for col in bldg_df.columns if (f"{direction[0]}" in col) & ("irrad_bulk_" in col)]
            irrad_buildings.append(np.sum(bldg_df[irrad_cols].values, axis=1))

        # bulk_generation
        directional_generation = np.sum(bulk_gen_buildings,axis=0)
        directional_generation = np.where(np.isnan(directional_generation), 0, directional_generation)
        cumulative_dict.update({f"electricity_gen_{direction[1]}_kwh": directional_generation})

        # capacity
        directional_capacity = np.sum(capacity_buildings, axis=0)
        cumulative_dict.update({f"surface_capacity_{direction[1]}_kwp": directional_capacity})

        # area
        directional_installed_area = np.sum(area_buildings, axis=0)
        cumulative_dict.update({f"surface_area_{direction[1]}_m2": directional_installed_area})

        cumulative_dict.update(
            {f"electricity_specific_yield_{direction[1]}_kwh_kwp": directional_generation / directional_capacity})
        cumulative_dict.update(
            {f"electricity_gen_intensity_{direction[1]}_kwh_m2": directional_generation / directional_installed_area})

        # irrad
        irrad_bulk = np.sum(irrad_buildings, axis=0)
        cumulative_dict.update({f"irrad_bulk_{direction[1]}_kwh": irrad_bulk})

        # irrad_intensity
        irrad_intensity = irrad_bulk / directional_installed_area
        cumulative_dict.update({f"irrad_intensity_{direction[1]}_kwh_m2": irrad_intensity})

        # self_sufficiency
        self_suff, self_cons = utils.calc_self_sufficiency_consumption(electricity_load_timeseries,
                                                                       directional_generation)
        cumulative_dict.update({f"self_sufficiency_{direction[1]}_percent": self_suff})
        cumulative_dict.update({f"self_consumption_{direction[1]}_percent": self_cons})

        # index datetime
        cumulative_dict.update({"index": time_utils.hoy_to_date(np.arange(0,8760,1))})

    df = pd.DataFrame(cumulative_dict).set_index("index").round(3)
    df.to_csv(cumulative_target_file)

    return df

def change_scenario_code(whole_scenario):
    scenario = whole_scenario.split("_")
    module = scenario[0][0]
    orientation = scenario[0][1]
    if module == 'A':
        module_long = 'monocrystalline'
    elif module == 'B':
        module_long = 'polycrystalline'
    elif module == 'C':
        module_long = 'cdte'
    elif module == 'D':
        module_long = 'cigs'
    else:
        module_long = 'asi'

    if orientation == 'P':
        orientation_long = 'portrait'
    else:
        orientation_long = 'landscape'

    return f"{module_long}_{orientation_long}_{scenario[1]}_{scenario[2]}_{scenario[3]}"


def write_condensed_result(project_folder, object_detail_dicts, df, scenario, topology):
    scenario_long = change_scenario_code(scenario)

    pmp = df['electricity_gen_cumulative_kwh'].sum()
    irrad = df['irradiance_cumulative_kwh'].sum()

    excel_dest = os.path.join(project_folder, 'objects', "cumulative_condensed",
                              f"{scenario_long}_{topology}_cumulative_results.xlsx")

    my_cols = ['electricity_gen_cumulative_kwh', 'electricity_gen_east_facade_kwh', 'electricity_gen_west_facade_kwh',
               'electricity_gen_south_facade_kwh', 'electricity_gen_north_facade_kwh', 'electricity_gen_roof_tops_kwh']
    dest_cols = ['E_PV_gen_kWh', 'Electricity production from photovoltaic panels on east facades [kWh]',
                 'Electricity production from photovoltaic panels on west facades [kWh]',
                 'Electricity production from photovoltaic panels on south facades [kWh]',
                 'Electricity production from photovoltaic panels on north facades [kWh]',
                 'Electricity production from photovoltaic panels on roof tops [kWh]']

    df[my_cols].rename(columns=dict(zip(my_cols, dest_cols))).to_excel(excel_dest)

    # load excel file
    workbook = openpyxl.load_workbook(filename=excel_dest)

    # open workbook
    sheet = workbook.active

    # area
    row = 2
    sheet[f"N{row}"] = "total PV area (m2)"
    sheet[f"O{row}"] = np.sum([object_dict_detail['installed_area_m2'] for object_dict_detail in object_detail_dicts])

    # capacity
    row = 5
    sheet[f"N{row}"] = "nominal power (W)"
    sheet[f"O{row}"] = np.sum([object_dict_detail['installed_capacity_Wp'] for object_dict_detail in object_detail_dicts])

    # area
    row = 6
    sheet[f"N{row}"] = "module area (m2)"
    sheet[f"O{row}"] = np.sum([object_dict_detail['installed_area_m2'] for object_dict_detail in object_detail_dicts])

    # capacity
    row = 8
    sheet[f"N{row}"] = "kW (inverter)"
    sheet[f"O{row}"] = np.sum([object_dict_detail['installed_capacity_Wp'] for object_dict_detail in object_detail_dicts]) / 1000

    # efficiency
    total_power = pmp#np.sum([pmp for pmp in pmp_results])
    total_irrad = irrad#np.sum([irrad for irrad in irrad_results])
    row = 10
    sheet[f"N{row}"] = "system efficiency (%)"
    sheet[f"O{row}"] = np.round(100 * (total_power / total_irrad), 3)

    # save the file
    workbook.save(filename=excel_dest)
    write_out_folder = os.path.join(r"C:\Users\Justin\Nextcloud\Teaching\22_HS\polikseni_bano\polikseni_share\condensed_simulation_results",
                                    f"{scenario_long}_{topology}_cumulative_results.xlsx")
    workbook.save(filename=write_out_folder)