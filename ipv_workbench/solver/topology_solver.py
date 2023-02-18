from ipv_workbench.translators import panelizer
from ipv_workbench.solver import simulations
from ipv_workbench.utilities import utils, time_utils, circuits

import numpy as np

def solve_central_inverter(panelizer_object, surface):
    strings_i, strings_v, strings_g = simulation_string_inverter(panelizer_object, surface)

    surface_i = {}
    surface_v = {}
    surface_g = {}

    for hoy in panelizer_object.all_hoy:
        strings_i_hoy = [strings_i[string][hoy] for string in panelizer_object.get_strings(surface)]
        strings_v_hoy = [strings_v[string][hoy] for string in panelizer_object.get_strings(surface)]
        strings_g_hoy = [strings_g[string][hoy] for string in panelizer_object.get_strings(surface)]

        Isrf, Vsrf = circuits.calc_parallel(np.array([strings_i_hoy, strings_v_hoy]))
        Gsrf = np.sum(strings_g_hoy)

        surface_i.update({hoy: Isrf})
        surface_v.update({hoy: Vsrf})
        surface_g.update({hoy: Gsrf})

        panelizer_object.get_dict_instance([surface])['CURVES'][panelizer_object.topology]['Isrf'].update(
            {hoy: np.round([Isrf], 5)})
        panelizer_object.get_dict_instance([surface])['CURVES'][panelizer_object.topology]['Vsrf'].update(
            {hoy: np.round([Vsrf], 5)})
        panelizer_object.get_dict_instance([surface])['YIELD'][panelizer_object.topology][
            'irrad'].update({hoy: [np.round(Gsrf, 1)]})

    return surface_i, surface_v, surface_g

def solve_string_inverter_iv(panelizer_object, surface, string):

    # if panelizer_object.simulation_suite == False:
    #     module_results_dict = run_mp_simulation(panelizer_object, surface, string)
    #     modules_i = [module_results_dict[module][0] for module in panelizer_object.get_modules(surface, string)]
    #     modules_v = [module_results_dict[module][1] for module in panelizer_object.get_modules(surface, string)]
    #     modules_g = [module_results_dict[module][2] for module in panelizer_object.get_modules(surface, string)]
    modules = panelizer_object.get_modules(surface, string)

    strings_i_hoy = {}
    strings_v_hoy = {}
    strings_g_hoy = {}

    for hoy in panelizer_object.all_hoy:

        # if panelizer_object.simulation_suite == False:
        #     modules_i_hoy = [modules_i[n][hoy] for n, m in enumerate(panelizer_object.get_modules(surface, string))]
        #     modules_v_hoy = [modules_v[n][hoy] for n, m in enumerate(panelizer_object.get_modules(surface, string))]
        #     modules_g_hoy = [modules_g[n][hoy] for n, m in enumerate(panelizer_object.get_modules(surface, string))]
        # else:

        modules_i_hoy = [
            panelizer_object.get_dict_instance([surface, string, module])['CURVES'][panelizer_object.topology][
                'Imod'][
                hoy] for module in modules]
        modules_v_hoy = [
            panelizer_object.get_dict_instance([surface, string, module])['CURVES'][panelizer_object.topology][
                'Vmod'][
                hoy] for module in modules]
        modules_g_hoy = [
            panelizer_object.get_dict_instance([surface, string, module])['YIELD'][panelizer_object.topology][
                'irrad'][
                hoy] for module in modules]

        module_curves = np.array([modules_i_hoy, modules_v_hoy])

        Istr, Vstr = circuits.calc_series(module_curves,
                                          breakdown_voltage=panelizer_object.cell.parameters_dict[
                                              'breakdown_voltage'],
                                          diode_threshold=panelizer_object.cell.parameters_dict['diode_threshold'],
                                          bypass=False)
        Gstr = np.sum(modules_g_hoy)

        strings_i_hoy.update({hoy: Istr})
        strings_v_hoy.update({hoy: Vstr})
        strings_g_hoy.update({hoy: Gstr})

    return strings_i_hoy, strings_v_hoy, strings_g_hoy

def solve_string_inverter_mpp(panelizer_object, string_dict):

    temp_results_dict = {}
    for hoy in panelizer_object.all_hoy:
        Istr_hoy = string_dict['CURVES'][panelizer_object.topology]['Istr'][hoy]
        Vstr_hoy = string_dict['CURVES'][panelizer_object.topology]['Vstr'][hoy]
        Gstr_hoy = string_dict['YIELD'][panelizer_object.topology]['irrad'][hoy]

        simulation_results_string_hoy = simulations.calcMPP_IscVocFF(Istr_hoy,
                                                                     Vstr_hoy)
        temp_results_dict_hoy = utils.generate_empty_results_dict(target='STRING')
        temp_results_dict_hoy.update(simulation_results_string_hoy)
        temp_results_dict_hoy.update({'irrad': Gstr_hoy})

        # store everything in a dict
        if Gstr_hoy == 0:
            temp_results_dict_hoy.update({'eff': 0})
        else:
            efficiency = temp_results_dict_hoy['pmp'] / temp_results_dict['irrad']
            temp_results_dict_hoy.update({'eff': efficiency})

        # iterate over the result keys and write then to the panelizer object

        for key in string_dict['YIELD'][panelizer_object.topology].keys():
            # write to the panelizer object at the string level
            string_dict['YIELD'][panelizer_object.topology][key].update({hoy: np.round(temp_results_dict_hoy[key], 3)})
        temp_results_dict.update({hoy:temp_results_dict_hoy})
    return temp_results_dict





def solve_micro_inverter_mpp(panelizer_object, module_dict):

    temp_results_dict = {}
    for hoy in panelizer_object.all_hoy:
        Imod_hoy = module_dict['CURVES'][panelizer_object.topology]['Imod'][hoy]
        Vmod_hoy = module_dict['CURVES'][panelizer_object.topology]['Vmod'][hoy]

        simulation_results_string = simulations.calcMPP_IscVocFF(Imod_hoy,
                                                                 Vmod_hoy)
        #  ['imp', 'vmp', 'pmp', 'isc', 'voc', 'ff']
        temp_results_dict_hoy = utils.generate_empty_results_dict(
            target='STRING')

        temp_results_dict_hoy.update(simulation_results_string)

        Gmod_hoy = module_dict['YIELD'][panelizer_object.topology]['irrad'][hoy]
        temp_results_dict_hoy.update({'irrad': Gmod_hoy})

        module_input_energy = temp_results_dict_hoy['irrad']
        if module_input_energy == 0:
            module_eff = 0
        else:
            module_eff = np.round(temp_results_dict_hoy['pmp'] / module_input_energy, 3)
        temp_results_dict_hoy.update({'module_eff': module_eff})
        temp_results_dict.update({hoy:temp_results_dict_hoy})
    return temp_results_dict