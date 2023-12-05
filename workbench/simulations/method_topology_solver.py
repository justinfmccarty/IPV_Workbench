from workbench.old_solver import simulations
from workbench.utilities import circuits, general

import numpy as np


def solve_central_inverter_iv(host_object, surface):
    # strings = panelizer_object.get_strings(surface)
    #
    # surface_i = {}
    # surface_v = {}
    # surface_g = {}
    #
    # for hoy in panelizer_object.analysis_period:
    #     strings_i_hoy = [
    #         panelizer_object.get_dict_instance([surface, string])['CURVES']['Istr'][hoy] for string in strings]
    #     strings_v_hoy = [
    #         panelizer_object.get_dict_instance([surface, string])['CURVES']['Vstr'][hoy] for string in strings]
    #     strings_g_hoy = [
    #         panelizer_object.get_dict_instance([surface, string])['YIELD'][panelizer_object.topology]['irrad'][hoy] for string in strings]
    #
    #     string_curves = np.array([strings_i_hoy, strings_v_hoy])
    #
    #     Isrf, Vsrf = circuits.calc_parallel(string_curves)
    #     Gsrf = np.sum(strings_g_hoy)
    #
    #     surface_i.update({hoy: Isrf})
    #     surface_v.update({hoy: Vsrf})
    #     surface_g.update({hoy: Gsrf})

    strings = host_object.get_string_keys(surface) #list(set([module['string_idx'] for module in modules]))
    surface_i = {}
    surface_v = {}
    surface_g = {}

    for hoy in host_object.analysis_period:
        strings_i_hoy = []
        strings_v_hoy = []
        strings_g_hoy = []

        for string_key in strings:
            modules = host_object.get_modules_on_string(surface, string_key)

            modules_i_hoy = [
                host_object.get_dict_instance([surface, module])['Curves']['Imod'][hoy] for module in modules]
            modules_v_hoy = [
                host_object.get_dict_instance([surface, module])['Curves']['Vmod'][hoy] for module in modules]
            modules_g_hoy = [
                host_object.get_dict_instance([surface, module])['Yield']['initial_simulation']['irrad'][hoy] for module in
                modules]

            module_curves = np.array([modules_i_hoy, modules_v_hoy])
            parameters = host_object.get_dict_instance([surface, modules[0]])['Parameters']
            Istr, Vstr = circuits.calc_series(module_curves,
                                              breakdown_voltage=parameters['bishop_breakdown_voltage'],
                                              bypass=False)
            Gstr = np.sum(modules_g_hoy)

            strings_i_hoy.append(Istr)
            strings_v_hoy.append(Vstr)
            strings_g_hoy.append(Gstr)

        string_curves = np.array([strings_i_hoy, strings_v_hoy])
        Isrf, Vsrf = circuits.calc_parallel(string_curves)
        Gsrf = np.sum(strings_g_hoy)

        surface_i.update({hoy: Isrf})
        surface_v.update({hoy: Vsrf})
        surface_g.update({hoy: Gsrf})


    surface_dict = host_object.get_dict_instance([surface])
    surface_dict['Curves']['Isrf'] = surface_i
    surface_dict['Curves']['Vsrf'] = surface_v

    surface_dict['Yield']["central_inverter"]['irrad'] = surface_g

    # return surface_i, surface_v, surface_g


def solve_central_inverter_mpp(host_object, surface):
    surface_dict = host_object.get_dict_instance([surface])

    temp_results_dict = {}
    for hoy in host_object.analysis_period:
        Isrf_hoy = surface_dict['Curves']['Isrf'][hoy]
        Vsrf_hoy = surface_dict['Curves']['Vsrf'][hoy]
        Gsrf_hoy = surface_dict['Yield']["central_inverter"]['irrad'][hoy]

        simulation_results_string_hoy = simulations.calcMPP_IscVocFF(Isrf_hoy,
                                                                     Vsrf_hoy)

        temp_results_dict_hoy = general.generate_empty_results_dict(target='SURFACE')
        temp_results_dict_hoy.update(simulation_results_string_hoy)
        temp_results_dict_hoy.update({'irrad': Gsrf_hoy})

        # store everything in a dict
        if Gsrf_hoy == 0:
            temp_results_dict_hoy.update({'eff': 0})
        else:
            efficiency = temp_results_dict_hoy['pmp'] / temp_results_dict_hoy['irrad']
            temp_results_dict_hoy.update({'eff': efficiency})

        for key in surface_dict['Yield']["central_inverter"].keys():
            if key in ['ff', 'isc', 'voc', 'area']:
                pass
            else:
                # write to the panelizer object at the string level
                surface_dict['Yield']["central_inverter"][key].update(
                    {hoy: np.round(temp_results_dict_hoy[key], 3)})
        temp_results_dict.update({hoy: temp_results_dict_hoy})
    return temp_results_dict

def solve_string_inverter_iv(host_object, surface, string):

    # if panelizer_object.simulation_suite == False:
    #     module_results_dict = run_mp_simulation(panelizer_object, surface, string)
    #     modules_i = [module_results_dict[module][0] for module in panelizer_object.get_modules(surface, string)]
    #     modules_v = [module_results_dict[module][1] for module in panelizer_object.get_modules(surface, string)]
    #     modules_g = [module_results_dict[module][2] for module in panelizer_object.get_modules(surface, string)]
    modules = host_object.get_modules_on_string(surface, string)

    string_i = {}
    string_v = {}
    string_g = {}

    for hoy in host_object.analysis_period:

        # if panelizer_object.simulation_suite == False:
        #     modules_i_hoy = [modules_i[n][hoy] for n, m in enumerate(panelizer_object.get_modules(surface, string))]
        #     modules_v_hoy = [modules_v[n][hoy] for n, m in enumerate(panelizer_object.get_modules(surface, string))]
        #     modules_g_hoy = [modules_g[n][hoy] for n, m in enumerate(panelizer_object.get_modules(surface, string))]
        # else:

        modules_i_hoy = [
            host_object.get_dict_instance([surface, module])['Curves']['Imod'][hoy] for module in modules]
        modules_v_hoy = [
            host_object.get_dict_instance([surface, module])['Curves']['Vmod'][hoy] for module in modules]
        modules_g_hoy = [
            host_object.get_dict_instance([surface, module])['Yield']['initial_simulation']['irrad'][hoy] for module in modules]

        module_curves = np.array([modules_i_hoy, modules_v_hoy])
        parameters = host_object.get_dict_instance([surface, modules[0]])['Parameters']
        Istr, Vstr = circuits.calc_series(module_curves,
                                          breakdown_voltage=parameters['bishop_breakdown_voltage'],
                                          bypass=False)
        Gstr = np.sum(modules_g_hoy)

        string_i.update({hoy: Istr})
        string_v.update({hoy: Vstr})
        string_g.update({hoy: Gstr})


    surface_dict = host_object.get_dict_instance([surface])
    string_dict = surface_dict['Strings'][string]
    string_dict['Curves']['Istr'] = string_i
    string_dict['Curves']['Vstr'] = string_v

    string_dict['Yield']['irrad'] = string_g
    # return string_i, string_v, string_g


def solve_string_inverter_mpp(host_object, surface, string):
    surface_dict = host_object.get_dict_instance([surface])
    string_dict = surface_dict['Strings'][string]


    temp_results_dict = {}
    for hoy in host_object.analysis_period:
        Istr_hoy = string_dict['Curves']['Istr'][hoy]
        Vstr_hoy = string_dict['Curves']['Vstr'][hoy]

        Gstr_hoy = string_dict['Yield']['irrad'][hoy]

        simulation_results_string_hoy = simulations.calcMPP_IscVocFF(Istr_hoy,
                                                                     Vstr_hoy)
        temp_results_dict_hoy = general.generate_empty_results_dict(target='STRING')
        temp_results_dict_hoy.update(simulation_results_string_hoy)
        temp_results_dict_hoy.update({'irrad': Gstr_hoy})

        # store everything in a dict
        if Gstr_hoy == 0:
            temp_results_dict_hoy.update({'eff': 0})
        else:
            efficiency = temp_results_dict_hoy['pmp'] / temp_results_dict_hoy['irrad']
            temp_results_dict_hoy.update({'eff': efficiency})

        # iterate over the result keys and write then to the panelizer object

        for key in string_dict['Yield'].keys():
            if key in ['ff','isc','voc']:
                pass
            else:
                # write to the panelizer object at the string level
                string_dict['Yield'][key].update({hoy: np.round(temp_results_dict_hoy[key], 3)})
        temp_results_dict.update({hoy:temp_results_dict_hoy})
    return temp_results_dict





def solve_micro_inverter_mpp(host_object, module_dict):
    temp_results_dict = {}
    for hoy in host_object.analysis_period:
        Imod_hoy = module_dict['Curves']['Imod'][hoy]
        Vmod_hoy = module_dict['Curves']['Vmod'][hoy]

        simulation_results_string = simulations.calcMPP_IscVocFF(Imod_hoy,
                                                                 Vmod_hoy)
        #  ['imp', 'vmp', 'pmp', 'isc', 'voc', 'ff']
        temp_results_dict_hoy = general.generate_empty_results_dict(
            target='STRING')

        temp_results_dict_hoy.update(simulation_results_string)

        Gmod_hoy = module_dict['Yield']['initial_simulation']['irrad'][hoy]
        temp_results_dict_hoy.update({'irrad': Gmod_hoy})

        module_input_energy = temp_results_dict_hoy['irrad']
        if module_input_energy == 0:
            module_eff = 0
        else:
            module_eff = np.round(temp_results_dict_hoy['pmp'] / module_input_energy, 3)
        temp_results_dict_hoy.update({'eff': module_eff})
        temp_results_dict.update({hoy:temp_results_dict_hoy})
    return temp_results_dict