"""
Factors like GWP are not calculated for a time series and are cumulative (temporally).
These tools provide quick functions that can be used to conduct rapid LCA based onthe configured object.
Inputs should be simple

GWP:
    based on allocation method (A self-allocation, B pro-rata, C grid displacement)
    Given the timeseries profile for electricity generation, system area, inverter type, and capacity
    A: for this method we only account for carbon savings based on the electricity from the grid that was not consumed.
"""

import numpy as np


def calculate_module_gwp(cell_type, area):

    if cell_type=='monocrystalline':
        module_gwp = area * 245.82 #kg Co2e per m2
    elif cell_type == 'polycrystalline':
        module_gwp = area * 202.41
    elif cell_type == 'cdte':
        module_gwp = area * 58.9
    elif cell_type == 'cigs':
        module_gwp = area * 68.9
    elif cell_type == 'asi':
        module_gwp = area * 125
    elif cell_type == 'dssc':
        module_gwp = area * 50
    else:
        module_gwp = 0

    return module_gwp

def module_projected_output(production_values, lifetime, max_performance=97, annual_factor=0.54, min_performance=80):
    derate_factors = np.linspace(max_performance,
                                 max_performance - (lifetime * annual_factor),
                                 num=lifetime).reshape(-1, 1)
    derate_factors = np.clip(derate_factors, min_performance, None) / 100
    lifetime_production = production_values * derate_factors
    return lifetime_production

def calculate_inverter_embodied(topology, array_capacity):
    if topology == 'central_inverter':
        inverter_gwp = array_capacity * 28.0 # kgCO2e per kWp
    elif topology == 'string_inverter':
        inverter_gwp = array_capacity * 48.0
    elif topology == 'micro_inverter':
        inverter_gwp = array_capacity * 96.8
    else:
        inverter_gwp = 0

    return inverter_gwp

