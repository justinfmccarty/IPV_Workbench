import numpy as np

def calculate_net_demand(demand, generation):
    return np.clip(demand - generation, 0, None)


def calc_self_sufficiency_consumption(demand, generation):
    net_demand = demand - generation
    net_demand_clipped = np.where(net_demand < 0, 0, net_demand)
    pv_consumed = demand - net_demand_clipped

    self_sufficiency = np.where(demand == 0, 0, 100 * (pv_consumed / demand))

    # preinit the array in case generation is zero (np.where does not work with division 0)
    self_consumption = 100 * np.divide(pv_consumed, generation, out=np.zeros_like(pv_consumed), where=generation != 0)
    return self_sufficiency, self_consumption


def calc_self_sufficiency_consumption_single_value(demand, generation):
    net_demand = demand - generation
    net_demand_clipped = np.where(net_demand < 0, 0, net_demand)
    pv_consumed = demand - net_demand_clipped

    self_sufficiency = np.sum(pv_consumed) / np.sum(demand) * 100
    self_consumption = np.sum(pv_consumed) / np.sum(generation) * 100

    return self_sufficiency, self_consumption