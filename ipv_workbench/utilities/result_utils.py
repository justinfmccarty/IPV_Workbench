import numpy as np

def calculate_net_demand(demand, generation):
    return np.clip(demand - generation, 0, None)