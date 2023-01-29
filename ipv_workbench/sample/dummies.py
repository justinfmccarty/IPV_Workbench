import pandas as pd
import sys
import os
import random


def read_sample_irradiance():
    return pd.read_csv(os.path.join(os.path.dirname(__file__),
                                    "irradiance_profiles.csv"))


def get_sample_profiles(number, method='random'):
    df = read_sample_irradiance()
    if method == 'random':
        randomlist = []
        for i in range(0, number):
            col = str(random.randint(0, 99))
            randomlist.append(df[col].tolist())
        return randomlist
