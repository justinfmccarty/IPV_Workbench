# coding=utf-8
import datetime
import math
import os
from ipv_workbench.utilities import utils
from ipv_workbench.utilities import time_utils
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pvlib
from scipy.spatial.distance import cdist
import glob

"""
This script will allow to consider the angle of incidence correction for modules.

Idea:   The direct-direct sunlight on a sensor point has a certain incidence angle. This angle shall
        be calculated from the time and the sensor point normal vector. To correctly get the sun angles
        the longitude and latitude as well as the elevation are needed.

        With the angle a correction according to the module properties are carried out for diffuse and direct
        irradiation. In the end the two components are added and then represent the irradiation on the 
        cells after the glass or module TCO

@author: Linus Walker
"""


def load_irradiance_file(bldg_radiance_dir, rad_surface_key, component):
    radiance_results_dir = os.path.join(bldg_radiance_dir, f"surface_{rad_surface_key}", "results")
    component_results_dir = os.path.join("annual_irradiance", "results", f"{component}")
    sun_up_path = os.path.join(radiance_results_dir, component_results_dir, "sun-up-hours.txt")
    ill_path = glob.glob(os.path.join(radiance_results_dir, component_results_dir, "*.ill"))[0]
    return utils.build_full_ill(sun_up_path, ill_path)


def load_grid_file(bldg_radiance_dir, rad_surface_key):
    radiance_results_dir = os.path.join(bldg_radiance_dir, f"surface_{rad_surface_key}", "results")
    pts_path = glob.glob(os.path.join(radiance_results_dir, "annual_irradiance", "model", "grid", "*.pts"))[0]
    return pd.read_csv(pts_path, delimiter=" ", header=None, names=["X", "Y", "Z", "X_v", "Y_v", "Z_v"])


def collect_raw_irradiance(pv_cells_xyz_arr, sensor_pts_xyz_arr, sensor_pts_irradiance_arr):
    # TODO change this to use rectangular sampling based on cell dimensions
    cdist_arr = cdist(pv_cells_xyz_arr, sensor_pts_xyz_arr)
    first = cdist_arr.argsort()[:,0]
    second = cdist_arr.argsort()[:,1]
    third = cdist_arr.argsort()[:,2]
    irrad_cell_mean = (sensor_pts_irradiance_arr[first] + sensor_pts_irradiance_arr[second] + sensor_pts_irradiance_arr[third]) / 3
    return irrad_cell_mean



def load_sensor_points(sensor_file):
    return pd.read_csv(sensor_file, sep=' ', header=None, names=['x', 'y', 'z', 'xdir', 'ydir', 'zdir'])

def load_irrad_data(irrad_complete_path, irrad_direct_path):
    irrad_complete = pd.read_csv(irrad_complete_path, delimiter=' ', header=None).iloc[:, 1:].T.reset_index(drop=True)
    irrad_dir_dir = pd.read_csv(irrad_direct_path, delimiter=' ', header=None).iloc[:, 1:].T.reset_index(drop=True)
    return irrad_complete, irrad_dir_dir

def angle_between_vectors(vector1, vector2):
    """
    Author: Linus Walker
    :param vector1:
    :param vector2:
    :return:
    """
    arcos_argument = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

    arcos_argument = arcos_argument - 10 ** (-9)  # This is to fix a bug. Somehow although arcos should take 1.0 as an
    # argument it fails, when arcos_argument = 1.0. With that fix, it works.

    angle = np.pi / 2 - (np.arccos(arcos_argument))
    return angle


# Azimuth = zero at North clockwise positive
def vector_to_tilt_and_azimuth(avector):
    """
    Author: Linus Walker

    :param avector:
    :return:
    """
    #TODO modifiy this to take an array and return an array in case of non-planar module
    if avector[0] == 0 and avector[1] == 0:
        tilt = 0
        azimuth = 0  # technically this is wrong
    else:
        horizontal = np.array([avector[0], avector[1], 0])
        tilt = angle_between_vectors(avector, horizontal)

        if avector[0] >= 0 and avector[1] > 0:
            azimuth = np.arctan(avector[0] / avector[1])
        elif (avector[0] >= 0) and (avector[1] < 0):
            azimuth = np.pi - (np.arctan(avector[0] / (-avector[1])))
        elif avector[0] < 0 and avector[1] < 0:
            azimuth = np.pi + (np.arctan(-avector[0] / (-avector[1])))
        elif avector[1] == 0:
            if avector[0] > 0:
                azimuth = np.pi / 2
            else:
                azimuth = np.pi * 3 / 2
        else:
            azimuth = 2 * np.pi - (np.arctan(-avector[0] / avector[1]))
    return azimuth, tilt

#
# # inputs in radians
# def calculate_angle_of_incidence(asolar_azimuth, solar_zenith, sensor_azimuth, sensor_tilt):
#     aoi = np.arccos(np.cos(solar_zenith) * np.cos(sensor_tilt) +
#                     np.sin(solar_zenith) * np.sin(sensor_tilt) * np.cos(asolar_azimuth - sensor_azimuth))
#     return aoi
#

# be sure to give theta_m in rad!!!
def k_martin_ruiz(theta_m, aa_r):
    """
    Author: Linus Walker, Justin McCarty
    :param theta_m:
    :param aa_r:
    :return:
    """
    # if theta_m > np.pi / 2:
    #     return 0
    # else:
    #     k = np.exp(1 / aa_r) * (1 - np.exp(-np.cos(theta_m) / aa_r)) / (np.exp(1 / aa_r) - 1)
    #     return k
    k = np.where(theta_m > np.pi / 2, 0,
             np.exp(1 / aa_r) * (1 - np.exp(-np.cos(theta_m) / aa_r)) / (np.exp(1 / aa_r) - 1))
    return k


k_list_global = []


def calc_angular_loss_martin_ruiz(G_dir, G_diff, theta, a_r=0.17):
    """
    Author: Linus Walker, Justin McCarty
    :param G_dir:
    :param G_diff:
    :param theta:
    :param a_r:
    :return:
    """
    k1 = k_martin_ruiz(theta, a_r)
    # This value stays constant and does not need to be recalculated each time
    # k2 =  k_martin_ruiz(np.pi/3, aa_r) (60 degrees)
    k2 = 0.94984480313119235
    G_eff = G_dir * k1 + G_diff * k2  # pi/3 is assumed for diffuse irrad
    return np.where(G_eff < 0, 0, G_eff)


def front_cover_loss(irradiance, color):
    """
    Author: Justin McCarty
    Adapted from https://doi.org/10.1016/j.enbuild.2019.109623
    P Reletive Loss in table 4, roughly mean reflectance across PV spectrum (mono)
    :param irradiance: An irradiance value to be corrected
    :param color: choose from the avaialble list...
        color         loss_factor
        gold          0.128
        purple        0.111
        pure_white    0.452
        basic_white   0.347
        medium_green  0.152
        terracotta    0.094
        dark_green    0.091
        light_grey    0.118
    :return: irradiance multiplied by the loss factor
    """
    table_filepath = os.path.join(os.path.dirname(__file__), "../simulator/front_cover_loss_table.csv")
    if os.path.exists(table_filepath):
        loss_table = pd.read_csv(table_filepath,index_col='Unnamed: 0')
    else:
        default_colors = ["gold", "purple", "pure_white", "basic_white", "medium_green", "terracotta", "dark_green",
                          "light_grey"]
        default_factors = [0.128, 0.111, 0.452, 0.347, 0.152, 0.094, 0.091,
                           0.118]
        loss_table = pd.Series(dict(zip(default_colors, default_factors)))

    loss_factor = loss_table.loc[color]
    return irradiance * (1 - loss_factor).values

def calculate_effective_irradiance_single_step(G_dir, G_diff, evaluated_normal_vector, hoy, tmy_location, pressure, drybulb, front_cover_color="clear"):
    """
    Author: Justin McCarty
    :param G_dir:
    :param G_diff:
    :param evaluated_normal_vector:
    :param hoy:
    :param tmy_location:
    :param pressure:
    :param drybulb:
    :param front_cover_color:
    :return:
    """
    # part 1: account for front cover loss
    if front_cover_color=='clear':
        G_eff_dir = G_dir
        G_eff_diff = G_diff
    else:
        G_eff_dir = front_cover_loss(G_dir, front_cover_color)
        G_eff_diff = front_cover_loss(G_diff, front_cover_color)

    # part 2: angle of incidence mod
    # TODO modifiy vector calcs to take an array and return an array in case of non-planar module
    surface_azimuth, surface_tilt = vector_to_tilt_and_azimuth(evaluated_normal_vector)
    solar_position_hoy = pvlib.solarposition.get_solarposition(time_utils.hoy_to_date(hoy),
                                                               tmy_location['lat'],
                                                               tmy_location['lon'],
                                                               altitude=tmy_location['elevation'],
                                                               pressure=pressure,
                                                               method='nrel_numpy',
                                                               temperature=drybulb
                                                               )

    surface_tilt_deg = np.rad2deg(surface_tilt)
    surface_azimuth_deg = np.rad2deg(surface_azimuth)
    solar_zenith_deg = solar_position_hoy['apparent_zenith'].values
    solar_azimuth_deg = solar_position_hoy['azimuth'].values

    aoi_mod_deg = pvlib.irradiance.aoi(surface_tilt_deg, surface_azimuth_deg, solar_zenith_deg, solar_azimuth_deg)
    aoi_mod_rad = np.deg2rad(aoi_mod_deg)
    return calc_angular_loss_martin_ruiz(G_eff_dir, G_eff_diff, aoi_mod_rad, a_r=0.17)

def calculate_effective_irradiance_timeseries(G_dir, G_diff, evaluated_normal_vector, hoy, tmy_location, pressure, drybulb, front_cover_color="clear"):
    """
    Author: Justin McCarty
    :param G_dir:
    :param G_diff:
    :param evaluated_normal_vector:
    :param hoy:
    :param tmy_location:
    :param pressure:
    :param drybulb:
    :param front_cover_color:
    :return:
    """
    # part 1: account for front cover loss
    if front_cover_color=='clear':
        G_eff_dir = G_dir
        G_eff_diff = G_diff
    else:
        G_eff_dir = front_cover_loss(G_dir, front_cover_color)
        G_eff_diff = front_cover_loss(G_diff, front_cover_color)

    # part 2: angle of incidence mod
    # TODO modifiy vector calcs to take an array and return an array in case of non-planar module
    surface_azimuth, surface_tilt = vector_to_tilt_and_azimuth(evaluated_normal_vector)

    solar_position_hoy = pvlib.solarposition.get_solarposition(time_utils.hoy_to_date(hoy),
                                                               tmy_location['lat'],
                                                               tmy_location['lon'],
                                                               altitude=tmy_location['elevation'],
                                                               pressure=pressure,
                                                               method='nrel_numpy',
                                                               temperature=drybulb
                                                               )

    surface_tilt_deg = np.rad2deg(surface_tilt)
    surface_azimuth_deg = np.rad2deg(surface_azimuth)
    solar_zenith_deg = solar_position_hoy['apparent_zenith'].values
    solar_azimuth_deg = solar_position_hoy['azimuth'].values

    aoi_mod_deg = pvlib.irradiance.aoi(surface_tilt_deg, surface_azimuth_deg, solar_zenith_deg, solar_azimuth_deg)
    aoi_mod_rad = np.deg2rad(aoi_mod_deg)
    return calc_angular_loss_martin_ruiz(G_eff_dir, G_eff_diff, aoi_mod_rad, a_r=0.17)


