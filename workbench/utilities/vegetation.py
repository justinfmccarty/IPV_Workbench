import pathlib
import numexpr
import numpy as np
import pandas as pd
import os


def get_dw_factor(species):
    library_root = pathlib.Path(__file__).parent.parent
    default_growth_table = os.path.join(library_root, "utilities", "util_resources", "urban_volume_equations.csv")
    volume_df = pd.read_csv(default_growth_table)
    density_weight_factor_kgm3 = int(volume_df[volume_df['scientific name'] == species].iloc[0]['dw density'])
    return density_weight_factor_kgm3


def calculate_tree_carbon(species, species_code, dbh_cm, height_m=None):
    if height_m == None:
        volume_m3 = calculate_volume_one_parameter(species_code, dbh_cm)
    else:
        volume_m3 = calculate_volume_two_parameter(species_code, dbh_cm, height_m)

    if type(species) is str:
        density_weight_factor_kgm3 = get_dw_factor(species)
    else:
        density_weight_factor_kgm3 = species
    # print(species_code, volume_m3, dbh_cm, height_m, density_weight_factor_kgm3)
    weighted_above_biomass_kg = volume_m3 * density_weight_factor_kgm3
    total_dw_kg = weighted_above_biomass_kg * 1.28
    carbon_kg = total_dw_kg * 0.5
    return round(carbon_kg,3)


def convert_c_co2(c_kg):
    return c_kg * 3.67


def calculate_volume_two_parameter(species_code, dbh_cm, height_m):
    species_code = species_code.lower()
    if "acsa" in species_code:
        volume = 0.0002383 * (dbh_cm ** 1.998) * (height_m ** 0.596)
    elif species_code == 'acpl':
        volume = 0.001011 * (dbh_cm ** 1.533) * (height_m ** 0.657)
    elif species_code == 'frla':
        volume = 0.0004143 * (dbh_cm ** 1.847) * (height_m ** 0.646)
    elif species_code == 'fasy':
        # doesnt exist in database use UGEB
        volume = 0.0001967 * (dbh_cm ** 1.951853) * (height_m ** 0.664255)
    elif species_code == "ugeb":
        volume = 0.001967 * (dbh_cm ** 1.951853) * (height_m ** 0.664255)
    elif species_code == "ugec":
        volume = 0.0000426 * (dbh_cm ** 2.24358) * (height_m ** 0.64956)
    else:
        print("Species code not specified. Returning tuple of two possible results."
              "Element one is for general urban broadleaf. Element two is for general urban conifer.")
        volume_a = 0.001967 * dbh_cm ** 1.951853 * height_m ** 0.664255
        volume_b = 0.0000426 * dbh_cm ** 2.24358 * height_m ** 0.64956
        volume = (volume_a, volume_b)

    return volume


def calculate_volume_one_parameter(species_code, dbh_cm):
    species_code = species_code.lower()
    if "acsa" in species_code:
        volume = 0.000363 * (dbh_cm ** 2.292)
    elif species_code == 'acpl':
        volume = 0.0019421 * (dbh_cm ** 1.785)
    elif species_code == 'frla':
        volume = 0.0005885 * (dbh_cm ** 2.206)
    elif species_code == 'fasy':
        # doesnt exist in database use UGEB
        volume = 0.0002835 * (dbh_cm ** 1.310647)
    elif species_code == "ugeb":
        volume = 0.0002835 * (dbh_cm ** 1.310647)
    elif species_code == "ugec":
        volume = 0.0000698 * (dbh_cm ** 2.578027)
    else:
        print("Species code not specified. Returning tuple of two possible results."
              "Element one is for general urban broadleaf. Element two is for general urban conifer.")
        volume_a = 0.0002835 * (dbh_cm ** 1.310647)
        volume_b = 0.0000698 * (dbh_cm ** 2.578027)
        volume = (volume_a, volume_b)

    return volume


def get_species_growth_df(region, species_code):
    library_root = pathlib.Path(__file__).parent.parent
    default_growth_table = os.path.join(library_root, "utilities", "util_resources", "growth_coefficients.csv")
    growth_df = pd.read_csv(default_growth_table)
    region_df = growth_df[growth_df['region'] == region]
    return region_df[region_df['spcode'] == species_code]


def calculate_growth(region, species_code, independent_variable_value, independent_variable_name, predict_variable):
    species_df = get_species_growth_df(region, species_code)
    equation_df = species_df[(species_df['predicts component'] == predict_variable) & (
                species_df['independent variable'] == independent_variable_name)]
    if len(equation_df)==0:
        print("Independent variable not found")
        return None
    else:

        equation = equation_df.eqname.iloc[0]
        if equation == 'lin':
            predict_result = equation_lin(equation_df, independent_variable_value)
        elif equation == 'quad':
            predict_result = equation_quad(equation_df, independent_variable_value)
        elif equation == 'cub':
            predict_result = equation_cub(equation_df, independent_variable_value)
        elif equation == 'quart':
            predict_result = equation_quart(equation_df, independent_variable_value)
        elif equation == 'loglogw1':
            predict_result = equation_logw1(equation_df, independent_variable_value)
        elif equation == 'loglogw2':
            predict_result = equation_logw2(equation_df, independent_variable_value)
        else:
            print(f"The equation {equation} is missing from database")
            predict_result = None
        return round(predict_result,3)


def format_model_weight(weight_equation, independent_variable):
    variable_strings = ["dbh", "age", "cdia", "ht"]
    if weight_equation == "1":
        return "1"
    else:
        for v in variable_strings:
            if v in weight_equation:
                formatted_equation = weight_equation.replace(v, str(independent_variable))
            else:
                pass
        return formatted_equation.replace("^", "**")


def equation_lin(equation_df, independent_variable):
    a = equation_df['a'].iloc[0]
    b = equation_df['b'].iloc[0]
    # model_weight = format_model_weight(equation_df['model weight'].iloc[0], independent_variable)
    # x = numexpr.evaluate(model_weight)
    x = independent_variable
    return a + b * x


def equation_quad(equation_df, independent_variable):
    a = equation_df['a'].iloc[0]
    b = equation_df['b'].iloc[0]
    c = equation_df['c'].iloc[0]
    # model_weight = format_model_weight(equation_df['model weight'].iloc[0], independent_variable)
    # x = numexpr.evaluate(model_weight)
    x = independent_variable
    return a + b * x + c * x ** 2


def equation_cub(equation_df, independent_variable):
    a = equation_df['a'].iloc[0]
    b = equation_df['b'].iloc[0]
    c = equation_df['c'].iloc[0]
    d = equation_df['d'].iloc[0]
    # model_weight = format_model_weight(equation_df['model weight'].iloc[0], independent_variable)
    # x = numexpr.evaluate(model_weight)
    x = independent_variable
    return a + b * x + c * x ** 2 + d * x ** 3


def equation_quart(equation_df, independent_variable):
    a = equation_df['a'].iloc[0]
    b = equation_df['b'].iloc[0]
    c = equation_df['c'].iloc[0]
    d = equation_df['d'].iloc[0]
    e = equation_df['e'].iloc[0]
    # model_weight = format_model_weight(equation_df['model weight'].iloc[0], independent_variable)
    # x = numexpr.evaluate(model_weight)
    x = independent_variable
    return a + b * x + c * x ** 2 + d * x ** 3 + e * x ** 4

def equation_logw1(equation_df, independent_variable):
    a = equation_df['a'].iloc[0]
    b = equation_df['b'].iloc[0]
    c = equation_df['c'].iloc[0]
    x = independent_variable
    return np.exp(a + b * np.log(np.log(x + 1) + (c / 2)))

def equation_logw2(equation_df, independent_variable):
    a = equation_df['a'].iloc[0]
    b = equation_df['b'].iloc[0]
    c = equation_df['c'].iloc[0]
    x = independent_variable
    return np.exp(a + b * np.log(np.log(x + 1)) + (np.sqrt(x) + (c / 2)))