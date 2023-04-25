import pathlib
import numexpr
import pandas as pd
import os

def get_dw_factor(species):
    library_root = pathlib.Path(__file__).parent.parent
    default_growth_table = os.path.join(library_root, "utilities", "util_resources", "urban_volume_equations.csv")
    volume_df = pd.read_csv(default_growth_table)
    density_weight_factor_kgm3 = int(volume_df[volume_df['scientific name'] == species].iloc[0]['dw density'])
    return density_weight_factor_kgm3


def calculate_tree_carbon(species, species_code, dbh_cm, height_m=None):
    if height_m==None:
        volume_m3 = calculate_volume_one_parameter(species_code, dbh_cm)
    else:
        volume_m3 = calculate_volume_two_parameter(species_code, dbh_cm, height_m)

    density_weight_factor_kgm3 = get_dw_factor(species)

    weighted_above_biomass_kg = volume_m3 * density_weight_factor_kgm3
    total_dw_kg = weighted_above_biomass_kg * 1.28
    carbon_kg = total_dw_kg * 0.5
    return carbon_kg

def convert_c_co2(c_kg):
    return c_kg * 3.67

def calculate_volume_two_parameter(species_code, dbh_cm, height_m):
    species_code = species_code.lower()
    if species_code == "acsa1":
        volume = 0.0002383 * (dbh_cm ** 1.998) * (height_m ** 0.596)
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
    if species_code == "acsa1":
        volume = 0.000363 * (dbh_cm ** 2.292)
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

def calculate_growth(region, species_code, independent_variable, predict_variable):
    library_root = pathlib.Path(__file__).parent.parent
    default_growth_table = os.path.join(library_root, "utilities", "util_resources", "growth_coefficients.csv")
    growth_df = pd.read_csv(default_growth_table)
    region_df = growth_df[growth_df['region'] == region]
    species_df = region_df[region_df['SpCode']==species_code]
    equation_df = species_df[species_df['Predicts component']==predict_variable]
    if equation_df['Independent variable']!=independent_variable:
        print("Indepedent variable not found")
        return None
    else:
        equation = equation_df.eqname.iloc[0]
        if equation=='lin':
            predict_result = equation_lin(equation_df, independent_variable)
        elif equation=='quad':
            predict_result = equation_lin(equation_df, independent_variable)
        elif equation=='cub':
            predict_result = equation_lin(equation_df, independent_variable)
        elif equation=='quart':
            predict_result = equation_lin(equation_df, independent_variable)

        return predict_result


def format_model_weight(weight_equation, independent_variable):
    variable_strings = ["dbh", "age", "cdia", "ht"]
    if weight_equation=="1":
        return "1"
    else:
        for v in variable_strings:
            if v in weight_equation:
                formatted_equation = weight_equation.replace(v, str(independent_variable))
            else:
                pass
        return formatted_equation.replace("^","**")

def equation_lin(equation_df, independent_variable):
    a = equation_df['a'].iloc[0]
    b = equation_df['b'].iloc[0]
    model_weight = format_model_weight(equation_df['model weight'].iloc[0], independent_variable)
    weight = numexpr.evaluate(model_weight)
    x = independent_variable * weight
    return a + b * x

def equation_quad(equation_df, independent_variable):
    a = equation_df['a'].iloc[0]
    b = equation_df['b'].iloc[0]
    c = equation_df['c'].iloc[0]
    model_weight = format_model_weight(equation_df['model weight'].iloc[0], independent_variable)
    weight = numexpr.evaluate(model_weight)
    x = independent_variable * weight
    return a + b * x + c * x**2

def equation_cub(equation_df, independent_variable):
    a = equation_df['a'].iloc[0]
    b = equation_df['b'].iloc[0]
    c = equation_df['c'].iloc[0]
    d = equation_df['d'].iloc[0]
    model_weight = format_model_weight(equation_df['model weight'].iloc[0], independent_variable)
    weight = numexpr.evaluate(model_weight)
    x = independent_variable * weight
    return a + b * x + c * x**2 + d * x**3

def equation_quart(equation_df, independent_variable):
    a = equation_df['a'].iloc[0]
    b = equation_df['b'].iloc[0]
    c = equation_df['c'].iloc[0]
    d = equation_df['d'].iloc[0]
    e = equation_df['e'].iloc[0]
    model_weight = format_model_weight(equation_df['model weight'].iloc[0], independent_variable)
    weight = numexpr.evaluate(model_weight)
    x = independent_variable * weight
    return a + b * x + c * x**2 + d * x**3 + e * x**4

