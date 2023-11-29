


def calculate_cell_temperature(G_effective, T_ambient, T_noct=45, Wind_speed=None, Pretrained_model=None, method="ross"):
    """

    :param G_effective:
    :param T_ambient:
    :param T_noct:
    :param Wind_speed:
    :param Pretrained_model:
    :param method:
    :return:
    """
    if method == "ross_simple":
        return ross_temperature_correction_simple(G_effective, T_ambient)
    elif method == "ross":
        return ross_temperature_correction(G_effective, T_ambient, T_noct=T_noct)
    elif method == "skoplaki":
        if Wind_speed is None:
            print("Wind Speed required in scalar or same shape as DBT")
            return None
        return skoplaki_temperature_correction(G_effective, T_ambient, Wind_speed)
    elif method == "simple":
        return simple_temperature_correction(G_effective, T_ambient)
    # elif method == "Pit's Super Cool Model":
    #     return super_cool_function(Pretrained_model, G_effective, T_ambient)
    else:
        print("No method specified, defaulting to Ross.")
        return ross_temperature_correction_simple(G_effective, T_ambient)


def simple_temperature_correction(G_effective, T_ambient, G_stc=1000, T_cell_stc=40, T_ambient_stc=25):
    """_summary_

    Args:
        T_ambient (_type_): ambient temperature
        G_effective (_type_): plane of array irradiance
        G_stc (_type_): plane of array irradiance during STC
        T_cell_stc (_type_): temperature of cell during STC
        T_ambient_stc (_type_): ambient temperature during STC

    Returns:
        _type_: _description_
    """
    return T_ambient + (G_effective / G_stc) * (T_cell_stc - T_ambient_stc)

def ross_temperature_correction(G_effective, T_ambient, T_noct=45):
    # factor of 0.1 converts irradiance from W/m2 to mW/cm2
    return T_ambient + (T_noct - 20.) / 80. * G_effective * 0.1

def ross_temperature_correction_simple(G_effective, T_ambient, k=0.0538):

    return T_ambient + (k * G_effective)


def skoplaki_temperature_correction(G_effective, T_ambient, Ws, w=2.4):
    return T_ambient + (w * G_effective * (0.32 / (8.91 + 2 * Ws)))  # Ws=wind speed
