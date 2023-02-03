# ================================ Irradiance Corrections ================================
# irradiance at cell

def calculate_effective_irradiance(direct_irrad, diffuse_irrad, film=None):
    if film == None:
        return account_for_cover(direct_irrad, diffuse_irrad)
    else:
        step_a_direct, step_a_diffuse = account_for_film(direct_irrad, diffuse_irrad)
        return account_for_cover(step_a_direct, step_a_diffuse)

def account_for_cover(direct_irrad, diffuse_irrad):
    # space filler for now
    # ideally able to represent glass, polycarbonate, PLA, etc.
    # Likely section 5.3 of Solar Engineering of Thermal Processes p.206
    return direct_irrad + diffuse_irrad


def account_for_film(direct_irrad, diffuse_irrad):
    return direct_irrad, diffuse_irrad



# ================================ Cell Temperature ================================
# cell temperature

def calculate_cell_temperature(G_effective, T_ambient, Wind_speed, method="ross"):
    G_effective = G_effective.T
    if method=="ross":
        return ross_temperature_correction(G_effective, T_ambient)
    elif method=="skoplaki":
        return skoplaki_temperature_correction(G_effective, T_ambient, Wind_speed)
    elif method=="simple":
        return simple_temperature_correction(G_effective,T_ambient)
    else:
        print("No method specificied, defaulting to Ross.")
        return ross_temperature_correction(G_effective, T_ambient)

def simple_temperature_correction(G_effective,T_ambient,G_stc=1000,T_cell_stc=40,T_ambient_stc=25):
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

def ross_temperature_correction(G_effective, T_ambient, k=0.0538):
    return T_ambient + (k * G_effective)

def skoplaki_temperature_correction(G_effective, T_ambient, Ws, w=2.4):
    return T_ambient + (w * G_effective * (0.32 / (8.91 + 2 * Ws))) #Ws=wind speed

