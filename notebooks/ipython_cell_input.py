
hoy = 5987

def func():
    surface = po.get_surfaces()[0]

    strings_i = []
    strings_v = []
    for string in po.get_strings(surface):
        modules_i = []
        modules_v = []
        for module in po.get_modules(surface,string)[0:2]:
            po.get_submodule_map(surface,string, module)
            po.get_diode_map(surface,string, module)

            module_irrad = po.get_cells_irrad_eff(surface,string,module)
            full_irrad = utils.expand_ndarray_2d_3d(module_irrad)
            irrad_hoy = full_irrad[:,:,hoy]

            module_temp = po.get_cells_temp(surface,string,module)
            full_temp = utils.expand_ndarray_2d_3d(module_temp)
            temp_hoy = full_temp[:,:,hoy]
            Imod, Vmod = po.calculate_module_curve(irrad_hoy,temp_hoy)
            modules_i.append(Imod)
            modules_v.append(Vmod)
        module_curves = np.array([modules_i, modules_v])
        Istr, Vstr = circuits.calc_series(module_curves, po.cell)
        strings_i.append(Istr)
        strings_v.append(Vstr)
    string_curves = np.array([strings_i,strings_v],dtype=float)
    
func()
