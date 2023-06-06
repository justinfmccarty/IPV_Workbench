import numpy as np
import sympy


def get_orientation(module_template_orientation):
    if module_template_orientation == "P":
        return "portrait"
    elif module_template_orientation == "L":
        return "landscape"
    else:
        print("Not a correct code. Need either P (portrait) or L (landscape)")


def get_cell_type(module_template_cell):
    if module_template_cell == "A":
        return "monocrystalline"
    elif module_template_cell == "B":
        return "polycrystalline"
    elif module_template_cell == "C":
        return "cdte"
    elif module_template_cell == "D":
        return "cigs"
    elif module_template_cell == "E":
        return "asi"
    elif module_template_cell == "N":
        return "nist"


# map defintiions
def detect_nonstandard_module(module_dict):
    if module_dict['DETAILS']['n_cols'] == module_dict['DETAILS']['n_cols_ideal']:
        if module_dict['DETAILS']['n_rows'] == module_dict['DETAILS']['n_rows_ideal']:
            return "standard"
        else:
            return "nonstandard"
    else:
        return "nonstandard"


def remap_module_maps(cell_type, parameters, default_diode_map, defaul_subcell_map):
    if cell_type == 'monocrystalline':
        remap_results = remap_monocrystalline(parameters, default_diode_map)
    elif cell_type == 'polycrystalline':
        remap_results = remap_polycrystalline(parameters, default_diode_map)
    elif cell_type == 'cigs':
        remap_results = remap_cigs_no_subcell(parameters)
    elif cell_type == 'cdte':
        remap_results = remap_subcell_cols(parameters, defaul_subcell_map)
    elif cell_type == 'asi':
        remap_results = remap_subcell_cols(parameters, defaul_subcell_map)
    elif cell_type == 'nist':
        remap_results = remap_polycrystalline(parameters, defaul_subcell_map)
    else:
        print("Current cell types are monocrystalline, polycrystalline, cigs, cdte, asi")
        remap_results = None
    return remap_results


def remap_monocrystalline(parameters, default_diode_map):
    actual_cols = parameters['n_cols']
    actual_rows = parameters['n_rows']

    if parameters['orientation'] == 'portrait':
        if (actual_cols > 1) & (actual_rows > 3):
            # build new submodule_map
            if actual_rows % 2 == 0:
                new_submodule_map = np.zeros((actual_rows, actual_cols))
                new_submodule_map[int(actual_rows / 2):, ] = 1
            else:
                # temp_submodule = np.zeros((actual_rows,actual_cols)).flatten()
                # temp_submodule[int(len(temp_submodule)/2):] = 1
                # new_submodule_map = temp_submodule.reshape((actual_rows,actual_cols))
                new_submodule_map = np.zeros((actual_rows, actual_cols))

            if actual_cols % 2 == 0:
                new_diode_map = default_diode_map[0:actual_rows, 0:actual_cols]
            else:
                # temp_diode = np.zeros((actual_rows,actual_cols)).flatten()
                # temp_diode[int(len(temp_diode)/2):] = 1
                # new_diode_map = temp_diode.reshape((actual_cols,actual_rows)).T
                new_diode_map = np.zeros((actual_rows, actual_cols))
        else:
            new_submodule_map = np.zeros((actual_rows, actual_cols))
            new_diode_map = np.zeros((actual_rows, actual_cols))

    elif parameters['orientation'] == 'landscape':
        if (actual_rows > 3) & (actual_cols > 1):
            # build new submodule_map
            if actual_cols % 2 == 0:
                new_submodule_map = np.zeros((actual_rows, actual_cols))
                new_submodule_map[:, int(actual_cols / 2):] = 1
            else:
                # temp_submodule = np.zeros((actual_rows,actual_cols)).flatten()
                # temp_submodule[int(len(temp_submodule)/2):] = 1
                # new_submodule_map = temp_submodule.reshape((actual_rows,actual_cols))
                new_submodule_map = np.zeros((actual_rows, actual_cols))
            if actual_rows % 2 == 0:
                new_diode_map = default_diode_map[0:actual_rows, 0:actual_cols]
            else:
                # temp_diode = np.zeros((actual_rows,actual_cols)).flatten()
                # temp_diode[int(len(temp_diode)/2):] = 1
                # new_diode_map = temp_diode.reshape((actual_cols,actual_rows)).T
                new_diode_map = np.zeros((actual_rows, actual_cols))

        else:
            new_submodule_map = np.zeros((actual_rows, actual_cols))
            new_diode_map = np.zeros((actual_rows, actual_cols))
    else:
        print("You must specify module orientation as portrait or landscape.")
        return None

    out_submodule_map = new_submodule_map
    if out_submodule_map.shape != (actual_rows, actual_cols):
        out_submodule_map = np.zeros((actual_rows, actual_cols))

    out_diode_map = new_diode_map
    if out_diode_map.shape != (actual_rows, actual_cols):
        out_diode_map = np.zeros((actual_rows, actual_cols))
    out_subcell_map = np.zeros((actual_rows, actual_cols))

    N_p = int(len(np.unique(out_submodule_map)))
    N_s = int((actual_cols * actual_rows) / N_p)
    N_diodes = int(len(np.unique(out_diode_map)))
    N_subcells = 1

    return out_submodule_map, out_diode_map, out_subcell_map, N_s, N_p, N_diodes, N_subcells


def remap_polycrystalline(parameters, default_diode_map):
    actual_cols = parameters['n_cols']
    actual_rows = parameters['n_rows']

    if parameters['orientation'] == 'portrait':
        if (actual_cols > 1) & (actual_rows > 3):
            new_submodule_map = np.zeros((actual_rows, actual_cols))
            if actual_rows % 2 == 0:
                new_diode_map = default_diode_map[0:actual_rows, 0:actual_cols]
            else:
                new_diode_map = np.zeros((actual_rows, actual_cols))

        else:
            new_submodule_map = np.zeros((actual_rows, actual_cols))
            new_diode_map = np.zeros((actual_rows, actual_cols))
    elif parameters['orientation'] == 'landscape':
        if (actual_rows > 3) & (actual_cols > 1):
            new_submodule_map = np.zeros((actual_rows, actual_cols))
            if actual_rows % 2 == 0:
                new_diode_map = default_diode_map[0:actual_rows, 0:actual_cols]
            else:
                new_diode_map = np.zeros((actual_rows, actual_cols))

        else:
            new_submodule_map = np.zeros((actual_rows, actual_cols))
            new_diode_map = np.zeros((actual_rows, actual_cols))
    else:
        print("You must specify module orientation as portrait or landscape.")
        return None

    out_submodule_map = new_submodule_map
    if out_submodule_map.shape != (actual_rows, actual_cols):
        out_submodule_map = np.zeros((actual_rows, actual_cols))

    out_diode_map = new_diode_map
    if out_diode_map.shape != (actual_rows, actual_cols):
        out_diode_map = np.zeros((actual_rows, actual_cols))
    out_subcell_map = np.zeros((actual_rows, actual_cols))

    N_p = int(len(np.unique(out_submodule_map)))
    N_s = int((actual_cols * actual_rows) / N_p)
    N_diodes = int(len(np.unique(out_diode_map)))
    N_subcells = 1

    return out_submodule_map, out_diode_map, out_subcell_map, N_s, N_p, N_diodes, N_subcells


def remap_cigs_no_subcell(parameters):
    actual_cols = parameters['n_cols']
    actual_rows = parameters['n_rows']

    if parameters['orientation'] == 'portrait':
        if (actual_cols > 1) & (actual_rows > 3):
            # build new submodule_map
            if actual_cols % 2 == 0:
                new_submodule_map = np.zeros((actual_rows, actual_cols))
                new_submodule_map[:, int(actual_cols / 2):] = 1
            else:
                # temp_submodule = np.zeros((actual_rows,actual_cols)).flatten()
                # temp_submodule[int(len(temp_submodule)/2):] = 1
                # new_submodule_map = temp_submodule.reshape((actual_rows,actual_cols))
                new_submodule_map = np.zeros((actual_rows, actual_cols))

            if actual_rows % 2 == 0:
                diode_rows = []
                n_diodes = int(actual_rows / 2)
                for d in np.arange(0, n_diodes):
                    row_a = np.ones(actual_cols) * d
                    row_b = np.ones(actual_cols) * d
                    diode_rows.append(row_a)
                    diode_rows.append(row_b)
                new_diode_map = np.array(diode_rows)
            else:
                # temp_diode = np.zeros((actual_rows,actual_cols)).flatten()
                # temp_diode[int(len(temp_diode)/2):] = 1
                # new_diode_map = temp_diode.reshape((actual_cols,actual_rows)).T
                new_diode_map = np.zeros((actual_rows, actual_cols))
        else:
            new_submodule_map = np.zeros((actual_rows, actual_cols))
            new_diode_map = np.zeros((actual_rows, actual_cols))


    elif parameters['orientation'] == 'landscape':
        if (actual_rows > 1) & (actual_cols > 1):
            # build new submodule_map
            if actual_cols % 2 == 0:
                new_submodule_map = np.zeros((actual_rows, actual_cols))
                new_submodule_map[int(actual_rows / 2):, :] = 1
            else:
                # temp_submodule = np.zeros((actual_rows,actual_cols)).flatten()
                # temp_submodule[int(len(temp_submodule)/2):] = 1
                # new_submodule_map = temp_submodule.reshape((actual_rows,actual_cols))
                new_submodule_map = np.zeros((actual_rows, actual_cols))

            if actual_rows % 2 == 0:
                diode_cols = []
                n_diodes = int(actual_cols / 2)
                for d in np.arange(0, n_diodes):
                    col_a = np.ones(actual_rows) * d
                    col_b = np.ones(actual_rows) * d
                    diode_cols.append(col_a)
                    diode_cols.append(col_b)
                new_diode_map = np.array(diode_cols).T
            else:
                # temp_diode = np.zeros((actual_rows,actual_cols)).flatten()
                # temp_diode[int(len(temp_diode)/2):] = 1
                # new_diode_map = temp_diode.reshape((actual_cols,actual_rows)).T
                new_diode_map = np.zeros((actual_rows, actual_cols))
        else:
            new_submodule_map = np.zeros((actual_rows, actual_cols))
            new_diode_map = np.zeros((actual_rows, actual_cols))
    else:
        print("You must specify module orientation as portrait or landscape.")
        return None

    out_submodule_map = new_submodule_map
    if out_submodule_map.shape != (actual_rows, actual_cols):
        out_submodule_map = np.zeros((actual_rows, actual_cols))

    out_diode_map = new_diode_map
    if out_diode_map.shape != (actual_rows, actual_cols):
        out_diode_map = np.zeros((actual_rows, actual_cols))
    out_subcell_map = np.zeros((actual_rows, actual_cols))

    N_p = int(len(np.unique(out_submodule_map)))
    N_s = int((actual_cols * actual_rows) / N_p)
    N_diodes = int(len(np.unique(out_diode_map)))
    N_subcells = 1

    return out_submodule_map, out_diode_map, out_subcell_map, N_s, N_p, N_diodes, N_subcells


def remap_subcell_cols(parameters, default_subcell_map):
    actual_cols = parameters['n_cols']
    actual_rows = parameters['n_rows']

    if parameters['orientation'] == 'portrait':
        # build submodule_map
        new_submodule_map = np.zeros((1, actual_cols))

        # build subdiode map
        if actual_cols % 2 == 0:
            cols = []
            for n in np.arange(0, actual_cols / 2):
                [cols.append(n) for x in [0] * 2]
            new_diode_map = np.array(cols).reshape(-1, actual_cols)

        elif sympy.isprime(actual_cols):
            # number is prime, diodes will be uneven
            remainder = actual_cols % 2
            cols = []
            for n in np.arange(0, np.floor(actual_cols / 2)):
                if n == np.floor(actual_cols / 2) - 1:
                    [cols.append(n) for x in [0] * 3]
                else:
                    [cols.append(n) for x in [0] * 2]
            new_diode_map = np.array(cols).reshape(-1, actual_cols)

        else:
            if actual_cols % 3 == 0:
                divisor = 3
                cols = []
                for n in np.arange(0, actual_cols / divisor):
                    [cols.append(n) for x in [0] * divisor]
                new_diode_map = np.array(cols).reshape(-1, actual_cols)
            elif actual_cols % 5 == 0:
                divisor = 5
                cols = []
                for n in np.arange(0, actual_cols / divisor):
                    [cols.append(n) for x in [0] * divisor]
                new_diode_map = np.array(cols).reshape(-1, actual_cols)
            elif actual_cols % 7 == 0:
                divisor = 7
                cols = []
                for n in np.arange(0, actual_cols / divisor):
                    [cols.append(n) for x in [0] * divisor]
                new_diode_map = np.array(cols).reshape(-1, actual_cols)
            else:
                # cannot detect diode count
                new_diode_map = np.zeros((1, actual_cols))

        out_submodule_map = new_submodule_map
        if out_submodule_map.shape != (1, actual_cols):
            out_submodule_map = np.zeros((1, actual_cols))

        out_diode_map = new_diode_map
        if out_diode_map.shape != (1, actual_cols):
            out_diode_map = np.zeros((1, actual_cols))

    elif parameters['orientation'] == 'landscape':
        # build submodule_map
        new_submodule_map = np.zeros((actual_cols, 1))

        # build subdiode map
        if actual_rows % 2 == 0:
            rows = []
            for n in np.arange(0, actual_rows / 2):
                [rows.append(n) for x in [0] * 2]
            new_diode_map = np.array(rows).reshape(actual_rows, -1)

        elif sympy.isprime(actual_rows):
            # number is prime, diodes will be uneven
            remainder = actual_rows % 2
            rows = []
            for n in np.arange(0, np.floor(actual_rows / 2)):
                if n == np.floor(actual_rows / 2) - 1:
                    [rows.append(n) for x in [0] * 3]
                else:
                    [rows.append(n) for x in [0] * 2]
            new_diode_map = np.array(rows).reshape(actual_rows, -1)
        else:
            if actual_rows % 3 == 0:
                divisor = 3
                rows = []
                for n in np.arange(0, actual_rows / divisor):
                    [rows.append(n) for x in [0] * divisor]
                new_diode_map = np.array(rows).reshape(actual_rows, -1)
            elif actual_rows % 5 == 0:
                divisor = 5
                rows = []
                for n in np.arange(0, actual_rows / divisor):
                    [rows.append(n) for x in [0] * divisor]
                new_diode_map = np.array(rows).reshape(actual_rows, -1)
            elif actual_rows % 7 == 0:
                divisor = 7
                rows = []
                for n in np.arange(0, actual_rows / divisor):
                    [rows.append(n) for x in [0] * divisor]
                new_diode_map = np.array(rows).reshape(actual_rows, -1)
            else:
                # cannot detect diode count
                new_diode_map = np.zeros((actual_rows, 1))

        out_submodule_map = new_submodule_map
        if out_submodule_map.shape != (actual_rows, 1):
            out_submodule_map = np.zeros((actual_rows, 1))

        out_diode_map = new_diode_map
        if out_diode_map.shape != (actual_rows, 1):
            out_diode_map = np.zeros((actual_rows, 1))
    else:
        print("You must specify module orientation as portrait or landscape.")
        return None

    # build subcell map
    new_subcell_map = default_subcell_map[:actual_rows, :actual_cols]

    out_subcell_map = new_subcell_map
    if out_subcell_map.shape != (actual_rows, actual_cols):
        out_subcell_map = np.array([np.arange(0, actual_cols)] * actual_rows)

    N_p = int(len(np.unique(out_submodule_map)))
    N_diodes = int(len(np.unique(out_diode_map)))
    N_subcells = new_subcell_map.shape[0] # parameters['N_subcells']
    N_s = int((actual_cols * actual_rows) / N_subcells / N_p)

    return out_submodule_map, out_diode_map, out_subcell_map, N_s, N_p, N_diodes, N_subcells
