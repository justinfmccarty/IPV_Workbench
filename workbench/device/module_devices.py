class Module:
    def __init__(self):
        self.module_description = None

        self.front_film = None
        self.front_cover = None
        self.encapsulant = None
        self.cell = None
        self.rear_cover = None
        self.frame = None

        self.mounting_system = None
        self.air_gap = None
        self.interconnection_type = None

        self.rows_cell = None
        self.row_gutter = None
        self.columns_cell = None
        self.column_gutter = None

        # maps
        self.cell_boolean_map = None
        self.series_map = None
        self.diode_map = None
        self.xyz_relative = None
        self.xyz_absolute = None
        self.normals = None
        self.direct_irradiance = None
        self.diffuse_irradiance = None
        self.effective_irradiance = None

    def create_diode_map(self, map_type, n_size=None, n_size_direction=None):
        if map_type == 'row':
            pass
        elif map_type == 'column':
            pass
        elif map_type == 'chessboard':
            pass
        elif map_type == 'n_size':
            pass
        elif map_type == 'random':
            pass
        else:
            # use n_size method with 3 diodes
            pass

    def print_cell_description(self):
        # this is more of a test than anything
        print(self.cell.cell_description)


def build_parameter_dict(module_dict, custom_module_data, base_parameters):
    module_details = module_dict['Details']

    for k, v in custom_module_data.items():
        base_parameters[k] = v

    base_parameters['N_subcells'] = int(max(base_parameters['Nsubcell_col'], base_parameters['Nsubcell_row']))
    for k, v in base_parameters.items():
        if type(v) is str:
            try:
                base_parameters[k] = float(v)
            except ValueError:
                pass

    base_parameters['nom_eff'] = (base_parameters['Wp'] / base_parameters['module_area']) / 1000
    base_parameters['n_cols'] = module_details['n_cols']
    base_parameters['n_rows'] = module_details['n_rows']
    ideal_cell_count = base_parameters['n_cols_ideal'] * base_parameters['n_rows_ideal']
    base_parameters['total_cells'] = base_parameters['n_cols'] * base_parameters['n_rows']
    watts_cell = base_parameters['Wp'] / ideal_cell_count
    base_parameters['cell_peak_Wp'] = watts_cell
    base_parameters['actual_capacity_Wp'] = watts_cell * base_parameters['total_cells']
    m2_cell = base_parameters['module_area'] / ideal_cell_count
    base_parameters['actual_module_area_m2'] = m2_cell * base_parameters['total_cells']
    base_parameters['Wp_m2_module'] = base_parameters['Wp'] / base_parameters['actual_module_area_m2']
    base_parameters['one_subcell_area_m2'] = ((base_parameters['cell_width'] * base_parameters['cell_height']) /
                                              base_parameters['N_subcells']) * 0.000001
    base_parameters['one_cell_area_m2'] = (base_parameters['cell_width'] * base_parameters['cell_height']) * 0.000001
    base_parameters['actual_cell_area_m2'] = base_parameters['one_cell_area_m2'] * base_parameters['total_cells']
    base_parameters['Wp_m2_cell'] = base_parameters['Wp'] / base_parameters['actual_cell_area_m2']

    # base_parameters['minimum_irradiance_cell'] = 100

    # assign subcell counts if present
    actual_cols = base_parameters['n_cols']
    actual_rows = base_parameters['n_rows']
    ideal_subcell_col = base_parameters['Nsubcell_col']
    ideal_subcell_row = base_parameters['Nsubcell_row']
    if ideal_subcell_col > ideal_subcell_row:
        # print("Subcells detected for columns.")
        base_parameters['Nsubcell_col'] = actual_cols
        base_parameters['N_subcells'] = actual_cols
    elif ideal_subcell_col < ideal_subcell_row:
        # print("Subcells detected for rows.")
        base_parameters['Nsubcell_row'] = actual_rows
        base_parameters['N_subcells'] = actual_rows
    else:
        pass

    base_parameters['cell_area'] = (base_parameters['cell_area'] / (
            base_parameters['n_rows_ideal'] * base_parameters['n_cols_ideal'])) * base_parameters['total_cells']

    # module_dict['PARAMETERS'] = base_parameters.to_dict()
    base_parameters = base_parameters.to_dict()
    return base_parameters