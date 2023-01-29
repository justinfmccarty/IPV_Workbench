from ipv_workbench.utilities import utils


class PanelizedObject:
    def __init__(self, panelizer_file):
        self.panelizer_file = panelizer_file
        self.panelizer_dict = utils.read_pickle(self.panelizer_file)
        self.object_type = list(self.panelizer_dict.keys())[0]
        self.object_surfaces = list(self.panelizer_dict[self.object_type]['SURFACES'].keys())
        self.cell = None
        self.module = None

    def get_strings(self, surface_name):
        self.active_strings = list(self.panelizer_dict[self.object_type]['SURFACES'][surface_name]['STRINGS'].keys())
        return self.active_strings

    def get_modules(self, surface_name, string_name):
        self.active_modules = list(self.panelizer_dict[self.object_type]['SURFACES'][surface_name]['STRINGS'][string_name]['MODULES'].keys())
        return self.active_modules

    def get_submodules(self, surface_name, string_name, module_name):
        self.active_submodules = list(self.panelizer_dict[self.object_type]['SURFACES'][surface_name]['STRINGS'][string_name]['MODULES'][module_name]['SUBMODULES'].keys())
        return self.active_submodules

    def get_diodes(self, surface_name, string_name, module_name, submodule_name):
        self.active_diodes = list(self.panelizer_dict[self.object_type]['SURFACES'][surface_name]['STRINGS'][string_name]['MODULES'][module_name]['SUBMODULES'][submodule_name]['DIODEPATHS'].keys())
        return self.active_diodes

    def get_cells_xyz(self, surface_name, string_name, module_name, submodule_name, diode_name):
        self.active_cells_xyz = list(self.panelizer_dict[self.object_type]['SURFACES'][surface_name]['STRINGS'][string_name]['MODULES'][module_name]['SUBMODULES'][submodule_name]['DIODEPATHS'][diode_name]['CELLSXYZ'].keys())
        return self.active_cells_xyz

    def get_cells_normals(self, surface_name, string_name, module_name, submodule_name, diode_name):
        self.active_cells_normals = list(self.panelizer_dict[self.object_type]['SURFACES'][surface_name]['STRINGS'][string_name]['MODULES'][module_name]['SUBMODULES'][submodule_name]['DIODEPATHS'][diode_name]['CELLSNORMALS'].keys())
        return self.active_cells_normals

    def get_cells_irrad(self, surface_name, string_name, module_name, submodule_name, diode_name):
        self.active_cells_irrad = list(self.panelizer_dict[self.object_type]['SURFACES'][surface_name]['STRINGS'][string_name]['MODULES'][module_name]['SUBMODULES'][submodule_name]['DIODEPATHS'][diode_name]['CELLSIRRAD'].keys())
        return self.active_cells_irrad

    # def calculate_module_yield(self, surface_name, string_name, module_name):
    #     self.get_submodules(surface_name, string_name, module_name)
    #     if len(self.active_submodules)>1:
    #         for sub in self.active_submodules:
    #             self.calculate_submodule
    #     else:



