import json

import inspect
import marshal
import types
import spatial_properties as sp_props
import numpy as np
import csv

class SimulationData:

    def __init__(self, file_name: str ):

        sim_data = self._read_json_file(file_name)
        self.domain = sim_data.get("domain", None)
        self.fractured_domain = sim_data.get("fractured_domain", None)
        self.mesh_args = sim_data.get("mesh_args", None)
        self._process_spatial_props(sim_data)
        self._verify_assigned_spatial_props()
        self.m_eta = sim_data.get("m_eta", None)
        self.m_c0 = sim_data.get("m_c0", None)
        self.m_kappa_c = sim_data.get("m_kappa_c", None)
        self.m_normal_kappa_c = sim_data.get("m_normal_kappa_c", None)
        self.delta_t = sim_data.get("delta_t", None)
        self.final_t = sim_data.get("final_t", None)
        self.report_each_steps = sim_data.get("report_each_steps", None)
        self.bc_mechanics = sim_data.get("bc_mechanics", None)
        self.bc_fluid = sim_data.get("bc_fluid", None)

        if self.fractured_domain["active"]:
            self._read_fractures_file()
        else:
            self.fractures = np.empty((0, 4), float)
        self.dimension = 0
        if self.domain.get("zmax", None) is not None:
            self.dimension = 3
        elif self.domain.get("ymax", None) is not None:
            self.dimension = 2


    def _read_fractures_file(self):
        file_name = "input_files/fractures/" + self.fractured_domain["fractures_file"]
        self.fractures = np.empty((0, 4), float)
        with open(file_name, 'rU') as file:
            loaded = csv.reader(file)
            for line in loaded:
                frac = [float(val) for val in line]
                self.fractures = np.append(self.fractures, np.array([frac]), axis=0)


    def _read_json_file(self, file_name):
        with open(file_name) as user_file:
            file_contents = user_file.read()
            return json.loads(file_contents)

    def _process_spatial_props(self, sim_data):
        suffix = sim_data.get("function_suffix", None)
        # self.lame_parameters = d
        funcs = []
        for name, value in vars(sp_props).items():
            if name.find(suffix) == -1 or not callable(value):
                continue
            doc = inspect.getdoc(value)
            code = marshal.dumps(value.__code__)
            funcs.append({"name": name, "docstring": doc, "body": code})

        # to define an verification you should register at exactly 9 functions
        assert len(funcs) == 11

        for value in funcs:
            name = value["name"]
            doc = value["docstring"]
            code = value["body"]

            # invoke all the functions
            func = types.FunctionType(marshal.loads(code), globals(), name)
            func.__doc__ = doc
            self._assign_spatial_props(name,func)

    def _assign_spatial_props(self, name, func):
        if name.find("kappa") != -1:
            self.m_kappa = func
        if name.find("lame_parameters") != -1:
            self.m_lame_parameters = func
        if name.find("alpha") != -1:
            self.m_alpha = func
        if name.find("f_mechanics") != -1:
            self.m_f_mechanics = func
        if name.find("f_fluid") != -1:
            self.m_f_fluid = func
        if name.find("u_bc") != -1:
            self.m_u_bc = func
        if name.find("sigma_n_bc") != -1:
            self.m_sigma_n_bc = func
        if name.find("p_bc") != -1:
            self.m_p_bc = func
        if name.find("q_n_bc") != -1:
            self.m_q_n_bc = func
        if name.find("u_ic") != -1:
            self.m_u_ic = func
        if name.find("p_ic") != -1:
            self.m_p_ic = func

    def _verify_assigned_spatial_props(self):
        if self.m_kappa is None:
            raise ValueError("memeber m_kappa was not assigned.")
        if self.m_lame_parameters is None:
            raise ValueError("memeber m_lame_parameters was not assigned.")
        if self.m_alpha is None:
            raise ValueError("memeber m_alpha was not assigned.")
        if self.m_f_mechanics is None:
            raise ValueError("memeber m_f_mechanics was not assigned.")
        if self.m_f_fluid is None:
            raise ValueError("memeber m_f_fluid was not assigned.")
        if self.m_u_bc is None:
            raise ValueError("memeber m_u_bc was not assigned.")
        if self.m_sigma_n_bc is None:
            raise ValueError("memeber m_sigma_n_bc was not assigned.")
        if self.m_p_bc is None:
            raise ValueError("memeber m_p_bc was not assigned.")
        if self.m_q_n_bc is None:
            raise ValueError("memeber m_q_n_bc was not assigned.")
        if self.m_u_ic is None:
            raise ValueError("memeber m_u_ic was not assigned.")
        if self.m_p_ic is None:
            raise ValueError("memeber m_p_ic was not assigned.")