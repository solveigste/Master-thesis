import numpy as np
import porepy as pp
import porepy.models.contact_mechanics_biot_model as model
from typing import List, Union, Dict
from functools import partial
import scipy.sparse as sps

import simulation_data as SimulationData

class ComputationalModel(model.ContactMechanicsBiot):

    def __init__(self, sim_data: SimulationData, params={"use_ad": True}):

        super().__init__(params)

        self.u_jump_variable: str = "u_jump"

        self.sim_data = sim_data

        # Mechanics problem:
        # Fluid problem:
        # material data
        self.m_kappa_c = sim_data.m_kappa_c
        self.m_normal_kappa_c = sim_data.m_normal_kappa_c
        self.m_eta = sim_data.m_eta
        self.m_c0 = sim_data.m_c0

        # fracture compliance data
        self.m_A_n = 0.0 # normal direction
        self.m_A_tau = 0.0 # tangential direction

        # fracture data
        self.initial_aperture = 0.0

        # source inside fracture
        self.scalar_source_fractures = 0
        self.p_reference = 0.0
        self.p_initial = 0.0

        self.zero_tol = 1e-6
        self.fix_only_bottom = False

        # Scaling coefficients
        # Most likely not needed
        self.scalar_scale: float = 1.0
        self.length_scale: float = 1.0

    def create_grid(self):
        """ Create the mixed-dimensional grid """

        points = topoloy = None
        constraints = None
        if self.sim_data.fractured_domain["active"]:
            angle = self.sim_data.fractured_domain["angle"]
            constraints = self.sim_data.fractured_domain["constraints"]
            if len(constraints) == 0:
                constraints = None
            theta = angle * np.pi / 180.0
            rotation = np.array(
                [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            n_fracs = self.sim_data.fractures.shape[0]
            x_data = self.sim_data.fractures[:, [0, 2]].ravel()
            y_data = self.sim_data.fractures[:, [1, 3]].ravel()
            points = np.array([x_data, y_data])
            topoloy = np.array(np.split(np.array(range(n_fracs*2)),n_fracs)).T
            for con in topoloy.T:
                xc = np.mean(points[:,con], axis=1)
                points[:,con] = rotation.dot(points[:,con] - xc) + xc

        domain = self.sim_data.domain
        if self.sim_data.dimension == 2:
            network = pp.FractureNetwork2d(points, topoloy, domain)
            mdg = network.mesh(self.sim_data.mesh_args,constraints=constraints)
            self.box = domain
            self.mdg = mdg
            self.nd = mdg.dim_max()
        elif self.sim_data.dimension == 3:
            network = pp.FractureNetwork3d(topoloy, domain)
            mdg = network.mesh(self.sim_data.mesh_args)
            self.box = domain
            self.mdg = mdg
            self.nd = mdg.dim_max()

    def allocate_u_jump_variable(self):
        for sd, data in self.mdg.subdomains(dim=self.nd - 1, return_data=True):

            displacement_jump = np.zeros(sd.num_cells * self.nd)
            data[pp.STATE][self.u_jump_variable] =  displacement_jump

    def prepare_simulation(self) -> None:
        super().prepare_simulation()
        self.allocate_u_jump_variable()

    def _set_scalar_parameters(self) -> None:
        super()._set_scalar_parameters()
        # Assign diffusivity in the normal direction of the fractures.
        for intf, data in self.mdg.interfaces(return_data=True):
            sd_primary, sd_secondary = self.mdg.interface_to_subdomain_pair(intf)
            if intf.codim == 2:
                continue
            a_secondary = self._aperture(sd_secondary)
            # Take trace of and then project specific volumes from sd_primary
            v_primary = (
                    intf.primary_to_mortar_avg()
                    * np.abs(sd_primary.cell_faces)
                    * self._specific_volume(sd_primary)
            )
            # Division by a/2 may be thought of as taking the gradient in the normal
            # direction of the fracture.
            kappa_secondary = self.m_normal_kappa_c / self._viscosity(
                sd_secondary
            )
            normal_diffusivity = intf.secondary_to_mortar_avg() * (
                    kappa_secondary * 2 / a_secondary
            )
            # The interface flux is to match fluxes across faces of sd_primary,
            # and therefore need to be weighted by the corresponding
            # specific volumes
            normal_diffusivity *= v_primary
            pp.initialize_data(
                intf,
                data,
                self.scalar_parameter_key,
                {
                    "normal_diffusivity": normal_diffusivity,
                    "vector_source": self._vector_source(intf),
                    "ambient_dimension": self.nd,
                },
            )

    def _source_scalar(self, sd: pp.Grid):
        tv = self.time_manager.time
        if sd.dim == self.nd:
            values = np.zeros(sd.num_cells)
            for i, cell_xc in enumerate(sd.cell_centers.T):
                values[i] = self.sim_data.m_f_fluid(cell_xc[0], cell_xc[1], tv)
        else:
            values = self.scalar_source_fractures * np.zeros(sd.num_cells)
        values *= self.time_manager.dt * sd.cell_volumes
        return values

    def _bc_type_mechanics(self, g):

        # collect faces indices
        _, east, west, north, south, _, _ = self._domain_boundary_sides(g)
        bc = pp.BoundaryConditionVectorial(g)

        bc_data = self.sim_data.bc_mechanics
        bc_types_neumann = np.zeros_like(north)
        bc_types_dirichlet = np.zeros_like(north)

        # Collect Dirichlet regions
        bc_regions = {"east": east, "south": south, "west": west, "north": north}
        bc_neumann_dir = []
        bc_dirichlet_dir = []
        for i, data in enumerate(bc_data["type"]):

            # Default BC is Neumann
            if "Neumann" == data:
                directions = bc_data["direction"][i]
                regions = bc_data["region"][i]
                for j, region in enumerate(regions):
                    direction = directions[j]
                    if direction == "x":
                        bc.is_neu[0, bc_regions[region]] = True
                        bc.is_dir[0, bc_regions[region]] = False
                    elif direction == "y":
                        bc.is_neu[1, bc_regions[region]] = True
                        bc.is_dir[1, bc_regions[region]] = False
                    else:
                        bc.is_neu[:, bc_regions[region]] = True
                        bc.is_dir[:, bc_regions[region]] = False

            if "Dirichlet" == data:
                directions = bc_data["direction"][i]
                regions = bc_data["region"][i]
                for j, region in enumerate(regions):
                    direction = directions[j]
                    if direction == "x":
                        bc.is_neu[0, bc_regions[region]] = False
                        bc.is_dir[0, bc_regions[region]] = True
                    elif direction == "y":
                        bc.is_neu[1, bc_regions[region]] = False
                        bc.is_dir[1, bc_regions[region]] = True
                    else:
                        bc.is_neu[:, bc_regions[region]] = False
                        bc.is_dir[:, bc_regions[region]] = True

        # Default internal BC is Neumann. We change to Dirichlet for the contact
        # problem. I.e., the mortar variable represents the displacement on the
        # fracture faces.
        frac_face = g.tags["fracture_faces"]
        bc.is_neu[:, frac_face] = False
        bc.is_dir[:, frac_face] = True

        return bc

    def _bc_type_scalar(self, g):
        # collect faces indices
        _, east, west, north, south, _, _ = self._domain_boundary_sides(g)

        bc_data = self.sim_data.bc_fluid
        bc_types_dirichlet = np.zeros_like(north)

        # Collect Dirichlet regions
        bc_regions = {"east": east, "south": south, "west": west, "north": north}
        for i, data in enumerate(bc_data["type"]):

            # Default BC is Neumann
            if "Neumann" == data:
                continue

            if "Dirichlet" == data:

                regions = bc_data["region"][i]
                for region in regions:
                    bc_types_dirichlet += bc_regions[region]

        bc = pp.BoundaryCondition(g, bc_types_dirichlet, "dir")
        return bc

    def _bc_values_mechanics(self, g):

        tv = self.time_manager.time

        # Set the boundary values
        _, east, west, north, south, _, _ = self._domain_boundary_sides(g)
        values = np.zeros((g.dim, g.num_faces))

        bc_data = self.sim_data.bc_mechanics
        # Collect Dirichlet regions
        bc_regions = {"east": east, "south": south, "west": west, "north": north}
        for i, data in enumerate(bc_data["type"]):

            if "Neumann" == data:
                x_pts = g.face_centers.T
                directions = bc_data["direction"][i]
                regions = bc_data["region"][i]
                for j, region in enumerate(regions):
                    direction = directions[j]
                    region_x_pts = g.face_centers[:, bc_regions[region]].T
                    for x_pt in region_x_pts:
                        diff = x_pts - x_pt
                        norm_diff = np.linalg.norm(diff, axis=1)
                        result = np.where(np.isclose(norm_diff, 0.0))
                        assert len(result) == 1
                        i = result[0][0]
                        sigma_n_val = self.sim_data.m_sigma_n_bc(x_pt[0], x_pt[1], tv)
                        if direction == "x":
                            values[0, i] = sigma_n_val[0]
                        elif direction == "y":
                            values[1, i] = sigma_n_val[1]
                        else:
                            values[0, i] = sigma_n_val[0]
                            values[1, i] = sigma_n_val[1]

            if "Dirichlet" == data:
                x_pts = g.face_centers.T
                directions = bc_data["direction"][i]
                regions = bc_data["region"][i]
                for j, region in enumerate(regions):
                    direction = directions[j]
                    region_x_pts = g.face_centers[:, bc_regions[region]].T
                    for x_pt in region_x_pts:
                        diff = x_pts - x_pt
                        norm_diff = np.linalg.norm(diff, axis=1)
                        result = np.where(np.isclose(norm_diff, 0.0))
                        assert len(result) == 1
                        i = result[0][0]
                        u_val = self.sim_data.m_u_bc(x_pt[0], x_pt[1], tv)
                        if direction == "x":
                            values[0, i] = u_val[0]
                        elif direction == "y":
                            values[1, i] = u_val[1]
                        else:
                            values[0, i] = u_val[0]
                            values[1, i] = u_val[1]


        return values.ravel("F")

    def _bc_values_scalar(self, sd: pp.Grid) -> np.ndarray:

        tv = self.time_manager.time

        # Set the boundary values
        _, east, west, north, south, _, _ = self._domain_boundary_sides(sd)
        values = np.zeros(sd.num_faces)

        bc_data = self.sim_data.bc_fluid
        # Collect Dirichlet regions
        bc_regions = {"east": east, "south": south, "west": west, "north": north}
        for i, data in enumerate(bc_data["type"]):

            if "Neumann" == data:
                x_pts = sd.face_centers.T
                regions = bc_data["region"][i]
                for region in regions:
                    region_x_pts = sd.face_centers[:, bc_regions[region]].T
                    for x_pt in region_x_pts:
                        diff = x_pts - x_pt
                        norm_diff = np.linalg.norm(diff, axis=1)
                        result = np.where(np.isclose(norm_diff, 0.0))
                        assert len(result) == 1
                        i = result[0][0]
                        q_n_val = self.sim_data.m_q_n_bc(x_pt[0], x_pt[1], tv)
                        values[i] = q_n_val[0]

            if "Dirichlet" == data:
                x_pts = sd.face_centers.T
                regions = bc_data["region"][i]
                for region in regions:
                    region_x_pts = sd.face_centers[:, bc_regions[region]].T
                    for x_pt in region_x_pts:
                        diff = x_pts - x_pt
                        norm_diff = np.linalg.norm(diff, axis=1)
                        result = np.where(np.isclose(norm_diff, 0.0))
                        assert len(result) == 1
                        i = result[0][0]
                        p_n_val = self.sim_data.m_p_bc(x_pt[0], x_pt[1], tv)
                        values[i] = p_n_val[0]
        return values

    def _compute_aperture(self, sd, from_iterate=True):

        apertures = np.ones(sd.num_cells)
        mdg = self.mdg
        if sd.dim == (self.nd - 1):
            # Initial aperture
            apertures *= self.initial_aperture

            data = mdg.subdomain_data(sd)
            proj = data["tangential_normal_projection"]

            # Reconstruct the displacement solution on the fracture
            sd_h = mdg.neighboring_subdomains(sd)[0]
            assert sd_h.dim == self.nd
            intf = mdg.subdomain_pair_to_interface((sd, sd_h))
            data_edge = mdg.interface_data(intf)
            if pp.STATE in data_edge:
                u_mortar_local = self.reconstruct_local_displacement_jump(
                    intf, projection=proj, from_iterate=from_iterate
                )
                # Magnitudes of normal and tangential components
                norm_u_n = np.absolute(u_mortar_local[-1])
                # Add contributions
                apertures += norm_u_n

        return apertures

    def _stiffness_tensor(self, sd: pp.Grid) -> pp.FourthOrderTensor:

        tv = self.time_manager.time
        # Rock parameters

        m_lamba = np.ones(sd.num_cells)
        m_mu = np.ones(sd.num_cells)

        for i, cell_xc in enumerate(sd.cell_centers.T):
            lame_pair = self.sim_data.m_lame_parameters(cell_xc[0], cell_xc[1], tv)
            m_lamba[i] = lame_pair[0]
            m_mu[i] = lame_pair[1]

        m_lamba = m_lamba / self.scalar_scale
        m_mu = m_mu / self.scalar_scale
        return pp.FourthOrderTensor(m_mu, m_lamba)



    def _permeability(self, sd: pp.Grid) -> np.ndarray:
        tv = self.time_manager.time
        kappa = np.zeros(sd.num_cells)
        assert self._use_ad
        if sd.dim == self.nd:
            for i, cell_xc in enumerate(sd.cell_centers.T):
                kappa[i] = self.sim_data.m_kappa(cell_xc[0], cell_xc[1], tv)
        else:
            kappa = self.m_kappa_c * np.ones(sd.num_cells)

        return kappa

    def _viscosity(self, sd: pp.Grid) -> np.ndarray:
        return self.sim_data.m_eta * np.ones(sd.num_cells)

    def _body_force(self, g: Union[pp.Grid, pp.MortarGrid]) -> np.ndarray:
        tv = self.time_manager.time
        vals = np.zeros((self.nd, g.num_cells))
        for i, cell_xc in enumerate(g.cell_centers.T):
            f_val = self.sim_data.m_f_mechanics(cell_xc[0], cell_xc[1], tv)
            vals[0, i] = f_val[0]
            vals[1, i] = f_val[1]
        vals *= g.cell_volumes
        return vals.ravel("F")

    def _reference_scalar(self, sd: pp.Grid) -> np.ndarray:
        """Reference scalar value.

        Used for the scalar (pressure) contribution to stress.
        Parameters
        ----------
        sd : pp.Grid
            Matrix grid.

        Returns
        -------
        np.ndarray
            Reference scalar value.

        """
        return np.zeros(sd.num_cells)

    def _storativity(self, sd: pp.Grid) -> np.ndarray:
        """Set unitary storativity.

        The storativity is also called Biot modulus or storage coefficient.

        Args:
            sd: Subdomain grid.

        Returns:
            np.ndarray of ones with shape (sd.num_cells, ).

        """

        return self.m_c0 * np.ones(sd.num_cells)

    def _biot_alpha(self, sd: pp.Grid) -> Union[float, np.ndarray]:
        """Set unitary Biot-Willis coefficient.
        """
        tv = self.time_manager.time
        alpha = np.zeros(sd.num_cells)
        assert self._use_ad
        for i, cell_xc in enumerate(sd.cell_centers.T):
            alpha[i] = self.sim_data.m_alpha(cell_xc[0], cell_xc[1], tv)
        return alpha

    def _aperture(self, sd: pp.Grid) -> np.ndarray:
        """
        Aperture is a characteristic thickness of a cell, with units [m].
        1 in matrix, thickness of fractures and "side length" of cross-sectional
        area/volume (or "specific volume") for intersections of co-dimension 2 and 3.
        See also specific_volume.
        """
        aperture = np.ones(sd.num_cells)
        if sd.dim < self.nd:
            aperture *= 0.1
        return aperture

    def _specific_volume(self, sd: pp.Grid) -> np.ndarray:
        """
        The specific volume of a cell accounts for the dimension reduction and has
        dimensions [m^(Nd - d)].
        Typically, equals 1 in Nd, the aperture in codimension 1 and the square/cube
        of aperture in codimensions 2 and 3.
        """
        a = self._aperture(sd)
        return np.power(a, self.nd - sd.dim)

    def _dilation_angle(self, sd):
        """Nonzero dilation angle."""
        vals = np.pi / 6 * np.ones(sd.num_cells)
        return vals

    def _reference_scalar(self, sd: pp.Grid):
        p_0_values = np.zeros(sd.num_cells)
        for i, cell_xc in enumerate(sd.cell_centers.T):
            p_0_values[i] = self.sim_data.m_p_ic(cell_xc[0], cell_xc[1])
        return p_0_values

    def _initial_condition(self) -> None:
        """
        Assign possibly nonzero (non-default) initial value.
        """
        super()._initial_condition()

        for sd, data in self.mdg.subdomains(return_data=True):
            # Initial value for the scalar variable.
            p_0_values = np.zeros(sd.num_cells)
            u_0_values = np.zeros((self.nd, sd.num_cells))
            for i, cell_xc in enumerate(sd.cell_centers.T):
                u_0_val = self.sim_data.m_u_ic(cell_xc[0], cell_xc[1])
                u_0_values[0, i] = u_0_val[0]
                u_0_values[1, i] = u_0_val[1]
                p_0_values[i] = self.sim_data.m_p_ic(cell_xc[0], cell_xc[1])

            data[pp.STATE].update({self.scalar_variable: p_0_values.copy()})
            data[pp.STATE][pp.ITERATE].update(
                {self.scalar_variable: p_0_values.copy()}
            )

            # if sd.dim == self.nd:
            data[pp.STATE].update({self.displacement_variable: u_0_values.ravel("F")})
            data[pp.STATE][pp.ITERATE].update(
                {self.displacement_variable: u_0_values.ravel("F").copy()}
            )
    def _postprocess_u_jump(self):
        sd_2 = self.mdg.subdomains(dim=self.nd)[0]
        for sd_1 in self.mdg.subdomains(dim=self.nd - 1):
            intf = self.mdg.subdomain_pair_to_interface((sd_1, sd_2))
            d_m = self.mdg.interface_data(intf)
            d_1 = self.mdg.subdomain_data(sd_1)
            u_mortar = d_m[pp.STATE][self.mortar_displacement_variable]
            displacement_jump_global_coord = (
                    intf.mortar_to_secondary_avg(nd=self.nd)
                    * intf.sign_of_mortar_sides(nd=self.nd)
                    * u_mortar
            )
            d_1[pp.STATE][self.u_jump_variable] = displacement_jump_global_coord

    def _export(self) -> None:
        diff = np.abs(self.time_manager.schedule - self.time_manager.time)
        export_data_q = np.any(np.isclose(diff, 0))
        if export_data_q:
            self._postprocess_u_jump()
            self.exporter.write_vtu([self.displacement_variable, self.scalar_variable,
                                     self.u_jump_variable,
                                     self.contact_traction_variable,
                                     self.scalar_variable],
                                    time_dependent=True)
            print("Exported quantities at time: ", self.time_manager.time)

        return None

    def _contact_mechanics_normal_equation(
        self,
        fracture_subdomains: List[pp.Grid],
    ) -> pp.ad.Operator:

        numerical_c_n = pp.ad.ParameterMatrix(
            self.mechanics_parameter_key,
            array_keyword="c_num_normal",
            subdomains=fracture_subdomains,
        )

        T_n: pp.ad.Operator = self._ad.normal_component_frac * self._ad.contact_traction

        MaxAd = pp.ad.Function(pp.ad.maximum, "max_function")
        zeros_frac = pp.ad.Array(np.zeros(self._num_frac_cells))
        u_n: pp.ad.Operator = self._ad.normal_component_frac * self._displacement_jump(
            fracture_subdomains
        )
        equation: pp.ad.Operator = T_n + MaxAd(
            (-1) * T_n - numerical_c_n * (u_n - self._gap(fracture_subdomains)),
            zeros_frac,
        )

        # Ad variable representing pressure on all fracture subdomains.
        p_frac = (
            self._ad.subdomain_projections_scalar.cell_restriction(fracture_subdomains)
            * self._ad.pressure
        )

        # In case of continuity check set self.m_A_n -> 0
        # equation: pp.ad.Operator = self.m_A_n * T_n - (u_n)
        # equation: pp.ad.Operator =  T_n + p_frac

        return equation

    def _contact_mechanics_tangential_equation(
        self,
        fracture_subdomains: List[pp.Grid],
    ) -> pp.ad.Operator:

        ad = self._ad

        # Parameter and constants
        numerical_c_t = pp.ad.ParameterMatrix(
            self.mechanics_parameter_key,
            array_keyword="c_num_tangential",
            subdomains=fracture_subdomains,
        )
        ones_frac = pp.ad.Array(np.ones(self._num_frac_cells * (self.nd - 1)))
        zeros_frac = pp.ad.Array(np.zeros(self._num_frac_cells))

        # Functions
        MaxAd = pp.ad.Function(pp.ad.maximum, "max_function")
        NormAd = pp.ad.Function(partial(pp.ad.l2_norm, self.nd - 1), "norm_function")
        tol = 1e-5
        Characteristic = pp.ad.Function(
            partial(pp.ad.functions.characteristic_function, tol),
            "characteristic_function_for_zero_normal_traction",
        )

        # Variables
        T_t = ad.tangential_component_frac * ad.contact_traction
        u_t_prime = ad.tangential_component_frac * (
            self._displacement_jump(fracture_subdomains)
            - self._displacement_jump(fracture_subdomains, previous_timestep=True)
        )
        u_t_prime.set_name("u_tau_increment")

        # Combine the above into expressions that enter the equation
        # tangential_sum = T_t + numerical_c_t * u_t_prime

        # In case of continuity check set self.m_A_tau -> 0
        return self.m_A_tau * T_t - u_t_prime

        # norm_tangential_sum = NormAd(tangential_sum)
        # norm_tangential_sum.set_name("norm_tangential")
        #
        # b_p = MaxAd(self._friction_bound(fracture_subdomains), zeros_frac)
        # b_p.set_name("bp")
        #
        # bp_tang = (ad.normal_to_tangential_frac * b_p) * tangential_sum
        #
        # maxbp_abs = ad.normal_to_tangential_frac * MaxAd(b_p, norm_tangential_sum)
        # characteristic: pp.ad.Operator = ad.normal_to_tangential_frac * Characteristic(
        #     b_p
        # )
        # characteristic.set_name("characteristic_function")
        #
        # # Compose the equation itself.
        # # The last term handles the case bound=0, in which case T_t = 0 cannot
        # # be deduced from the standard version of the complementary function
        # # (i.e. without the characteristic function). Filter out the other terms
        # # in this case to improve convergence
        # complementary_eq: pp.ad.Operator = (ones_frac - characteristic) * (
        #     bp_tang - maxbp_abs * T_t
        # ) + characteristic * (T_t)
        # return complementary_eq

    def reconstruct_normal_stress(self, previous_iterate: bool = False) -> np.ndarray:
        # First the mechanical part of the stress
        sd = self._nd_subdomain()
        data = self.mdg.subdomain_data(sd)

        # Pick the relevant displacement field
        if previous_iterate:
            u = data[pp.STATE][pp.ITERATE][self.displacement_variable]
        else:
            u = data[pp.STATE][self.displacement_variable]

        matrix_dictionary: Dict[str, sps.spmatrix] = data[pp.DISCRETIZATION_MATRICES][
            self.mechanics_parameter_key
        ]

        # Make a discretization object to get hold of the right keys to access the
        # matrix_dictionary
        mpsa = pp.Mpsa(self.mechanics_parameter_key)
        # Stress contribution from internal cell center displacements
        stress: np.ndarray = matrix_dictionary[mpsa.stress_matrix_key] * u

        # Contributions from global boundary conditions
        bound_stress_discr: sps.spmatrix = matrix_dictionary[
            mpsa.bound_stress_matrix_key
        ]
        global_bc_val: np.ndarray = data[pp.PARAMETERS][self.mechanics_parameter_key][
            "bc_values"
        ]
        stress += bound_stress_discr * global_bc_val

        # Contributions from the mortar displacement variables
        for intf, intf_data in self.mdg.interfaces(return_data=True):
            # Only contributions from interfaces to the highest dimensional grid
            if intf.dim == self.nd - 1:
                if previous_iterate:
                    u_intf: np.ndarray = intf_data[pp.STATE][pp.ITERATE][
                        self.mortar_displacement_variable
                    ]
                else:
                    u_intf = intf_data[pp.STATE][self.mortar_displacement_variable]

                stress += (
                        bound_stress_discr * intf.mortar_to_primary_avg(
                    nd=self.nd) * u_intf
                )

        data[pp.STATE]["stress"] = stress
        forces_at_faces = stress
        return forces_at_faces

    def reconstruct_fluxes(self, previous_iterate: bool = False) -> np.ndarray:
        # First the mechanical part of the stress
        pp.fvutils.compute_darcy_flux(self.mdg,
                                      keyword_store=self.scalar_parameter_key,
                                      p_name= self.scalar_variable,
                                      lam_name=self.mortar_scalar_variable,
                                      from_iterate=False)

        sd = self._nd_subdomain()
        data = self.mdg.subdomain_data(sd)
        fluxes_at_faces = data[pp.PARAMETERS][self.scalar_parameter_key]["darcy_flux"]
        return fluxes_at_faces
