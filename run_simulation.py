import numpy as np
import porepy as pp
from simulation_data import SimulationData
from computational_model import ComputationalModel
from scipy.spatial.distance import cdist

sim_data = SimulationData("input_files/simulation_1.json")
comp_model = ComputationalModel(sim_data)

# Time manager
tf = sim_data.final_t
dt = sim_data.delta_t
report_each_steps = sim_data.report_each_steps
schedule_values = np.linspace(0,tf,round(tf/(report_each_steps*dt))+1)
time_manager = pp.TimeManager(schedule=list(schedule_values), dt_init=dt, constant_dt=True)
comp_model.time_manager: pp.TimeManager = comp_model.params.get(
    "time_manager", time_manager
)
pp.run_time_dependent_model(comp_model, {"convergence_tol": 1e-5, "nl_convergence_tol": 1e-5})

file = open("visualization/overall_data.txt", "w")
n_dof = comp_model.equation_system.num_dofs()
grid_2d_data = comp_model.mdg.subdomains(dim=2)
grid_1d_data = comp_model.mdg.subdomains(dim=1)
file.write(n_dof.__str__())
file.write("\n")
file.write(grid_2d_data.__str__())
file.write(grid_1d_data.__str__())
file.close()

# Write last state
# Fetching variables related to the presence of fractures
# mdg = comp_model.mdg
# nd = comp_model.nd
#
# sd_2 = mdg.subdomains(dim=nd)[0]
# sd_1 = mdg.subdomains(dim=nd - 1)[0]
# intf = mdg.subdomain_pair_to_interface((sd_1, sd_2))
# d_m = mdg.interface_data(intf)
# d_1 = mdg.subdomain_data(sd_1)
#
# u_mortar = d_m[pp.STATE][comp_model.mortar_displacement_variable]
# contact_force = d_1[pp.STATE][comp_model.contact_traction_variable]
# fracture_pressure = d_1[pp.STATE][comp_model.scalar_variable]
#
# displacement_jump_global_coord = (
#     intf.mortar_to_secondary_avg(nd=nd)
#     * intf.sign_of_mortar_sides(nd=nd)
#     * u_mortar
# )
# projection = d_1["tangential_normal_projection"]
#
# project_to_local = projection.project_tangential_normal(int(intf.num_cells / 2))
# u_mortar_local = project_to_local * displacement_jump_global_coord
# u_mortar_local_decomposed = u_mortar_local.reshape((nd, -1), order="F")
#
# contact_force = contact_force.reshape((nd, -1), order="F")


