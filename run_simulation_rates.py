import numpy as np
import porepy as pp
from simulation_data import SimulationData
from computational_model import ComputationalModel
from scipy.spatial.distance import cdist


verification_2_q = True
sim_data = SimulationData("input_files/verification_1.json")
if verification_2_q:
    sim_data = SimulationData("input_files/verification_2.json")

verbose_q = False
h_sizes = [0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125]

error_data = np.empty((0, 5), float)
for h_size in h_sizes:

    # update sizes
    sim_data.mesh_args["mesh_size_frac"] = h_size
    sim_data.mesh_args["mesh_size_min"] = h_size
    sim_data.mesh_args["mesh_size_bound"] = h_size

    # run the whole setting
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
    pp.run_time_dependent_model(comp_model, {"convergence_tol": 1e-6})


    def u_exact(x, y, t):
        # linear in time
        u_x = t * (1 - y) * y * np.sin(np.pi * x)
        u_y = t * (1 - x) * x * np.sin(np.pi * y)
        data = np.array([u_x,u_y])
        return data

    def p_exact(x, y, t):

        # linear in time
        p = t*np.sin(np.pi*x)*np.sin(np.pi*y)
        data = np.array([p])
        return data

    def sigma_exact(x, y, t):
        # linear in time
        sigma_xx = -3 * np.pi * t * (-1 + y) * y * np.cos(np.pi * x) - np.pi * t * (-1 + x) * x * np.cos(np.pi * y)
        sigma_yy = -(np.pi * t * (-1 + y) * y * np.cos(np.pi * x)) - 3 * np.pi * t * (-1 + x) * x * np.cos(np.pi * y)
        sigma_xy = t * (1 - 2 * y) * np.sin(np.pi * x) + t * (1 - 2 * x) * np.sin(np.pi * y)

        data = np.array([[sigma_xx,sigma_xy, 0.0],[sigma_xy,sigma_yy, 0.0],[0.0, 0.0, 0.0]])
        return data

    def q_exact(x, y, t):

        # linear in time
        q_x = -(np.pi*t*np.cos(np.pi*x)*np.sin(np.pi*y))
        q_y = -(np.pi*t*np.cos(np.pi*y)*np.sin(np.pi*x))
        data = np.array([q_x,q_y, 0.0])
        return data

    def sigma_star_exact(x, y, t):
        # linear in time
        sigma_xx = -2*np.pi*t*(-1 + y)*y*(1 + (x**2) + (y**2))*np.cos(np.pi*x) + (10 + (x**2) + (y**2))*(-(np.pi*t*(-1 + y)*y*np.cos(np.pi*x)) - np.pi*t*(-1 + x)*x*np.cos(np.pi*y))
        sigma_yy = -2*np.pi*t*(-1 + x)*x*(1 + (x**2) + (y**2))*np.cos(np.pi*y) + (10 + (x**2) + (y**2))*(-(np.pi*t*(-1 + y)*y*np.cos(np.pi*x)) - np.pi*t*(-1 + x)*x*np.cos(np.pi*y))
        sigma_xy = (1 + (x**2) + (y**2))*(t*(1 - 2*y)*np.sin(np.pi*x) + t*(1 - 2*x)*np.sin(np.pi*y))
        data = np.array([[sigma_xx,sigma_xy, 0.0],[sigma_xy,sigma_yy, 0.0],[0.0, 0.0, 0.0]])
        return data

    def q_star_exact(x, y, t):

        # linear in time
        q_x = -(np.pi*t*(1 + (x**2) + (y**2))*np.cos(np.pi*x)*np.sin(np.pi*y))
        q_y = -(np.pi*t*(1 + (x**2) + (y**2))*np.cos(np.pi*y)*np.sin(np.pi*x))
        data = np.array([q_x,q_y, 0.0])
        return data

    # At final time compute l2 errors for normal fluxes and potentials
    tv = tf + dt
    print("tf: ", tv)
    print("tf from time_man: ", comp_model.time_manager.time)

    for sd, data in comp_model.mdg.subdomains(return_data=True):

        # fetch primary variables
        p_h = data[pp.STATE][comp_model.scalar_variable]
        u_h = np.array(np.split(data[pp.STATE][comp_model.displacement_variable],p_h.shape[0])).T

        # compute fluxes
        q_n_h = comp_model.reconstruct_fluxes()

        # compute forces
        comp_model.reconstruct_normal_stress()
        data = comp_model.mdg.subdomain_data(sd)
        sigma_n_h = np.array(np.split(data[pp.STATE]["stress"],q_n_h.shape[0])).T

        sigma_n_e = np.zeros((comp_model.nd,q_n_h.shape[0]))
        q_n_e = np.zeros_like(q_n_h)
        normals = sd.face_normals.T
        for i, cell_xc in enumerate(sd.face_centers.T):
            n = normals[i]
            sigma_e = sigma_exact(cell_xc[0], cell_xc[1], tv)
            q_e = q_exact(cell_xc[0], cell_xc[1], tv)
            sigma_n_e_val = np.dot(sigma_e, n)
            if verification_2_q:
                sigma_e = sigma_star_exact(cell_xc[0], cell_xc[1], tv)
                q_e = q_star_exact(cell_xc[0], cell_xc[1], tv)

            sigma_n_e_val = np.dot(sigma_e, n)
            q_n_e[i] = np.dot(q_e,n)
            sigma_n_e[0, i] = sigma_n_e_val[0]
            sigma_n_e[1, i] = sigma_n_e_val[1]


        u_e = np.zeros((comp_model.nd,p_h.shape[0]))
        p_e = np.zeros_like(p_h)

        for i, cell_xc in enumerate(sd.cell_centers.T):
            u_val = u_exact(cell_xc[0], cell_xc[1], tv)
            p_val = p_exact(cell_xc[0], cell_xc[1], tv)
            u_e[0, i] = u_val[0]
            u_e[1, i] = u_val[1]
            p_e[i] = p_val[0]

        norm_u_e = np.sqrt(np.sum( np.sum(u_e * u_e, axis = 0) * sd.cell_volumes))
        norm_p_e = np.sqrt(np.sum(p_e * p_e * sd.cell_volumes))

        du = (u_h - u_e)
        dp = (p_h - p_e)
        error_u = np.sqrt(np.sum( np.sum(du * du, axis = 0) * sd.cell_volumes))/norm_u_e
        error_p = np.sqrt(np.sum(dp * dp * sd.cell_volumes))/norm_p_e

        norm_sigma_n_e = np.sqrt(np.sum( np.sum(sigma_n_e * sigma_n_e, axis = 0) * sd.face_areas))
        norm_q_n_e = np.sqrt(np.sum(q_n_e * q_n_e * sd.face_areas))

        dsigma_n = (sigma_n_h - sigma_n_e)
        dq_n = (q_n_h - q_n_e)
        error_sigma_n = np.sqrt(np.sum( np.sum(dsigma_n * dsigma_n, axis = 0) * sd.face_areas))/norm_sigma_n_e
        error_q_n = np.sqrt(np.sum(dq_n * dq_n * sd.face_areas))/norm_q_n_e

        if verbose_q:
            print("")
            print("Norm for u: ", norm_u_e)
            print("Norm for p: ", norm_p_e)
            print("Norm for sigma_n: ", norm_sigma_n_e)
            print("Norm for q_n: ", norm_q_n_e)

            print("")
            print("Relative L2-error for u: ",error_u)
            print("Relative L2-error for p: ", error_p)
            print("Relative L2-error for sigma_n: ",error_sigma_n)
            print("Relative L2-error for q_n: ", error_q_n)
            print("Setting complete.")

        error_data = np.append(error_data,np.array([[h_size, error_u, error_p, error_sigma_n, error_q_n]]), axis=0)

rates_data = np.empty((0, 4), float)
for i in range(error_data.shape[0] - 1):
    chunk_b = np.log(error_data[i])
    chunk_e = np.log(error_data[i + 1])
    h_step = chunk_e[0]-chunk_b[0]
    partial = (chunk_e - chunk_b) / h_step
    rates_data = np.append(rates_data,np.array([list(partial[1:5])]), axis=0)

print("error data: ", error_data)
print("error rates data: ", rates_data)

np.set_printoptions(precision=4)
print("rounded error data: ", error_data)
print("rounded error rates data: ", rates_data)





