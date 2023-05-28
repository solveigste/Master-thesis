# this file define a collection of patial properties
import numpy as np

################################################################
################### Functions for verification 1 ####################
################################################################

def kappa_verification_1(x, y, t):
    kappa = 1.0
    kappa = np.array([kappa])
    return kappa

def lame_parameters_verification_1(x, y, t):
    m_lambda = 1.0
    m_mu = 1.0
    data = np.array([m_lambda,m_mu])
    return data

def alpha_verification_1(x, y, t):
    alpha = 1.0
    data = np.array([alpha])
    return data

def f_mechanics_verification_1(x, y, t):
    # linear in time
    f_x = t*(-2*np.pi*(-1 + 2*x)*np.cos(np.pi*y) + (-2 + 3*(np.pi**2)*(-1 + y)*y)*np.sin(np.pi*x) - np.pi*np.cos(np.pi*x)*np.sin(np.pi*y))
    f_y = t*(-2*np.pi*(-1 + 2*y)*np.cos(np.pi*x) - np.pi*np.cos(np.pi*y)*np.sin(np.pi*x) + (-2 + 3*(np.pi**2)*(-1 + x)*x)*np.sin(np.pi*y))
    data = np.array([f_x, f_y])
    return data

def f_fluid_verification_1(x, y, t):
    # linear in time
    f = -(np.pi*(-1 + y)*y*np.cos(np.pi*x)) - np.pi*(-1 + x)*x*np.cos(np.pi*y) +  (1 + 2*(np.pi**2)*t)*np.sin(np.pi*x)*np.sin(np.pi*y)
    data = np.array([f])
    return data

def u_bc_verification_1(x, y, t):
    u_x = 0.0
    u_y = 0.0
    data = np.array([u_x,u_y])
    return data

def sigma_n_bc_verification_1(x, y, t):
    sigma_n_x = 0.0
    sigma_n_y = 0.0
    data = np.array([sigma_n_x,sigma_n_y])
    return data

def p_bc_verification_1(x, y, t):
    p = 0.0
    data = np.array([p])
    return data

def q_n_bc_verification_1(x, y, t):
    q_n = 0.0
    data = np.array([q_n])
    return data

def u_ic_verification_1(x, y):
    u_x = 0.0
    u_y = 0.0
    data = np.array([u_x,u_y])
    return data

def p_ic_verification_1(x, y):
    p = 0.0
    data = np.array([p])
    return data

################################################################
################### Functions for verification 2 ####################
################################################################

def kappa_verification_2(x, y, t):
    kappa = x**2 + y**2 + 1.0
    kappa = np.array([kappa])
    return kappa

def lame_parameters_verification_2(x, y, t):
    m_lambda = x**2 + y**2 + 10.0
    m_mu = x**2 + y**2 + 1.0
    data = np.array([m_lambda,m_mu])
    return data

def alpha_verification_2(x, y, t):
    alpha = (x**2 + y**2 + 1.0)/4.0
    data = np.array([alpha])
    return data

def f_mechanics_verification_2(x, y, t):
    # linear in time
    f_x = (1 + (x**2) + (y**2))*(np.pi*t*(1 - 2*x)*np.cos(np.pi*y) - 2*t*np.sin(np.pi*x)) + 2*y*((t - 2*t*y)*np.sin(np.pi*x) + t*(1 - 2*x)*np.sin(np.pi*y)) + np.pi*t*((10 + 3*(x**2) - 4*(x**3) + (y**2) - 2*x*(10 + (y**2)))*np.cos(np.pi*y) + 3*np.pi*(-1 + y)*y*(4 + (x**2) + (y**2))*np.sin(np.pi*x) - np.cos(np.pi*x)*(6*x*(-1 + y)*y + np.sin(np.pi*y)))
    f_y = (1 + (x**2) + (y**2))*(np.pi*(t - 2*t*y)*np.cos(np.pi*x) - 2*t*np.sin(np.pi*y)) + 2*x*((t - 2*t*y)*np.sin(np.pi*x) + t*(1 - 2*x)*np.sin(np.pi*y)) + np.pi*t*((10 + (x**2)*(1 - 2*y) - 20*y + 3*(y**2) - 4*(y**3))*np.cos(np.pi*x) - np.cos(np.pi*y)*(6*(-1 + x)*x*y + np.sin(np.pi*x)) + 3*np.pi*(-1 + x)*x*(4 + (x**2) + (y**2))*np.sin(np.pi*y))

    f_x = -4*np.pi*t*x*(-1 + y)*y*np.cos(np.pi*x) + 2*np.pi*t*x*(-((-1 + y)*y*np.cos(np.pi*x)) - (-1 + x)*x*np.cos(np.pi*y)) + 2*(np.pi**2)*t*(-1 + y)*y*(1 + (x**2) + (y**2))*np.sin(np.pi*x) + (1 + (x**2) + (y**2))*(np.pi*t*(1 - 2*x)*np.cos(np.pi*y) - 2*t*np.sin(np.pi*x)) + np.pi*t*(10 + (x**2) + (y**2))*((1 - 2*x)*np.cos(np.pi*y) + np.pi*(-1 + y)*y*np.sin(np.pi*x)) - (np.pi*t*(1 + (x**2) + (y**2))*np.cos(np.pi*x)*np.sin(np.pi*y))/4. - (t*x*np.sin(np.pi*x)*np.sin(np.pi*y))/2. + 2*y*((t - 2*t*y)*np.sin(np.pi*x) + t*(1 - 2*x)*np.sin(np.pi*y))
    f_y = (t*(-4*np.pi*(-11 + 22*y - 4*(y**2) + 6*(y**3) + (x**2)*(-2 + 4*y))*np.cos(np.pi*x) - np.pi*np.cos(np.pi*y)*(24*(-1 + x)*x*y + (1 + (x**2) + (y**2))*np.sin(np.pi*x)) + 4*(-3*(np.pi**2)*(x**3) + 3*(np.pi**2)*(x**4) - 2*(1 + (y**2)) + x*(2 - 3*(np.pi**2)*(4 + (y**2))) + 3*(x**2)*(-2 + (np.pi**2)*(4 + (y**2))))*np.sin(np.pi*y) - 2*np.sin(np.pi*x)*(-4*x + 8*x*y + y*np.sin(np.pi*y))))/4.
    data = np.array([f_x, f_y])
    return data

def f_fluid_verification_2(x, y, t):
    # linear in time
    f = -(np.pi*np.cos(np.pi*y)*((-1 + x)*x + 2*t*y*np.sin(np.pi*x))) +(1 + 2*(np.pi**2)*t*(1 + (x**2) + (y**2)))*np.sin(np.pi*x)*np.sin(np.pi*y) -np.pi*np.cos(np.pi*x)*((-1 + y)*y + 2*t*x*np.sin(np.pi*y))

    f = -0.25*(np.pi*(1 + (x**2) + (y**2))*((-1 + y)*y*np.cos(np.pi*x) + (-1 + x)*x*np.cos(np.pi*y))) - 2*np.pi*t*y*np.cos(np.pi*y)*np.sin(np.pi*x) - 2*np.pi*t*x*np.cos(np.pi*x)*np.sin(np.pi*y) + np.sin(np.pi*x)*np.sin(np.pi*y) + 2*(np.pi**2)*t*(1 + (x**2) + (y**2))*np.sin(np.pi*x)*np.sin(np.pi*y)
    data = np.array([f])
    return data

def u_bc_verification_2(x, y, t):
    u_x = 0.0
    u_y = 0.0
    data = np.array([u_x,u_y])
    return data

def sigma_n_bc_verification_2(x, y, t):
    sigma_n_x = 0.0
    sigma_n_y = 0.0
    data = np.array([sigma_n_x,sigma_n_y])
    return data

def p_bc_verification_2(x, y, t):
    p = 0.0
    data = np.array([p])
    return data

def q_n_bc_verification_2(x, y, t):
    q_n = 0.0
    data = np.array([q_n])
    return data

def u_ic_verification_2(x, y):
    u_x = 0.0
    u_y = 0.0
    data = np.array([u_x,u_y])
    return data

def p_ic_verification_2(x, y):
    p = 0.0
    data = np.array([p])
    return data

################################################################
################ Functions for simulation 1 ####################
################################################################

def kappa_simulation_1(x, y, t):
    kappa = 1e-13
    kappa = np.array([kappa])
    return kappa

def lame_parameters_simulation_1(x, y, t):
    m_lambda = 10.0e9
    m_mu = 1.0e9
    data = np.array([m_lambda,m_mu])
    return data

def alpha_simulation_1(x, y, t):
    alpha = 1.0
    data = np.array([alpha])
    return data

def f_mechanics_simulation_1(x, y, t):
    f_x = 0.0
    f_y = 0.0
    data = np.array([f_x, f_y])
    return data

def f_fluid_simulation_1(x, y, t):
    # This could represent a well injecting fluid
    f = 0.0
    data = np.array([f])
    return data

def u_bc_simulation_1(x, y, t):
    u_x = 0.0
    u_y = 0.0
    data = np.array([u_x,u_y])
    return data

def sigma_n_bc_simulation_1(x, y, t):
    sigma_n_x = 0.0
    sigma_n_y = -10.0e6
    data = np.array([sigma_n_x,sigma_n_y])
    return data

def p_bc_simulation_1(x, y, t):
    p = 0.0
    data = np.array([p])
    return data

def q_n_bc_simulation_1(x, y, t):
    q_n = 0.0
    data = np.array([q_n])
    return data

def u_ic_simulation_1(x, y):
    u_x = 0.0
    u_y = 0.0
    data = np.array([u_x,u_y])
    return data

def p_ic_simulation_1(x, y):
    p = 10.0e6
    data = np.array([p])
    return data


################################################################
################ Functions for simulation 2 ####################
################################################################

def kappa_simulation_2(x, y, t):
    kappa = 1e-13
    kappa = np.array([kappa])
    return kappa

def lame_parameters_simulation_2(x, y, t):
    m_lambda = 10.0e9
    m_mu = 1.0e9
    data = np.array([m_lambda,m_mu])
    return data

def alpha_simulation_2(x, y, t):
    alpha = 1.0
    data = np.array([alpha])
    return data

def f_mechanics_simulation_2(x, y, t):
    f_x = 0.0
    f_y = 0.0
    data = np.array([f_x, f_y])
    return data

def f_fluid_simulation_2(x, y, t):
    # This could represent a well injecting fluid
    f = 0.0
    data = np.array([f])
    return data

def u_bc_simulation_2(x, y, t):
    u_x = 0.0
    u_y = 0.0
    data = np.array([u_x,u_y])
    return data

def sigma_n_bc_simulation_2(x, y, t):
    sigma_n_x = 0.0
    sigma_n_y = -60.0e6
    data = np.array([sigma_n_x,sigma_n_y])
    return data

def p_bc_simulation_2(x, y, t):
    p = 0.01 * t * (30e6 * (100 - x)) + 15e6
    if p > 40e6:
        p = 40e6
    data = np.array([p])
    return data

def q_n_bc_simulation_2(x, y, t):
    q_n = 0.0
    data = np.array([q_n])
    return data

def u_ic_simulation_2(x, y):
    u_x = 0.0
    u_y = 0.0
    data = np.array([u_x,u_y])
    return data

def p_ic_simulation_2(x, y):
    p = 15.0e6
    data = np.array([p])
    return data


################################################################
################ Functions for simulation 3 ####################
################################################################

def kappa_simulation_3(x, y, t):
    kappa = 1e-13
    kappa = np.array([kappa])
    return kappa

def lame_parameters_simulation_3(x, y, t):
    m_lambda = 10.0e9
    m_mu = 1.0e9
    data = np.array([m_lambda,m_mu])
    return data

def alpha_simulation_3(x, y, t):
    alpha = 1.0
    data = np.array([alpha])
    return data

def f_mechanics_simulation_3(x, y, t):
    f_x = 0.0
    f_y = 0.0
    data = np.array([f_x, f_y])
    return data

def f_fluid_simulation_3(x, y, t):
    # This could represent a well injecting fluid
    f = 0.0
    data = np.array([f])
    return data

def u_bc_simulation_3(x, y, t):
    u_x = 0.0
    u_y = 0.0
    data = np.array([u_x,u_y])
    return data

def sigma_n_bc_simulation_3(x, y, t):
    sigma_n_x = 0.0
    sigma_n_y = -60.0e6
    data = np.array([sigma_n_x,sigma_n_y])
    return data

def p_bc_simulation_3(x, y, t):
    p = 0.01 * t * (30e6 * (100 - x)) + 15e6
    if p > 40e6:
        p = 40e6
    data = np.array([p])
    return data

def q_n_bc_simulation_3(x, y, t):
    q_n = 0.0
    data = np.array([q_n])
    return data

def u_ic_simulation_3(x, y):
    u_x = 0.0
    u_y = 0.0
    data = np.array([u_x,u_y])
    return data

def p_ic_simulation_3(x, y):
    p = 15.0e6
    data = np.array([p])
    return data