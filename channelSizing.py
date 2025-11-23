from nasaCEA import Engine
import numpy as np
import math
import pandas as pd
import scipy
from scipy.optimize import approx_fprime
from scipy.differentiate import derivative
import CoolProp.CoolProp as CP

class ChannelSizing:
    def __init__(self, engine: Engine, mdot=1.56, N=np.linspace(10, 15, 2), h=np.linspace(0.01, 0.1, 2), w=np.linspace(0.001, 0.1, 2), temp=298.15, pres=3.93e6):
        self.engine = engine
        self.mdot = mdot
        self.temp = temp
        self.pres = pres
        self.N_sweep = N
        self.h_sweep = h
        self.w_sweep = w
        self.engineProps = engine.engineProps
        self.L_ax = np.diff(np.append(engine.engineProps[:,0], 19.38949941 * 0.0254)) # takes difference between each axial station 
        self.channel_props = {k: [] for k in range(1, 31)}

    def axial_solver(self):
        T_local = self.temp
        P_local = self.pres
        A_previous = 3

        for row in range(29, -1, -1):
            #iter = 'iteration ' + str(30 - row)
            #print("Pressure and temp at start of station loop " + str(iter) + ": " + str(T_local) + ", " + str(P_local))

            # uses 'n-Dodecane', a similar fuel to RP1, for thermodynamic properties of coolant
            rho_p = CP.PropsSI('D', 'T', T_local, 'P', P_local, 'n-Dodecane') # density, used in pressure drop
            mu_p = CP.PropsSI('V', 'T', T_local, 'P', P_local, 'n-Dodecane') # dynamic viscosity, used in reynolds number 
            Cp_p = CP.PropsSI('C', 'T', T_local, 'P', P_local, 'n-Dodecane') # specific heat, used in dT
            cond_c_p = CP.PropsSI('CONDUCTIVITY', 'T', T_local, 'P', P_local, 'n-Dodecane') # thermal conductivity, used in wall temperature
            
            P_previous = P_local
            
            for Ncc in self.N_sweep:
                for height in self.h_sweep:
                    for width in self.w_sweep:
                        [T_hw, T_cw, q_rp, q_gas, q_wall, flux, dT, dP, A_previous] = newton_raphson(self.engineProps, Ncc, height, width, row, self.L_ax, T_local, mu_p, rho_p, cond_c_p, A_previous, Cp_p)
                        # print("pre pressure change: " + str(P_local))
                        
                        if Ncc == self.N_sweep[0] and height == self.h_sweep[0] and width == self.w_sweep[0]:
                            geom = Geometry(height, width, Ncc)
                            channel = Channel(T_hw, T_cw, q_rp, q_gas, q_wall, flux)
                            score_previous = dP / P_previous                            
                        else:
                            score_current = dP / P_previous

                            # keeps optimal geometry based on viable temperature, pressure drop, and viable aspect ratio
                            if T_hw < 800 and T_cw < 800 and score_current < score_previous and height / width < 7:
                                geom = Geometry(height, width, Ncc)
                                channel = Channel(T_hw, T_cw, q_rp, q_gas, q_wall, flux)
                                score_previous = score_current

            self.channel_props[row+1].append(geom)
            self.channel_props[row+1].append(channel) # append best geometry for station
            T_local += dT
            P_local += dP 
            # print("after pressure change of ideal geom: " + str(P_local))
            A_previous = geom.height * geom.width

def newton_raphson(data, Ncc, h, w, row, stationDistances, T_rp, mu, rho, lambda_rp, A_previous, Cp_p):
    gamma = data[row, 15] 
    M = data[row, 4]
    if row == 0:
        M = data[1, 4]
    gammaM = 1 + ((M ** 2) * (gamma - 1) / 2)
    D_t = 2 * data[19, 1]
    nu = data[row, 17] * 0.0001 # conversion from mP to Pa*s is 1 mP = 0.0001 Pa*s
    C_p = data[row, 14]
    Pr = data[row, 19]
    p_c = data[row, 8] * 100000 # conversion from bar to Pa is 1 bar = 100000 Pa
    c_star = 0.95
    radius = data[row, 1]
    R_u = 1.249569e+00 * 0.0254
    A = math.pi * radius * radius
    A_t = math.pi * 1.39 * 0.0254 * 1.39 * 0.0254 
    T = data[row, 9]
    T_s = T * gammaM
    h_gas_inter = 0.026 / (D_t ** 0.2) * (nu ** 0.2) * C_p / (Pr ** 0.6) * (p_c ** 0.8) / (c_star ** 0.8) * (D_t ** 0.1) / (R_u ** 0.1) * (A_t ** 0.9) / (A ** 0.9)
    def h_gas(T_hw):
        return h_gas_inter / ((((T_hw * gammaM / 2 / T_s) + 0.5) ** 0.68) * (gammaM ** 0.12))
    dx = stationDistances[row] * 0.0254
    A_gas = 2 * math.pi * radius * dx / Ncc
    T_aw = T_s / gammaM * (1 + ((Pr ** 0.33) * (M ** 2) * (gamma - 1) / 2))
    def q_gas(T_hw):
        return h_gas(T_hw) * A_gas * (T_aw - T_hw)
    
    # q_wall
    thickness = 0.003 * 0.0254
    k = 50 # thermal conductivity constant
    def T_cw(T_hw):
        return T_hw - (thickness * h_gas(T_hw) * (T_aw - T_hw) / k)
    q_wall_inter = k * A_gas / thickness
    def q_wall(T_hw):
        return q_wall_inter * (T_hw - T_cw(T_hw))
    
    # q_rp
    D_h = 2 * h * w / (w + h)
    l = D_h
    v = 1.56 / rho / h / w
    Re = rho * v * l / mu 
    f = get_friction(Re, D_h)

    nusselt = (f / 8) * (Re - 1000) * Pr / (1 + 12.7 * ((f / 8) ** 0.5) * (Pr ** (2 / 3) - 1))
    h_rp = nusselt / D_h * lambda_rp

    radius_of_curvature = math.sqrt(((thickness + radius) ** 2) + ((w / 2) ** 2))
    arc_length = 2 * radius_of_curvature * math.asin(w / 2 / radius_of_curvature)
    delta = ((2 * math.pi * (radius_of_curvature)) - (Ncc * arc_length)) / Ncc

    def m(T_hw):
        return math.sqrt(2 * h_rp / k / delta)
    L_c = h + (delta / 2)
    def nu_fin(T_hw):
        return (math.tanh(m(T_hw) * L_c)) / (m(T_hw) * L_c)
    def A_rp(T_hw):
        return ((2 * nu_fin(T_hw) * h) + w) * dx
    def q_rp(T_hw):
        return h_rp * A_rp(T_hw) * (T_cw(T_hw) - T_rp)

    def f1(T_hw):
        return q_rp(T_hw) - q_wall(T_hw)
    T_hw_val = 300 # initial guess
    eps = np.sqrt(np.finfo(float).eps)

    deriv = approx_fprime(np.array(T_hw_val), f1, eps) # first iteration
    T_hw_val = T_hw_val - (f1(T_hw_val) / deriv)
    while np.abs(f1(T_hw_val)) > 1e-5:
        f_val = f1(T_hw_val)
        deriv = approx_fprime(np.array(T_hw_val), f1, eps)
        T_hw_val = T_hw_val - (f_val / deriv)
    finalq_gas = q_gas(T_hw_val)
    finalq_rp = q_rp(T_hw_val)
    finalq_wall = q_wall(T_hw_val)

    flux = k / thickness * (np.abs(T_hw_val - T_cw(T_hw_val)))
    A_current = h * w
    dT = finalq_rp[0] / 1.56 * Ncc / Cp_p 
    v_local = 1.56 / (rho * A_current)  # mdot / (rho * total flow area)
    dP = f * (stationDistances[row]) / D_h * 0.5 * rho * (v_local**2)
    dP3 = 0 # just setting initial value
    if row == 29:
        A_previous = A_current
    if row != 0:
        dP3 = (2 / ((A_previous * Ncc) + (A_current * Ncc))) * ((1 / rho / A_previous / Ncc) - (1 / rho / A_current / Ncc)) * (1.56 ** 2)
        dP = dP + dP3
    T_cw_val = T_cw(T_hw_val)
    # print("dP1: " + str(dP) + ", dp3: " + str(dP3))

    return T_hw_val, T_cw_val, finalq_rp, finalq_gas, finalq_wall, flux, dT, dP, A_previous
    
# helper function to determine dary friction factor, taken from code snippets Kieran sent
def get_friction(Re, D_hyd): 
        def colebrook(Re, D_hyd):
            k = 0.005e-3  # Surface roughness in meters (5 microns)
            fric = 1 / ((-1.8 * np.log10((k / D_hyd / 3.7) ** 1.11 + 6.9 / Re)) ** 2)
            cole_diff = -2 * np.log10(k / D_hyd / 3.7 + 2.51 / Re / np.sqrt(fric)) - 1 / np.sqrt(fric)
            while np.abs(cole_diff) > 1e-5:
                deriv = -2 / (np.log(10) * (k / D_hyd / 3.7 + 2.51 / Re / np.sqrt(fric))) * \
                        (2.51 / Re * (-0.5 * fric ** (-1.5))) + 0.5 * fric ** (-1.5)
                fric = fric - cole_diff / deriv
                cole_diff = -2 * np.log10(k / D_hyd / 3.7 + 2.51 / Re / np.sqrt(fric)) - 1 / np.sqrt(fric)
            return fric

        return 64 / Re if Re <= 2320 else colebrook(Re, D_hyd)

class Geometry:
    def __init__(self, height=None, width=None, num_channels=None):
        self.height = height
        self.width = width
        self.num_channels = num_channels

class Channel:
    def __init__(self, T_hw, T_cw, q_RP, q_gas, q_wall, flux):
        self.T_hw = T_hw
        self.T_cw = T_cw
        self.q_RP = q_RP
        self.q_gas = q_gas
        self.q_wall = q_wall
        self.flux = flux
        self.diff = q_gas - q_RP 

class PressureDrop:
    def __init__(self, f, v, D_h, A_current):
        self.f = f
        self.v = v
        self.D_h = D_h
        self.A_current = A_current