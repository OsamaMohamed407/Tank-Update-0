import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import fsolve
import time
import math
import base64
import re
from pathlib import Path
import random

# =============================================================================
# ===== (CODE 1) CONSTANTS AND PHYSICS ========================================
# =============================================================================
st.set_page_config(page_title="Integrated Urea Plant Simulation", layout="wide", initial_sidebar_state="expanded")

# --- (CODE 1) CONSTANTS ---
DT = 0.1 # Unified Time Step
A_HEAT_TRANSFER_AREA_PRE_EVAP = 316.0
SHELL_VOL_M3_PRE_EVAP = 0.70
TUBE_VOL_M3_PRE_EVAP = 1.79
FLASH_VESSEL_VOL_M3_PRE_EVAP = 23.0
A_BIURET_PRE_EVAP = 3.9e+7
EA_BIURET_PRE_EVAP = 128000.0
MASTER_P_EVAPORATOR_PRE_EVAP = 3.80
STREAM1_CONC = 73.00
STREAM1_BIURET = 0.39
STREAM2_CONC = 40.00
STREAM2_BIURET = 0.45
R_GAS_IDEAL = 8.314 # Shared

# --- (CODE 3) CONSTANTS (Melt Tank System) ---
MELT_TANK_HEIGHT_M = 9.0
MELT_TANK_DIAMETER_M = 0.200 # 200 mm as requested
MELT_TANK_AREA_M2 = math.pi * ((MELT_TANK_DIAMETER_M / 2) ** 2)
MELT_TANK_VOL_M3 = MELT_TANK_AREA_M2 * MELT_TANK_HEIGHT_M
MELT_PUMP_MAX_CAPACITY = 110.0
DRAIN_VALVE_DIAMETER_IN = 1.0
DRAIN_AREA_M2 = math.pi * (((DRAIN_VALVE_DIAMETER_IN * 0.0254) / 2) ** 2)

# =============================================================================
# ===== (CODE 2) CONSTANTS AND PHYSICS (1st & 2nd Evaporator) =================
# =============================================================================
STEAM_PRESSURE_BAR_EVAP = np.array([0.0, 1.0, 2.0, 3.0, 3.8, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
STEAM_TEMP_C_EVAP = np.array([100.0, 120.4, 133.7, 143.7, 150.4, 152.0, 158.9, 165.1, 170.4, 175.4, 179.9])
M_steam_EVAP = 18.015; M_urea_EVAP = 60.06; M_BIURET_EVAP = 103.08
kgph_to_kgps_EVAP = 1.0 / 3600
A_antoine_EVAP, B_antoine_EVAP, C_antoine_EVAP = 8.07131, 1730.63, 233.426
M_STEAM_KG_MOL_EVAP = 0.018015
R_STEAM_SPECIFIC_EVAP = R_GAS_IDEAL / M_STEAM_KG_MOL_EVAP

# *** U-VALUE (DESIGN BASE) - Shared Default unless overridden ***
U_HEAT_TRANSFER_COEFF_EVAP_DEFAULT = 1400.0 
A_HEAT_TRANSFER_AREA_EVAP_DEFAULT = 510.0
shell_vol_m3_EVAP_DEFAULT = 4.1
tube_vol_m3_EVAP_DEFAULT = 2.5
FLASH_VESSEL_VOL_M3_EVAP_DEFAULT = 76.4
Ea_BIURET_EVAP_DEFAULT = 128000.0
A_BIURET_EVAP_DEFAULT = 3.90e+7
R_GAS_EVAP = 8.314

# --- Pipe Delay Constants ---
PIPE_LENGTH_M = 15.0
PIPE_DIAMETER_IN = 6.0
PIPE_RADIUS_M = (PIPE_DIAMETER_IN * 0.0254) / 2
PIPE_AREA_M2 = math.pi * (PIPE_RADIUS_M ** 2)
PIPE_VOL_M3 = PIPE_AREA_M2 * PIPE_LENGTH_M 

# =============================================================================
# ===== (CODE 1) HELPER FUNCTIONS & MODELS (Pre-evaporator) ===================
# =============================================================================
def calculate_U_value_pre_evap(flow_rate_kgps):
    U_design = 1533.0; flow_design_kgps = 111727 / 3600
    if flow_design_kgps > 0:
        flow_ratio = flow_rate_kgps / flow_design_kgps
        U_current = U_design * (flow_ratio ** 0.8)
    else: U_current = U_design / 2
    return max(500, U_current)

def get_steam_properties_pre_evap(pressure_barg):
    STEAM_PRESSURE_BAR = np.array([0.0, 1.0, 2.0, 3.0, 3.8, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    STEAM_TEMP_C = np.array([100.0, 120.4, 133.7, 143.7, 150.4, 152.0, 158.9, 165.1, 170.4, 175.4, 179.9])
    temp_c = np.interp(pressure_barg, STEAM_PRESSURE_BAR, STEAM_TEMP_C)
    latent_heat_j = (2501 - 2.361 * temp_c) * 1000
    return temp_c, latent_heat_j

def get_steam_density_pre_evap(pressure_barg, temp_c):
    M_STEAM_KG_MOL = 0.018015; R_STEAM_SPECIFIC = R_GAS_IDEAL / M_STEAM_KG_MOL
    P_abs_pa = (pressure_barg + 1.01325) * 101325
    T_k = temp_c + 273.15
    if T_k <= 0: return 0.6
    return P_abs_pa / (R_STEAM_SPECIFIC * T_k)

def rho_from_conc_pre_evap(conc): return np.interp(conc, [0,40,80,98.5], [998,1135,1174.04,1195])
def cp_from_conc_pre_evap(conc): return np.interp(conc, [0,40,77,98.5], [4.18,3.60,3.20,2.20])

def conc_from_TP_evap_pre_evap(temp_C, P_barg, guess_conc_wt=80.0):
    M_urea=60.06; M_steam=18.015; A_antoine=8.07131; B_antoine=1730.63; C_antoine=233.426
    P_sys = (P_barg + 1.01325) * 750.062
    if P_sys <= 0: return guess_conc_wt
    wu = guess_conc_wt/100
    mu = wu/M_urea; mw = (1-wu)/M_steam; xw_ref = mw/(mu+mw)
    def residual(xw):
        xw = np.clip(xw,1e-5,0.9999); Psat = 10**(A_antoine - B_antoine/(C_antoine+temp_C))
        return xw*Psat - P_sys
    try: xw = np.clip(fsolve(residual,xw_ref)[0],1e-5,0.9999)
    except: return guess_conc_wt
    return ((1-xw)*M_urea)/((1-xw)*M_urea+xw*M_steam)*100

def compute_steam_flow_kgps_pre_evap(pct, upstream_pressure_barg, downstream_pressure_barg):
    pct_points = np.array([0, 33.4, 54.7, 70.5, 100.0])
    base_flow_factor_points = np.array([0, 1136, 2402, 3793, 4500])
    flow_factor = np.interp(pct, pct_points, base_flow_factor_points)
    delta_p = (upstream_pressure_barg - downstream_pressure_barg)
    if delta_p <= 0: return 0.0
    flow_kgph = (flow_factor * np.sqrt(delta_p)) * 2
    return flow_kgph / 3600

def calculate_shell_pressure_ideal_gas_pre_evap(steam_flow_in_kgps, vapor_space_vol_m3, shell_temp_c, q_transferred_kw, latent_heat_j_kg, previous_steam_mass_kg, dt_s):
    M_STEAM_KG_MOL = 0.018015; R_STEAM_SPECIFIC = R_GAS_IDEAL / M_STEAM_KG_MOL
    shell_temp_k = shell_temp_c + 273.15; mass_in = steam_flow_in_kgps * dt_s
    mass_condensed_per_sec = (q_transferred_kw * 1000) / latent_heat_j_kg if latent_heat_j_kg > 0 else 0
    mass_out = mass_condensed_per_sec * dt_s
    mass_loss_venting_kgps = 0.0
    mass_loss = mass_loss_venting_kgps * dt_s
    current_steam_mass_kg = max(0.0, previous_steam_mass_kg + mass_in - mass_out - mass_loss)
    if current_steam_mass_kg <= 0.01: return -1.0, 0.0
    pressure_abs_pa = (current_steam_mass_kg * R_STEAM_SPECIFIC * shell_temp_k) / max(0.01, vapor_space_vol_m3)
    pressure_barg = (pressure_abs_pa / 101325.0) - 1.0
    return pressure_barg, current_steam_mass_kg

def calculate_vacuum_pressure_tuned_pre_evap(vac_valve_pct, current_evap_kgph):
    base_vac_at_100_pct = -0.62
    base_vacuum = np.interp(vac_valve_pct, [0, 60, 100], [0, -0.58, base_vac_at_100_pct])
    load_points = [0, 13242, 19000]
    vacuum_at_load_points = [-0.62, -0.51, -0.25]
    penalty = np.interp(current_evap_kgph, load_points, (np.array(vacuum_at_load_points) - base_vac_at_100_pct))
    final_vacuum = base_vacuum + penalty
    return np.clip(final_vacuum, -0.98, 2.0)

def solve_for_T_enhanced_pre_evap(Q_effective_kW, T0, cp0, m_gps, m_urea_gps, P_vac, feed_conc, effective_steam_temp_c):
    if feed_conc < 0.1: return T0
    max_temp = min(effective_steam_temp_c - 2, 150); initial_guess = min(T0, max_temp - 5)
    def energy_balance(TT):
        if TT > max_temp: return (TT - max_temp) * 1e5
        conc = conc_from_TP_evap_pre_evap(TT, P_vac, feed_conc)
        if conc < feed_conc: conc = feed_conc
        cp1 = cp_from_conc_pre_evap(conc)*1000; cp_avg = (cp0+cp1)/2
        m_out = m_urea_gps/(conc/100) if conc > 0 else m_gps
        m_evap = max(0, m_gps - m_out)
        Q_sens = m_gps*cp_avg*(TT-T0); Q_evap = m_evap*2260*1000
        return Q_effective_kW*1000 - (Q_sens+Q_evap)
    try: return min(fsolve(energy_balance, initial_guess)[0], max_temp)
    except: return initial_guess

def calculate_streams_corrected_pre_evap(m_gps, m_urea_gps, Tprod, Cprod):
    s14_mass = m_gps*3600
    if Cprod > 0 and Cprod > (m_urea_gps/m_gps * 100 if m_gps > 0 else 0):
        s15_mass = (m_urea_gps / (Cprod/100)) * 3600
    else: s15_mass = s14_mass
    m_evap_kg_h = s14_mass - s15_mass
    return {'s15': {'mass': max(0, s15_mass)}, 's16': {'mass': max(0, m_evap_kg_h)}}

class PID_pre_evap:
    def __init__(self, Kp, Ki, Kd, setpoint, output_limits=(0, 100)):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd; self.setpoint = setpoint
        self.min_output, self.max_output = output_limits; self.integral = 0; self.last_error = 0
    def initialize(self, initial_output):
        if self.Ki>0: self.integral = initial_output/self.Ki
        else: self.integral = 0.0
        self.last_error=0
    def update(self, pv, dt):
        error = self.setpoint - pv; self.integral += error * dt
        if self.Ki>0: self.integral=np.clip(self.integral,self.min_output/self.Ki,self.max_output/self.Ki)
        derivative = (error-self.last_error)/dt if dt>0 else 0.0
        output = self.Kp*error + self.Ki*self.integral + self.Kd*derivative; self.last_error = error
        return np.clip(output, self.min_output, self.max_output)

def calculate_biuret_kinetic_pre_evap(initial_biuret_total_pct, feed_conc_pct, product_temp_c, product_conc_pct, residence_time_s):
    M_urea=60.06; M_BIURET=103.08
    if product_conc_pct < 1.0 or residence_time_s <= 0: return initial_biuret_total_pct
    biuret_to_solids_ratio = initial_biuret_total_pct/feed_conc_pct if feed_conc_pct>0 else 0
    conc_initial_biuret_pct = biuret_to_solids_ratio * product_conc_pct
    urea_in_product_pct = product_conc_pct - conc_initial_biuret_pct
    if urea_in_product_pct <= 0: return conc_initial_biuret_pct
    urea_mass_frac = urea_in_product_pct/100.0; rho_prod_kg_m3 = rho_from_conc_pre_evap(product_conc_pct)
    urea_conc_mol_m3 = (rho_prod_kg_m3 * urea_mass_frac) / (M_urea/1000.0)
    T_kelvin = product_temp_c + 273.15
    k = A_BIURET_PRE_EVAP * np.exp(-EA_BIURET_PRE_EVAP / (R_GAS_IDEAL * T_kelvin))
    rate_biuret_formation = 0.5 * k * (urea_conc_mol_m3 ** 2)
    increase_biuret_mol_m3 = rate_biuret_formation * residence_time_s
    increase_biuret_kg_m3 = increase_biuret_mol_m3 * (M_BIURET / 1000.0)
    newly_gen_biuret_pct = (increase_biuret_kg_m3 / rho_prod_kg_m3) * 100.0
    return max(conc_initial_biuret_pct, conc_initial_biuret_pct + newly_gen_biuret_pct)

# =============================================================================
# ===== (CODE 2) HELPER FUNCTIONS & MODELS (Shared for Evap 1 & 2) ============
# =============================================================================

# *** NEW: PHYSICS ENGINE FUNCTIONS ADDED HERE ***
def update_valve_physics(target_pos, current_pos, stroke_time_sec, dt):
    """Simulates realistic valve movement with stroke time."""
    if stroke_time_sec <= 0.1: return target_pos
    max_step = (100.0 / stroke_time_sec) * dt
    delta = target_pos - current_pos
    if abs(delta) <= max_step:
        return target_pos
    else:
        return current_pos + math.copysign(max_step, delta)

def update_pipe_lag(current_props, target_props, pipe_vol_m3, dt_s, decay_time_s=5.0):
    """Simulates plug flow pipe lag including decay on stop."""
    vol_flow_in = target_props.get('vol_flow_m3ph', 0.0)
    vol_flow_m3ps = vol_flow_in / 3600.0
    if vol_flow_m3ps > 1e-5:
        tau = pipe_vol_m3 / vol_flow_m3ps
    else:
        tau = decay_time_s 
    new_props = {}
    keys_to_lag = ['mass_flow_kgph', 'temp_c', 'conc_pct', 'biuret_pct', 'rho_kg_m3', 'vol_flow_m3ph']
    for key in keys_to_lag:
        curr_val = current_props.get(key, 0.0)
        targ_val = target_props.get(key, 0.0)
        change_rate = (targ_val - curr_val) / (tau + 1e-9)
        new_props[key] = curr_val + change_rate * dt_s
    return new_props

def update_mixing_lag(current_val, target_val, volume_m3, flow_m3ph, dt_s, min_residence_time=1.0):
    """Simulates CSTR mixing lag."""
    if volume_m3 <= 0: return target_val
    flow_m3ps = flow_m3ph / 3600.0
    if flow_m3ps > 1e-5:
        residence_time = volume_m3 / flow_m3ps
    else:
        residence_time = 3600.0 # Very slow change if no flow
    tau = max(residence_time, min_residence_time)
    change = (target_val - current_val) / tau * dt_s
    return current_val + change

def get_steam_properties_evap(pressure_barg):
    temp_c = np.interp(pressure_barg, STEAM_PRESSURE_BAR_EVAP, STEAM_TEMP_C_EVAP)
    latent_heat_j = (2501 - 2.361 * temp_c) * 1000
    return temp_c, latent_heat_j

def get_steam_density_evap(pressure_barg, temp_c):
    P_abs_pa = (pressure_barg + 1.01325) * 101325
    T_k = temp_c + 273.15
    if T_k <= 0: return 0.6
    return P_abs_pa / (R_STEAM_SPECIFIC_EVAP * T_k)

def rho_from_conc_evap(conc): return np.interp(conc, [0,40,80,98.5], [998,1135,1174.04,1195])
def cp_from_conc_evap(conc): return np.interp(conc, [0,40,77,98.5], [4.18,3.60,3.20,2.20])

def conc_from_TP_evap_main(temp_C, P_barg, guess_conc_wt=80.0):
    P_sys = (P_barg + 1.01325) * 750.062; wu = guess_conc_wt/100
    mu = wu/M_urea_EVAP; mw = (1-wu)/M_steam_EVAP; xw_ref = mw/(mu+mw)
    def residual(xw):
        xw = np.clip(xw,1e-5,0.9999); Psat = 10**(A_antoine_EVAP - B_antoine_EVAP/(C_antoine_EVAP+temp_C))
        return xw*Psat - P_sys
    try: xw = np.clip(fsolve(residual,xw_ref)[0],1e-5,0.9999)
    except: return guess_conc_wt
    return ((1-xw)*M_urea_EVAP)/((1-xw)*M_urea_EVAP+xw*M_steam_EVAP)*100

def compute_steam_flow_kgps_evap(pct, max_flow_kgph, upstream_p_barg, downstream_p_barg):
    pct_points = np.array([0, 20, 40, 60, 80, 100])
    base_flow_points = np.array([0, 0.08, 0.20, 0.40, 0.70, 1.0]) 
    flow_factor = np.interp(pct, pct_points, base_flow_points)
    nominal_flow_kgph = flow_factor * max_flow_kgph
    delta_p = upstream_p_barg - downstream_p_barg
    if delta_p <= 0: return 0.0
    design_delta_p = 0.2
    pressure_factor = np.sqrt(delta_p / design_delta_p)
    corrected_flow_kgph = nominal_flow_kgph * pressure_factor
    return corrected_flow_kgph * kgph_to_kgps_EVAP

def calculate_shell_physics_evap(steam_flow_in_kgps, vapor_space_vol_m3, current_P_shell_barg, T_product_out_c, T_product_in_c, U_val, Area, dt_s):
    T_steam_c, Lh_steam_j_kg = get_steam_properties_evap(current_P_shell_barg)
    dT1 = T_steam_c - T_product_out_c 
    dT2 = T_steam_c - T_product_in_c  
    if dT1 <= 0 or dT2 <= 0: LMTD = 0.0
    elif abs(dT1 - dT2) < 0.1: LMTD = dT1
    else: LMTD = (dT1 - dT2) / np.log(dT1 / dT2)

    if LMTD > 0 and Lh_steam_j_kg > 0:
        q_transferred_kw = (U_val * Area * LMTD) / 1000.0
        mass_condensed_kgps = (q_transferred_kw * 1000.0) / Lh_steam_j_kg
    else:
        q_transferred_kw = 0.0; mass_condensed_kgps = 0.0

    M_STEAM_KG_MOL = 0.018015; R_STEAM_SPECIFIC = 8.314 / M_STEAM_KG_MOL
    T_kelvin = T_steam_c + 273.15
    P_abs_pa_current = (current_P_shell_barg + 1.01325) * 101325.0
    current_steam_mass_kg = (P_abs_pa_current * vapor_space_vol_m3) / (R_STEAM_SPECIFIC * T_kelvin)
    mass_in = steam_flow_in_kgps * dt_s
    mass_out_condensed = mass_condensed_kgps * dt_s
    mass_vent = 0.005 * current_steam_mass_kg * dt_s 
    new_steam_mass_kg = max(0.001, current_steam_mass_kg + mass_in - mass_out_condensed - mass_vent)
    new_P_abs_pa = (new_steam_mass_kg * R_STEAM_SPECIFIC * T_kelvin) / vapor_space_vol_m3
    new_P_barg = (new_P_abs_pa / 101325.0) - 1.0
    return new_P_barg, new_steam_mass_kg, q_transferred_kw

def calculate_vacuum_pressure_evap(vac_valve_pct, master_pressure, current_evap_kgph, params):
    base_vacuum_ideal = np.interp(
        vac_valve_pct, 
        [0, params['vp1_pct'], params['vp2_pct'], 100.0], 
        [0.0, params['vp1_vac'], params['vp2_vac'], params['vp3_vac']]
    )
    pressure_scaling_factor = master_pressure / params['master_p_norm'] if params['master_p_norm'] > 0 else 1.0
    scaled_ideal_vacuum = base_vacuum_ideal * pressure_scaling_factor
    vapor_load_penalty = (current_evap_kgph / params['evap_norm']) * params['penalty_factor'] if params['evap_norm'] > 0 else 0.0
    final_pressure = scaled_ideal_vacuum + vapor_load_penalty
    return max(params['min_vac'], min(final_pressure, params['max_vac']))

# *** UPDATED FUNCTION: SPECIAL VACUUM PHYSICS FOR EVAP 2 WITH PRESSURE LINK ***
def calculate_vacuum_pressure_evap2_special(vac_valve_pct, current_evap_kgph, master_pressure):
    # 1. Base Curve
    base_curve_pct = [0, 30, 40, 100]
    base_curve_vac = [0, -0.92, -0.95, -0.96]
    base_vac = np.interp(vac_valve_pct, base_curve_pct, base_curve_vac)
    
    # 2. Scaling by Main Steam Pressure (The new requirement)
    # Linked to the pressure that Evap 1 uses (master_pressure)
    # Base values achieved at 3.8 barg.
    scaling_factor = master_pressure / 3.8 if 3.8 > 0 else 1.0
    base_vac_scaled = base_vac * scaling_factor

    # 3. Variable Penalty Factor
    penalty_curve_pct = [0, 34, 40, 100]
    penalty_curve_val = [0, 0.05, 0.03, 0.01] 
    penalty_at_ref_load = np.interp(vac_valve_pct, penalty_curve_pct, penalty_curve_val)
    
    # 4. Load Penalty
    ref_evap_load = 3653.0
    penalty_exponent = 2.26
    
    if ref_evap_load > 0:
        load_ratio = current_evap_kgph / ref_evap_load
    else:
        load_ratio = 0
        
    actual_penalty = penalty_at_ref_load * (load_ratio ** penalty_exponent)
    
    final_pressure = base_vac_scaled + actual_penalty 
    return np.clip(final_pressure, -0.99, 0.5)

def calculate_streams_corrected_evap_generic(m_gps, m_urea_gps, Tprod, Cprod, T0, C0, Q_available_kW):
    s_in_mass = m_gps * 3600
    if Cprod > C0 and Cprod > 0:
        s_out_mass = s_in_mass * (C0 / Cprod)
    else: s_out_mass = s_in_mass
    m_evap_kg_h = s_in_mass - s_out_mass
    s_vapor_mass = max(0, m_evap_kg_h)
    return { 'mass_in': s_in_mass, 'mass_out': max(0, s_out_mass), 'mass_vapor': s_vapor_mass, 'temp_out': Tprod, 'conc_out': Cprod}

class PID_evap_generic: 
    def __init__(self, Kp, Ki, Kd, setpoint, output_limits=(0, 100)):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd; self.setpoint = setpoint
        self.min_output, self.max_output = output_limits; self.integral = 0; self.last_error = 0
    def initialize(self, initial_output):
        if self.Ki > 0: self.integral = initial_output / self.Ki
        else: self.integral = 0.0
        self.last_error = 0
    def update(self, pv, dt):
        error = self.setpoint - pv; self.integral += error * dt
        if self.Ki > 0: self.integral = np.clip(self.integral, self.min_output/self.Ki, self.max_output/self.Ki)
        derivative = (error - self.last_error) / dt if dt > 0 else 0.0
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.last_error = error
        return np.clip(output, self.min_output, self.max_output)

def calculate_biuret_kinetic_evap_generic(initial_biuret_total_pct, feed_conc_pct, product_temp_c, product_conc_pct, residence_time_s, A_BIURET, Ea_BIURET, R_GAS):
    if product_conc_pct < 1.0 or feed_conc_pct < 1.0 or residence_time_s <= 0: return initial_biuret_total_pct
    biuret_to_solids_ratio = initial_biuret_total_pct / feed_conc_pct if feed_conc_pct > 0 else 0
    concentrated_initial_biuret_pct = biuret_to_solids_ratio * product_conc_pct
    urea_in_product_pct = product_conc_pct - concentrated_initial_biuret_pct
    if urea_in_product_pct <= 0: return concentrated_initial_biuret_pct
    urea_mass_frac_solution = urea_in_product_pct / 100.0
    rho_prod_kg_m3 = rho_from_conc_evap(product_conc_pct)
    urea_conc_mol_m3 = (rho_prod_kg_m3 * urea_mass_frac_solution) / (M_urea_EVAP / 1000.0)
    T_kelvin = product_temp_c + 273.15
    k = A_BIURET * np.exp(-Ea_BIURET / (R_GAS * T_kelvin))
    rate_urea_consumption = k * (urea_conc_mol_m3 ** 2)
    rate_biuret_formation = 0.5 * rate_urea_consumption
    increase_biuret_mol_m3 = rate_biuret_formation * residence_time_s
    increase_biuret_kg_m3 = increase_biuret_mol_m3 * (M_BIURET_EVAP / 1000.0)
    newly_generated_biuret_pct = (increase_biuret_kg_m3 / rho_prod_kg_m3) * 100.0
    final_biuret_pct = concentrated_initial_biuret_pct + newly_generated_biuret_pct
    return max(concentrated_initial_biuret_pct, final_biuret_pct)


            
# =============================================================================
# ===== STREAMLIT UI AND MAIN APP STATE =======================================
# =============================================================================
if "time_sec" not in st.session_state:
    st.session_state.time_sec = 0
    
    # --- (CODE 1) State Initialization ---
    st.session_state.tank_volume = 80.0 * (15.6 / 100.0) 
    st.session_state.tank_conc = 77.2
    st.session_state.tank_temp = 99.0
    st.session_state.tank_biuret_conc = 0.460
    
    st.session_state.pump_running = True 
    if "fv_auto_sp_flow" not in st.session_state: st.session_state.fv_auto_sp_flow = 88.0 
    st.session_state.fv_manual_sp = 80.0
    if "fv_324401_sp" not in st.session_state: st.session_state.fv_324401_sp = 80.0 
    if "fv_324401_actual" not in st.session_state: st.session_state.fv_324401_actual = 80.0
    
    st.session_state.stream10_conc_last = 77.2
    st.session_state.stream10_biuret_last = 0.460
    st.session_state.stream10_temp_last = 99.0
    
    st.session_state.pre_evap_state = { 
        "Tprod_c": 100.07, "Psh_barg": 0.73, "Cprod_pct": 77.03, "Bprod_pct": 0.427,
        "pct_valve": 50.35, "Pvac_barg": -0.51, "shell_steam_mass_kg": 0.20
    }
    st.session_state.last_evap_rate_pre_evap = 0 
    st.session_state.last_mode_pre_evap = "Cascade" 
    st.session_state.active_alarms = []
    st.session_state.pid_pressure_pre_evap = PID_pre_evap(Kp=2.5, Ki=0.1, Kd=0.0, setpoint=0.73) 
    st.session_state.pid_temp_pre_evap = PID_pre_evap(Kp=4.0, Ki=0.2, Kd=0.0, setpoint=100.07) 
    initial_valve_pos = st.session_state.pre_evap_state['pct_valve']
    st.session_state.pid_pressure_pre_evap.initialize(initial_valve_pos)
    st.session_state.pid_temp_pre_evap.initialize(initial_valve_pos)

    st.session_state.actual_plant_load = 100.0
    st.session_state.plant_load_noise = 0.0
    st.session_state.noise_timer_sec = 0.0
    
    # --- (CODE 2) State Initialization (Evap 1) ---
    st.session_state.evap1_running = True
    st.session_state.evap1_history = []
    
    st.session_state.evap1_state = { 
        "Tprod_c": 129.86, "Psh_barg": 2.55, "Psh_display_barg": 2.55,
        "Cprod_pct": 95.67, "Bprod_pct": 0.7489, "pct_valve": 58.0,
        "Pvac_barg": -0.66, "shell_steam_mass_kg": 8.0
    }
    st.session_state.evap1_last_evap_rate = 0.0
    st.session_state.evap1_last_mode = "Cascade" 
    st.session_state.pid_pressure_evap1 = PID_evap_generic(Kp=0.3, Ki=0.004, Kd=0.0, setpoint=2.55)
    st.session_state.pid_temp_evap1 = PID_evap_generic(Kp=1.5, Ki=0.0046875, Kd=0.0, setpoint=129.86)
    st.session_state.pid_temp_evap1.initialize(58.0)

    # --- (CODE 3) State Initialization (Evap 2) ---
    st.session_state.evap2_running = True
    st.session_state.evap2_history = []
    
    st.session_state.evap2_state = { 
        "Tprod_c": 140.34, "Psh_barg": 4.48, "Psh_display_barg": 4.48,
        "Cprod_pct": 99.05, "Bprod_pct": 0.8784, "pct_valve": 39.7,
        "Pvac_barg": -0.90, "shell_steam_mass_kg": 8.0
    }
    st.session_state.evap2_last_evap_rate = 0.0
    st.session_state.evap2_last_mode = "Cascade" 

# --- (CODE 4) State Initialization (Melt Tank System) ---
    initial_conc_s15 = 99.05 
    initial_biuret_s15 = 0.88
    
    st.session_state.melt_system = {
        "level_pct": 50.0, "temp_c": 138.0, 
        "conc_pct": initial_conc_s15, "biuret_pct": initial_biuret_s15,
        "pump_running": True, "pump_trip_active": False, "pump_trip_timer": 0.0,
        "valve_A_pos": 50.0, "valve_B_pos": 0.0, "valve_condensate_pos": 0.0,
        "drain_open": False, "suction_open": True, "discharge_open": True,
        "condensate_open_cmd": False, "drain_open_cmd": False,
        "suction_open_cmd": True, "discharge_open_cmd": True,
        "stream19_flow_m3ph": 40.0,
        "header_pressure_filtered": 3.8, 
        "A_frozen_pos": 0.0, "high_pressure_override_active": False
    }

    # *** NEW: RECYCLE STREAM STATE INITIALIZATION ***
    if "stream20_props" not in st.session_state:
        st.session_state.stream20_props = {
            'mass_kgph': 0.0, 
            'temp_c': 138.0, 
            'conc_pct': initial_conc_s15, 
            'biuret_pct': initial_biuret_s15
        }

    st.session_state.lic_324501_mode = "Cascade"
    st.session_state.pid_level_melt = PID_evap_generic(Kp=0.85, Ki=0.00425, Kd=0.0, setpoint=50.0, output_limits=(0, 200)) 
    st.session_state.pid_level_melt.initialize(50.0)
    
    st.session_state.pid_pressure_evap2 = PID_evap_generic(Kp=2.5, Ki=0.1, Kd=0.0, setpoint=4.48)
    st.session_state.pid_temp_evap2 = PID_evap_generic(Kp=4.0, Ki=0.2, Kd=0.0, setpoint=140.34)
    st.session_state.pid_temp_evap2.initialize(39.7)

    # --- Pipe Lag State ---
    initial_rho_s10 = rho_from_conc_pre_evap(77.2 + 0.460) 
    initial_mass_s10 = 80.0 * initial_rho_s10
    st.session_state.stream11_props_lagged = {
        'mass_flow_kgph': initial_mass_s10, 'temp_c': 99.0, 'conc_pct': 77.2,
        'biuret_pct': 0.460, 'rho_kg_m3': initial_rho_s10, 'vol_flow_m3ph': 80.0
    }
    
    # --- (CODE 1) UI Display Definitions ---
    st.session_state.all_display_items = {
        "General": {
            "Plant Load": {'visible': True, 'top': 28.7, 'left': 14.6, 'width': 7.0, 'height': 5.0, 'color': '#0000FF', 'size': 1.0, 'format': '{:.1f}%'},
        },
        "Pre-evaporator": {
            "Product Temperature": {'visible': True, 'top': 38.3, 'left': 51.0, 'width': 8.0, 'height': 5.0, 'color': '#0000FF', 'size': 1.0, 'format': '{:.2f} C'},
            "Shell Pressure": {'visible': True, 'top': 30.9, 'left': 29.0, 'width': 8.0, 'height': 5.0, 'color': '#0000FF', 'size': 1.0, 'format': '{:.2f} barg'},
            "Vacuum": {'visible': True, 'top': 25.0, 'left': 50.8, 'width': 8.0, 'height': 5.0, 'color': '#0000FF', 'size': 1.0, 'format': '{:.2f} barg'},
            "Product Concentration": {'visible': True, 'top': 30.5, 'left': 64.0, 'width': 8.0, 'height': 5.0, 'color': '#BC00BC', 'size': 1.0, 'format': '{:.2f}%'},
        },
        "Tank": {
            "Level (%)": {'visible': True, 'top': 69.2, 'left': 53.4, 'width': 7.0, 'height': 5.0, 'color': '#0000FF', 'size': 0.9, 'format': '{:.1f}%'},
            "Temperature": {'visible': True, 'top': 81.5, 'left': 53.5, 'width': 7.0, 'height': 5.0, 'color': '#BC00BC', 'size': 0.9, 'format': '{:.1f} C'},
            "Urea Concentration": {'visible': True, 'top': 73.7, 'left': 53.7, 'width': 7.0, 'height': 5.0, 'color': '#BC00BC', 'size': 0.9, 'format': '{:.2f}%'},
            "Biuret Concentration": {'visible': True, 'top': 77.5, 'left': 53.7, 'width': 7.0, 'height': 5.0, 'color': '#BC00BC', 'size': 0.9, 'format': '{:.3f}%'},
            "Level Bar": {'visible': True, 'top': 71.0, 'left': 61.0, 'width': 2.1, 'height': 13.0, 'color': '#05F325'},
        },
        "Valves & Pump": {
            "Pump Status": {'visible': True, 'top': 89.0, 'left': 70.5, 'width': 10.0, 'height': 5.0, 'color': '#0000FF', 'size': 1.0, 'format': '{}'},
            "Suction Valve Status": {'visible': True, 'top': 89.0, 'left': 63.0, 'width': 10.0, 'height': 5.0, 'color': '#0000FF', 'size': 1.0, 'format': '{}'},
            "Discharge Valve Status": {'visible': True, 'top': 89.0, 'left': 76.0, 'width': 10.0, 'height': 5.0, 'color': '#0000FF', 'size': 1.0, 'format': '{}'},
            "Tank Outlet Valve Opening": {'visible': True, 'top': 87.0, 'left': 82.5, 'width': 8.0, 'height': 5.0, 'color': '#0000FF', 'size': 1.0, 'format': '{:.1f}%'},
            "Vacuum Valve Opening": {'visible': True, 'top': 31.0, 'left': 79.0, 'width': 7.0, 'height': 5.0, 'color': '#0000FF', 'size': 1.0, 'format': '{:.1f}%'},
            "Steam Valve Opening": {'visible': True, 'top': 45.5, 'left': 29.5, 'width': 8.0, 'height': 5.0, 'color': '#0000FF', 'size': 1.0, 'format': '{:.1f}%'},
            "Steam Valve Mode": {'visible': True, 'top': 40.0, 'left': 27.5, 'width': 6.0, 'height': 5.0, 'color': '#0000FF', 'size': 1.0, 'format': '{}'},
            "Tank Outlet Valve Mode": {'visible': True, 'top': 81.0, 'left': 80.0, 'width': 8.0, 'height': 5.0, 'color': '#0000FF', 'size': 1.0, 'format': '{}'},
        },
        "Stream 1 (Plant Feed)": {
            "Mass Flow (kg/h)": {'visible': True, 'top': 33.0, 'left': 14.0, 'width': 12.0, 'height': 4.0, 'color': '#0000FF', 'size': 1.0, 'format': '{:.0f} kg/h'},
            "Volume Flow (m³/h)": {'visible': True, 'top': 36.8, 'left': 13.5, 'width': 12.0, 'height': 4.0, 'color': '#0000FF', 'size': 1.0, 'format': '{:.1f} m³/h'},
            "Urea Conc": {'visible': True, 'top': 41.0, 'left': 12.8, 'width': 12.0, 'height': 4.0, 'color': '#0000FF', 'size': 1.0, 'format': '{:.2f} %'},
            "Temperature": {'visible': True, 'top': 45.0, 'left': 12.8, 'width': 12.0, 'height': 4.0, 'color': '#0000FF', 'size': 1.0, 'format': '{:.1f} C'},
            "Biuret Conc": {'visible': True, 'top': 48.8, 'left': 12.8, 'width': 12.0, 'height': 4.0, 'color': '#0000FF', 'size': 1.0, 'format': '{:.3f} %'},
        },
        "Stream 2 (Dissolving)": {
            "Volume Flow (m³/h)": {'visible': True, 'top': 74.4, 'left': 11.2, 'width': 12.0, 'height': 4.0, 'color': '#0000FF', 'size': 1.0, 'format': '{:.1f} m³/h'},
            "Urea Conc": {'visible': True, 'top': 78.5, 'left': 10.5, 'width': 12.0, 'height': 4.0, 'color': '#0000FF', 'size': 1.0, 'format': '{:.2f} %'},
            "Temperature": {'visible': True, 'top': 82.5, 'left': 10.7, 'width': 12.0, 'height': 4.0, 'color': '#0000FF', 'size': 1.0, 'format': '{:.1f} C'},
            "Biuret Conc": {'visible': True, 'top': 87.0, 'left': 10.8, 'width': 12.0, 'height': 4.0, 'color': '#0000FF', 'size': 1.0, 'format': '{:.3f} %'},
        },
        "Stream 3 (Evap Feed)": {
            "Volume Flow (m³/h)": {'visible': True, 'top': 74.5, 'left': 34.5, 'width': 12.0, 'height': 4.0, 'color': '#BC00BC', 'size': 1.0, 'format': '{:.1f} m³/h'},
            "Temperature": {'visible': True, 'top': 82.8, 'left': 33.7, 'width': 12.0, 'height': 4.0, 'color': '#BC00BC', 'size': 1.0, 'format': '{:.1f} C'},
            "Urea Conc": {'visible': True, 'top': 78.6, 'left': 33.7, 'width': 12.0, 'height': 4.0, 'color': '#BC00BC', 'size': 1.0, 'format': '{:.2f} %'},
        },
        "Stream 10 (Pump Out)": {
            "Volume Flow (m³/h)": {'visible': True, 'top': 59.2, 'left': 87.2, 'width': 12.0, 'height': 4.0, 'color': '#0000FF', 'size': 1.0, 'format': '{:.1f} m³/h'},
            "Urea Conc": {'visible': True, 'top': 63.0, 'left': 87.0, 'width': 12.0, 'height': 4.0, 'color': '#BC00BC', 'size': 1.0, 'format': '{:.2f} %'},
        },
    }
    
    # *** Updated Coordinates with Level Bar for Melt Tank ***
    st.session_state.evap_view_display_items = {
        "Tank (From Code 1)": {
            "Level": {'visible': True, 'top': 76.0, 'left': 0.0, 'width': 15.0, 'height': 5.0, 'color': '#0000FF', 'size': 1.0, 'format': '{:.1f}%'},
            "Pump Status": {'visible': True, 'top': 88.3, 'left': 5.6, 'width': 15.0, 'height': 5.0, 'color': '#0000FF', 'size': 1.0, 'format': '{}'},
            "FV-324401 Mode": {'visible': True, 'top': 71.6, 'left': 12.6, 'width': 15.0, 'height': 5.0, 'color': '#0000FF', 'size': 1.0, 'format': '{}'},
            "FV-324401 Opening": {'visible': True, 'top': 76.0, 'left': 7.5, 'width': 15.0, 'height': 5.0, 'color': '#0000FF', 'size': 1.0, 'format': '{:.1f}%'},
            "FV-324401 SP": {'visible': True, 'top': 67.0, 'left': 17.0, 'width': 15.0, 'height': 5.0, 'color': '#0000FF', 'size': 1.0, 'format': '{:.1f} m³/h'},
        },
        "1st Evaporator KPIs": {
            "Product Temp (S12)": {'visible': True, 'top': 38.8, 'left': 22.2, 'width': 15.0, 'height': 5.0, 'color': '#0000FF', 'size': 1.0, 'format': '{:.2f} C'},
            "Product Conc (S12)": {'visible': True, 'top': 31.0, 'left': 33.0, 'width': 15.0, 'height': 5.0, 'color': '#0000FF', 'size': 1.0, 'format': '{:.2f} %'},
            "Product Biuret (S12)": {'visible': True, 'top': 41.5, 'left': 33.3, 'width': 15.0, 'height': 5.0, 'color': '#BC00BC', 'size': 1.0, 'format': '{:.4f} %'},
            "Product Mass (S12)": {'visible': False, 'top': 25.0, 'left': 30.0, 'width': 15.0, 'height': 5.0, 'color': '#0000FF', 'size': 1.0, 'format': '{:.0f} kg/h'},
            "Vapor Flow (S13)": {'visible': False, 'top': 30.0, 'left': 30.0, 'width': 15.0, 'height': 5.0, 'color': '#BC00BC', 'size': 1.0, 'format': '{:.0f} kg/h'},
            "Steam Consumption": {'visible': True, 'top': 55.2, 'left': 6.5, 'width': 15.0, 'height': 5.0, 'color': '#BC00BC', 'size': 1.0, 'format': '{:.0f} kg/h'},
            "Heat Duty (Q)": {'visible': True, 'top': 61.0, 'left': 6.5, 'width': 15.0, 'height': 5.0, 'color': '#BC00BC', 'size': 1.0, 'format': '{:.1f} kW'},
        },
        "1st Evaporator Valves": {
            "Steam Valve Mode": {'visible': True, 'top': 40.0, 'left': 4.3, 'width': 15.0, 'height': 5.0, 'color': '#0000FF', 'size': 1.0, 'format': '{}'},
            "Steam Valve Opening": {'visible': True, 'top': 46.0, 'left': 2.0, 'width': 15.0, 'height': 5.0, 'color': '#0000FF', 'size': 1.0, 'format': '{:.1f} %'},
            "Steam ISO Valve": {'visible': False, 'top': 20.0, 'left': 50.0, 'width': 15.0, 'height': 5.0, 'color': '#0000FF', 'size': 1.0, 'format': '{:.0f} %'},
            "Vacuum Valve Opening": {'visible': True, 'top': 25.5, 'left': 35.0, 'width': 15.0, 'height': 5.0, 'color': '#0000FF', 'size': 1.0, 'format': '{:.1f} %'},
            "Vacuum ISO Valve": {'visible': False, 'top': 30.0, 'left': 50.0, 'width': 15.0, 'height': 5.0, 'color': '#0000FF', 'size': 1.0, 'format': '{:.0f} %'},
            "Shell Vent": {'visible': False, 'top': 35.0, 'left': 50.0, 'width': 15.0, 'height': 5.0, 'color': '#0000FF', 'size': 1.0, 'format': '{}'},
            "Shell Pressure": {'visible': True, 'top': 31.5, 'left': 1.0, 'width': 15.0, 'height': 5.0, 'color': '#0000FF', 'size': 1.0, 'format': '{:.2f} barg'},
            "Flash Vacuum": {'visible': True, 'top': 26.1, 'left': 22.5, 'width': 15.0, 'height': 5.0, 'color': '#0000FF', 'size': 1.0, 'format': '{:.2f} barg'},
        },
        "2nd Evaporator KPIs": {
            "Product Temp (S15)": {'visible': True, 'top': 38.0, 'left': 66.5, 'width': 15.0, 'height': 5.0, 'color': '#0000FF', 'size': 1.0, 'format': '{:.2f} C'},
            "Product Conc (S15)": {'visible': True, 'top': 32.3, 'left': 78.0, 'width': 15.0, 'height': 5.0, 'color': '#0000FF', 'size': 1.0, 'format': '{:.2f} %'},
            "Product Biuret (S15)": {'visible': True, 'top': 39.2, 'left': 78.5, 'width': 15.0, 'height': 5.0, 'color': '#BC00BC', 'size': 1.0, 'format': '{:.4f} %'},
            "Product Mass (S15)": {'visible': False, 'top': 60.0, 'left': 85.0, 'width': 15.0, 'height': 5.0, 'color': '#FF5733', 'size': 1.0, 'format': '{:.0f} kg/h'},
            "Vapor Flow (S16)": {'visible': False, 'top': 65.0, 'left': 85.0, 'width': 15.0, 'height': 5.0, 'color': '#FF5733', 'size': 1.0, 'format': '{:.0f} kg/h'},
            "Steam Consumption": {'visible': True, 'top': 52.5, 'left': 52.0, 'width': 15.0, 'height': 5.0, 'color': '#BC00BC', 'size': 1.0, 'format': '{:.0f} kg/h'},
             "Heat Duty (Q)": {'visible': True, 'top': 58.2, 'left': 52.0, 'width': 15.0, 'height': 5.0, 'color': '#BC00BC', 'size': 1.0, 'format': '{:.1f} kW'},
        },
        "2nd Evaporator Valves": {
            "Steam Valve Mode": {'visible': True, 'top': 38.8, 'left': 52.0, 'width': 15.0, 'height': 5.0, 'color': '#0000FF', 'size': 1.0, 'format': '{}'},
            "Steam Valve Opening": {'visible': True, 'top': 43.0, 'left': 50.0, 'width': 15.0, 'height': 5.0, 'color': '#0000FF', 'size': 1.0, 'format': '{:.1f} %'},
            "Steam ISO Valve": {'visible': False, 'top': 38.0, 'left': 75.0, 'width': 15.0, 'height': 5.0, 'color': '#FF5733', 'size': 1.0, 'format': '{:.0f} %'},
            "Vacuum Valve Opening": {'visible': True, 'top': 25.3, 'left': 75.5, 'width': 15.0, 'height': 5.0, 'color': '#0000FF', 'size': 1.0, 'format': '{:.1f} %'},
            "Vacuum ISO Valve": {'visible': False, 'top': 28.0, 'left': 90.0, 'width': 15.0, 'height': 5.0, 'color': '#FF5733', 'size': 1.0, 'format': '{:.0f} %'},
            "Shell Vent": {'visible': False, 'top': 42.0, 'left': 75.0, 'width': 15.0, 'height': 5.0, 'color': '#FF5733', 'size': 1.0, 'format': '{}'},
            "Shell Pressure": {'visible': True, 'top': 32.0, 'left': 48.0, 'width': 15.0, 'height': 5.0, 'color': '#0000FF', 'size': 1.0, 'format': '{:.2f} barg'},
            "Flash Vacuum": {'visible': True, 'top': 28.8, 'left': 67.0, 'width': 15.0, 'height': 5.0, 'color': '#0000FF', 'size': 1.0, 'format': '{:.2f} barg'},
        },
         "Melt Tank System": {
            "Level": {'visible': True, 'top': 55.0, 'left': 70.0, 'width': 7.0, 'height': 5.0, 'color': '#0000FF', 'size': 1.0, 'format': '{:.1f}%'},
            "Level Bar": {'visible': True, 'top': 50, 'left': 68.7, 'width': 2.1, 'height': 13.0, 'color': '#05F325'}, # Added Bar
            "Temperature": {'visible': True, 'top': 87.0, 'left': 90.0, 'width': 7.0, 'height': 5.0, 'color': '#0000FF', 'size': 1.0, 'format': '{:.1f} C'},
            "Concentration": {'visible': True, 'top': 91.0, 'left': 90.0, 'width': 7.0, 'height': 5.0, 'color': '#BC00BC', 'size': 1.0, 'format': '{:.2f}%'},
            "Pump Status": {'visible': True, 'top': 71.0, 'left': 68.0, 'width': 10.0, 'height': 5.0, 'color': '#0000FF', 'size': 1.0, 'format': '{}'},
            "Prod. Flow (S19)": {'visible': True, 'top': 51.5, 'left': 89.0, 'width': 10.0, 'height': 5.0, 'color': '#0000FF', 'size': 1.0, 'format': '{:.1f} m³/h'},
            "Recycle Flow (S20)": {'visible': False, 'top': 70.0, 'left': 38.0, 'width': 10.0, 'height': 5.0, 'color': '#0000FF', 'size': 1.0, 'format': '{:.1f} m³/h'},
            "Header Pressure": {'visible': True, 'top': 56.4, 'left': 90.0, 'width': 10.0, 'height': 5.0, 'color': '#BC00BC', 'size': 1.0, 'format': '{:.2f} barg'},
            "Valve A (Prod)": {'visible': True, 'top': 69.0, 'left': 75.5, 'width': 8.0, 'height': 5.0, 'color': '#0000FF', 'size': 1.0, 'format': '{:.1f}%'},
            "Valve B (Recyc)": {'visible': True, 'top': 82.0, 'left': 74.0, 'width': 8.0, 'height': 5.0, 'color': '#0000FF', 'size': 1.0, 'format': '{:.1f}%'},
            "Stream 23 (Mixer Out/S5)": {'visible': False, 'top': 80.0, 'left': 60.0, 'width': 10.0, 'height': 5.0, 'color': '#BC00BC', 'size': 1.0, 'format': '{:.2f}%'},
        },
    }
    
# =============================================================================
# ===== (CODE 1) SIDEBAR UI ===================================================
# =============================================================================
st.sidebar.title("Simulation Controls")
selected_view = st.sidebar.radio("Select View", ["Tank System", "Alarms", "Performance KPIs", "Streams Data", "Future View 2"], horizontal=True)

if selected_view in ["Tank System", "Alarms", "Performance KPIs", "Streams Data", "Future View 2"]:
    st.sidebar.header("Upstream Plant Controls")
    UIC1 = st.sidebar.number_input("UIC1: Urea Plant Load Setpoint (%)", 0.0, 115.0, 100.0)
    stream2_vol_flow = st.sidebar.number_input("Dissolving Tank Flow (FIC-335407) [m³/h]", 0.0, 20.0, 6.0)

    st.sidebar.markdown("---")
    st.sidebar.header("Pre-evaporator Controls") 
    mode = st.sidebar.radio("Mode (PV-329208)", ["Manual", "Auto", "Cascade"], horizontal=True, key="evap_mode", index=2)
    if mode != st.session_state.last_mode_pre_evap:
        current_pv_temp = st.session_state.pre_evap_state['Tprod_c']
        current_pv_press = st.session_state.pre_evap_state['Psh_barg']
        last_valve_pos = st.session_state.pre_evap_state['pct_valve']
        if mode == "Auto": st.session_state.pid_pressure_pre_evap.setpoint = current_pv_press; st.session_state.pid_pressure_pre_evap.initialize(last_valve_pos)
        elif mode == "Cascade": st.session_state.pid_temp_pre_evap.setpoint = current_pv_temp; st.session_state.pid_temp_pre_evap.initialize(last_valve_pos)
        st.session_state.last_mode_pre_evap = mode

    if mode == "Manual": st.session_state.pre_evap_state['pct_valve'] = st.sidebar.slider(f"Steam Valve (PV-329208) [%]", 0.0, 100.0, st.session_state.pre_evap_state['pct_valve'])
    elif mode == "Auto": st.session_state.pid_pressure_pre_evap.setpoint = st.sidebar.number_input("Press SP [barg]", 0.0, 9.0, st.session_state.pid_pressure_pre_evap.setpoint)
    elif mode == "Cascade": st.session_state.pid_temp_pre_evap.setpoint = st.sidebar.number_input("Temp SP [°C]", 90.0, 150.0, st.session_state.pid_temp_pre_evap.setpoint)
    pct_vac_valve = st.sidebar.slider("Vacuum Valve (HV-323605) [%]", 0.0, 100.0, 60.80, 0.1)

    # *** 1st Evaporator CONTROLS ***
    st.sidebar.markdown("---")
    st.sidebar.header("1st Evaporator Controls")
    
    master_p_evap1 = st.sidebar.slider("1st Evap Steam Pressure [barg]", 0.0, 9.0, 3.8, 0.1, key="evap1_steam_p")
    
    with st.sidebar.expander("1st Evap Advanced Parameters"):
        A_HEAT_TRANSFER_AREA_EVAP1 = st.number_input("Heat Transfer Area [m²]", value=A_HEAT_TRANSFER_AREA_EVAP_DEFAULT, step=10.0, key="evap1_area")
        U_HEAT_TRANSFER_COEFF_EVAP1 = st.slider("Global Heat Transfer Coeff (U) [W/m²K]", 500.0, 5000.0, U_HEAT_TRANSFER_COEFF_EVAP_DEFAULT, step=50.0, key="evap1_u_val")
        shell_vol_m3_EVAP1 = st.number_input("Shell Volume [m³]", value=shell_vol_m3_EVAP_DEFAULT, step=0.1, key="evap1_shell_vol")
        tube_vol_m3_EVAP1 = st.number_input("Tubes Volume [m³]", value=tube_vol_m3_EVAP_DEFAULT, step=0.1, key="evap1_tube_vol")
        FLASH_VESSEL_VOL_M3_EVAP1 = st.number_input("Flash Vessel Volume [m³]", value=FLASH_VESSEL_VOL_M3_EVAP_DEFAULT, step=1.0, key="evap1_flash_vol")
        max_steam_flow_kgph_evap1 = st.number_input("Max Steam Flow [kg/h]", value=25000.0, step=100.0, key="evap1_steam_cap")
        A_BIURET_EVAP1 = st.number_input("Biuret Arrhenius Factor (A)", value=A_BIURET_EVAP_DEFAULT, format="%e", key="evap1_A_biuret")
        Ea_BIURET_EVAP1 = st.number_input("Biuret Activation Energy (Ea) [J/mol]", value=Ea_BIURET_EVAP_DEFAULT, step=1000.0, key="evap1_Ea_biuret")
    
    modes_evap1 = ["Manual", "Auto", "Cascade"]
    mode_evap1 = st.sidebar.radio("Mode (PV-329203)", modes_evap1, index=2, key="evap1_mode_radio", horizontal=True)
    
    if mode_evap1 != st.session_state.evap1_last_mode:
        current_pv_temp_evap1 = st.session_state.evap1_state['Tprod_c']
        current_pv_press_evap1 = st.session_state.evap1_state['Psh_display_barg']
        last_valve_pos_evap1 = st.session_state.evap1_state['pct_valve']
        if mode_evap1 == "Auto":
            st.session_state.pid_pressure_evap1.setpoint = current_pv_press_evap1; st.session_state.pid_pressure_evap1.initialize(last_valve_pos_evap1)
        elif mode_evap1 == "Cascade":
            st.session_state.pid_temp_evap1.setpoint = max(90.0, current_pv_temp_evap1); st.session_state.pid_temp_evap1.initialize(last_valve_pos_evap1)
        st.session_state.evap1_last_mode = mode_evap1
        
    if mode_evap1 == "Auto": st.session_state.pid_pressure_evap1.setpoint = st.sidebar.number_input("1st Evap Press SP [barg]", 0.0, 9.0, st.session_state.pid_pressure_evap1.setpoint, key="evap1_sp_p")
    elif mode_evap1 == "Cascade": st.session_state.pid_temp_evap1.setpoint = st.sidebar.number_input("1st Evap Temp SP [°C]", 90.0, 150.0, st.session_state.pid_temp_evap1.setpoint, key="evap1_sp_t")

    if mode_evap1 == "Manual": st.session_state.evap1_state['pct_valve'] = st.sidebar.slider("1st Evap Valve (PV-329203) [%]", 0.0, 100.0, st.session_state.evap1_state['pct_valve'], key="evap1_valve_man")
    pct_vac_valve_evap1 = st.sidebar.slider("1st Evap Vacuum Valve (HV-329605) [%]", 0.0, 100.0, 60.0, 1.0, key="evap1_vac_valve")
    
    # *** 2nd Evaporator CONTROLS ***
    st.sidebar.markdown("---")
    st.sidebar.header("2nd Evaporator Controls")
    
    master_p_evap2 = st.sidebar.slider("2nd Evap Steam Pressure [barg]", 0.0, 9.0, 9.0, 0.1, key="evap2_steam_p")
    
    with st.sidebar.expander("2nd Evap Advanced Parameters"):
        A_HEAT_TRANSFER_AREA_EVAP2 = st.number_input("Heat Transfer Area [m²]", value=55.0, step=10.0, key="evap2_area")
        U_HEAT_TRANSFER_COEFF_EVAP2 = st.slider("Global Heat Transfer Coeff (U) [W/m²K]", 500.0, 5000.0, 2143.0, step=50.0, key="evap2_u_val")
        shell_vol_m3_EVAP2 = st.number_input("Shell Volume [m³]", value=1.768, step=0.1, key="evap2_shell_vol")
        tube_vol_m3_EVAP2 = st.number_input("Tubes Volume [m³]", value=0.441, step=0.1, key="evap2_tube_vol")
        FLASH_VESSEL_VOL_M3_EVAP2 = st.number_input("Flash Vessel Volume [m³]", value=12.00, step=1.0, key="evap2_flash_vol")
        max_steam_flow_kgph_evap2 = st.number_input("Max Steam Flow [kg/h]", value=3800.0, step=100.0, key="evap2_steam_cap")
        A_BIURET_EVAP2 = st.number_input("Biuret Arrhenius Factor (A)", value=3.9e+7, format="%e", key="evap2_A_biuret")
        Ea_BIURET_EVAP2 = st.number_input("Biuret Activation Energy (Ea) [J/mol]", value=128000.0, step=1000.0, key="evap2_Ea_biuret")
    
    mode_evap2 = st.sidebar.radio("Mode", ["Manual", "Auto", "Cascade"], index=2, key="evap2_mode_radio", horizontal=True)
    
    if mode_evap2 != st.session_state.evap2_last_mode:
        current_pv_temp_evap2 = st.session_state.evap2_state['Tprod_c']
        current_pv_press_evap2 = st.session_state.evap2_state['Psh_display_barg']
        last_valve_pos_evap2 = st.session_state.evap2_state['pct_valve']
        if mode_evap2 == "Auto":
            st.session_state.pid_pressure_evap2.setpoint = current_pv_press_evap2; st.session_state.pid_pressure_evap2.initialize(last_valve_pos_evap2)
        elif mode_evap2 == "Cascade":
            st.session_state.pid_temp_evap2.setpoint = max(90.0, current_pv_temp_evap2); st.session_state.pid_temp_evap2.initialize(last_valve_pos_evap2)
        st.session_state.evap2_last_mode = mode_evap2
        
    if mode_evap2 == "Auto": st.session_state.pid_pressure_evap2.setpoint = st.sidebar.number_input("2nd Evap Press SP [barg]", 0.0, 9.0, st.session_state.pid_pressure_evap2.setpoint, key="evap2_sp_p")
    elif mode_evap2 == "Cascade": st.session_state.pid_temp_evap2.setpoint = st.sidebar.number_input("2nd Evap Temp SP [°C]", 90.0, 160.0, st.session_state.pid_temp_evap2.setpoint, key="evap2_sp_t")

    if mode_evap2 == "Manual": st.session_state.evap2_state['pct_valve'] = st.sidebar.slider("2nd Evap Valve [%]", 0.0, 100.0, st.session_state.evap2_state['pct_valve'], key="evap2_valve_man")
    pct_vac_valve_evap2 = st.sidebar.slider("2nd Evap Vacuum Valve [%]", 0.0, 100.0, 32.0, 1.0, key="evap2_vac_valve")

    # Vacuum Params (Shared Structure, Configurable)
    with st.sidebar.expander("Configure Vacuum Equations"):
        st.write("Using default params for now (tunable in code)")
        vac_params = {'vp1_pct': 60, 'vp1_vac': -0.79, 'vp2_pct': 80, 'vp2_vac': -0.81, 'vp3_vac': -0.83, 'master_p_norm': 3.8, 'evap_norm': 16770, 'penalty_factor': 0.11, 'min_vac': -0.83, 'max_vac': 2.0}

    st.sidebar.markdown("---")
    st.sidebar.header("Downstream Controls")
    valve_mode = st.sidebar.radio("FV-324401 Valve Control Mode", ["Auto", "Manual"], horizontal=True, key="tank_valve_mode", index=0) # Default Auto
    
    if valve_mode == "Manual":
        st.sidebar.slider("FV-324401 Setpoint (%)", 0.0, 100.0, st.session_state.fv_manual_sp, key="fv_manual_sp") 
        st.session_state.fv_324401_sp = st.session_state.fv_manual_sp
    else: 
        st.sidebar.number_input("FIC-324401 Flow Setpoint (m³/h)", 0.0, 110.0, key="fv_auto_sp_flow")
        st.session_state.fv_324401_sp = st.session_state.fv_auto_sp_flow / 1.1 if 1.1 > 0 else st.session_state.fv_auto_sp_flow
    fic_sp = st.session_state.fv_auto_sp_flow if valve_mode == "Auto" else 0.0

    st.sidebar.markdown("---")
    st.sidebar.header("Site Actions")
    open_cond_suction = st.sidebar.checkbox("Open Condensate on Suction of 323P003", value=False)
    if open_cond_suction: stream8_condensate = st.sidebar.slider("Condensate Flow (m³/h)", 0.0, 30.0, 0.0)
    else: stream8_condensate = 0.0
        
    pump_status_selection = st.sidebar.radio("Pump Status", ["On", "Off"], horizontal=True, index=0) 
    st.session_state.pump_running = (pump_status_selection == "On")
    pump_fault = (pump_status_selection == "Off")
    suction_valve_selection = st.sidebar.radio("Pump Suction Valve", ["Open", "Close"], horizontal=True, index=0)
    close_323P003_suction = (suction_valve_selection == "Close")
    discharge_valve_selection = st.sidebar.radio("Pump Discharge Valve", ["Open", "Close"], horizontal=True, index=0)
    close_323P003 = (discharge_valve_selection == "Close")
    
    # 1st Evap Site Actions
    iso_valve_steam_evap1 = st.sidebar.slider("1st Evap Steam ISO Valve [%]", 0.0, 100.0, 100.0)
    iso_valve_vac_evap1 = st.sidebar.slider("1st Evap Vac ISO Valve [%]", 0.0, 100.0, 100.0)
    shell_vent_evap1 = st.sidebar.checkbox("1st Evap Shell Vent Open", value=False)

    # 2nd Evap Site Actions
    iso_valve_steam_evap2 = st.sidebar.slider("2nd Evap Steam ISO Valve [%]", 0.0, 100.0, 100.0)
    iso_valve_vac_evap2 = st.sidebar.slider("2nd Evap Vac ISO Valve [%]", 0.0, 100.0, 100.0)
    shell_vent_evap2 = st.sidebar.checkbox("2nd Evap Shell Vent Open", value=False)
    
    st.sidebar.markdown("---")
    st.sidebar.header("Melt Tank System Controls")
    
    # 1. Mode Selection with Bumpless Transfer Logic
    mode_melt = st.sidebar.radio("LIC-324501 Mode", ["Manual", "Auto", "Cascade"], index=2, horizontal=True)
    
    if mode_melt != st.session_state.lic_324501_mode:
        prev_A = st.session_state.melt_system['valve_A_pos']
        prev_B = st.session_state.melt_system['valve_B_pos']
        
        if mode_melt == "Cascade":
            if prev_B > 0: init_pid = 50.0 + (prev_B / 2.0)
            else: init_pid = prev_A / 2.0
            st.session_state.pid_level_melt.initialize(init_pid)
        elif mode_melt == "Auto":
            st.session_state.pid_level_melt.initialize(prev_B)
            
        st.session_state.lic_324501_mode = mode_melt

    # 2. Setpoints and Valve Controls based on Mode
    if mode_melt == "Manual":
        st.session_state.melt_system['valve_A_pos'] = st.sidebar.slider("LV-324501A (Prod) [%]", 0.0, 100.0, st.session_state.melt_system['valve_A_pos'])
        st.session_state.melt_system['valve_B_pos'] = st.sidebar.slider("LV-324501B (Recycle) [%]", 0.0, 100.0, st.session_state.melt_system['valve_B_pos'])
        
    elif mode_melt == "Auto":
        st.write("🤖 **Auto Mode:** PID controls Valve B (Recycle)")
        st.session_state.pid_level_melt.setpoint = st.sidebar.number_input("Level SP [%]", 0.0, 100.0, st.session_state.pid_level_melt.setpoint)
        st.session_state.melt_system['valve_A_pos'] = st.sidebar.slider("🖐 LV-324501A (Manual Control) [%]", 0.0, 100.0, st.session_state.melt_system['valve_A_pos'])
        
    elif mode_melt == "Cascade":
        st.write("🔗 **Cascade Mode:** Split Range (A Primary -> B Secondary)")
        st.session_state.pid_level_melt.setpoint = st.sidebar.number_input("Level SP [%]", 0.0, 100.0, st.session_state.pid_level_melt.setpoint)

    # 3. Site Actions & Parameters
    with st.sidebar.expander("Melt System Site Actions"):
        col_site1, col_site2 = st.columns(2)
        melt_pump_cmd = col_site1.radio("Melt Pump Cmd", ["Start", "Stop"], index=0 if st.session_state.melt_system['pump_running'] else 1)
        st.session_state.melt_system['suction_open_cmd'] = col_site2.checkbox("Open Suction Valve", value=st.session_state.melt_system['suction_open_cmd'])
        st.session_state.melt_system['discharge_open_cmd'] = col_site1.checkbox("Open Discharge Valve", value=st.session_state.melt_system['discharge_open_cmd'])
        st.session_state.melt_system['drain_open_cmd'] = col_site2.checkbox("Open Drain Valve", value=st.session_state.melt_system['drain_open_cmd'])
        
        st.write("---")
        # HV-335602 Control
        st.session_state.melt_system['hv_335602_pos'] = st.slider("HV-335602 (Header Bypass) [%]", 0.0, 100.0, st.session_state.melt_system.get('hv_335602_pos', 0.0))
        st.session_state.melt_system['active_headers'] = st.number_input("Active Headers Count", 1, 20, st.session_state.melt_system.get('active_headers', 12))
        
        st.write("---")
        st.session_state.melt_system['condensate_open_cmd'] = st.checkbox("Open Condensate on Melt Pump", value=st.session_state.melt_system['condensate_open_cmd'])
        if st.session_state.melt_system['condensate_open_cmd']:
            st.session_state.melt_system['valve_condensate_pos'] = st.slider("Condensate Valve [%]", 0.0, 100.0, st.session_state.melt_system['valve_condensate_pos'])
        else:
             st.session_state.melt_system['valve_condensate_pos'] = 0.0
             
        # *** NEW: 501B MIXER CONTROL ***
        st.write("---")
        st.write("🔧 **501B Process Condensate Mixer**")
        open_proc_cond_501b = st.checkbox("Open Process Condensate on 501B", value=False)
        
        if open_proc_cond_501b:
            stream21_flow = st.slider("Stream 21 Flow (m³/h)", 0.0, 10.0, 0.0, step=0.1)
        else:
            stream21_flow = 0.0

        if melt_pump_cmd == "Stop": st.session_state.melt_system['pump_running'] = False
        elif melt_pump_cmd == "Start" and not st.session_state.melt_system['pump_trip_active']: 
            if st.session_state.melt_system['level_pct'] > 5.0: st.session_state.melt_system['pump_running'] = True
            
    # *** DISPLAY SETTINGS ***
    with st.sidebar.expander("إعدادات عرض المخرجات", expanded=True):
        if selected_view == "Tank System":
            items_dict = st.session_state.all_display_items
            st.write("Configuring: Tank System (P&ID 1)")
        elif selected_view == "Future View 2":
            items_dict = st.session_state.evap_view_display_items
            st.write("Configuring: 1st & 2nd Evaporator (P&ID 2)")
        else:
            items_dict = None
            st.write("Display settings only available for 'Tank System' or 'Future View 2'.")

        if items_dict:
            entity_to_configure = st.selectbox("اختر المكون أو التيار:", list(items_dict.keys()))
            if entity_to_configure:
                st.subheader(f"خصائص: {entity_to_configure}")
                properties = items_dict[entity_to_configure]
                for prop_name, settings in properties.items():
                    s_key = re.sub(r'[^A-Za-z0-9_]', '', entity_to_configure)
                    p_key = re.sub(r'[^A-Za-z0-9_]', '', prop_name)
                    key_prefix = "view1" if selected_view == "Tank System" else "view2"
                    settings['visible'] = st.checkbox(f"إظهار: {prop_name}", settings['visible'], key=f"{key_prefix}_{s_key}_{p_key}_vis")
                    if settings['visible']:
                        cols = st.columns(2)
                        settings['left'] = cols[0].number_input(f"Left (%)##{key_prefix}_{s_key}_{p_key}", 0.0, 100.0, float(settings['left']), 0.1, format="%.1f")
                        settings['top'] = cols[1].number_input(f"Top (%)##{key_prefix}_{s_key}_{p_key}", 0.0, 100.0, float(settings['top']), 0.1, format="%.1f")
                        if 'size' in settings:
                            cols2 = st.columns(2)
                            settings['size'] = cols2[0].number_input(f"Font Size (vw)##{key_prefix}_{s_key}_{p_key}", 0.1, 5.0, float(settings['size']), 0.1, format="%.1f")
                            settings['color'] = cols2[1].color_picker(f"Color##{key_prefix}_{s_key}_{p_key}", settings['color'])
                        else: settings['color'] = st.color_picker(f"Color##{key_prefix}_{s_key}_{p_key}", settings['color'])
                        st.markdown("---")  
# =============================================================================
# ===== (CODE 1) MAIN CALCULATION LOOP ========================================
# =============================================================================
st.session_state.time_sec += DT
st.session_state.noise_timer_sec += DT
if st.session_state.noise_timer_sec >= 8.0:
    st.session_state.plant_load_noise = random.uniform(-0.3, 0.3)
    st.session_state.noise_timer_sec = 0.0
target_load = UIC1; current_load = st.session_state.actual_plant_load
load_error = target_load - current_load
new_actual_load = current_load + (load_error / 20.0) * DT
if (load_error > 0 and new_actual_load > target_load) or (load_error < 0 and new_actual_load < target_load): new_actual_load = target_load
st.session_state.actual_plant_load = new_actual_load
effective_plant_load = np.clip(st.session_state.actual_plant_load + st.session_state.plant_load_noise, 0.0, 120.0)

plant_flow = (effective_plant_load * 17.5) / 24; urea_tph = plant_flow; urea_mass_flow_kgph = urea_tph * 1000
stream1_rho = rho_from_conc_pre_evap(STREAM1_CONC + STREAM1_BIURET)
total_mass_1 = urea_mass_flow_kgph / (STREAM1_CONC / 100) if STREAM1_CONC > 0 else 0
stream1_vol_flow = total_mass_1 / stream1_rho if stream1_rho > 0 else 0
stream2_rho=rho_from_conc_pre_evap(STREAM2_CONC+STREAM2_BIURET); total_mass_2=stream2_vol_flow*stream2_rho
total_urea_1=total_mass_1*STREAM1_CONC/100; total_urea_2=total_mass_2*STREAM2_CONC/100; total_biuret_1=total_mass_1*STREAM1_BIURET/100; total_biuret_2=total_mass_2*STREAM2_BIURET/100
stream3_mass=total_mass_1+total_mass_2; stream3_urea=total_urea_1+total_urea_2; stream3_biuret_total=total_biuret_1+total_biuret_2
stream3_conc=(stream3_urea/stream3_mass)*100 if stream3_mass>0 else 0; stream3_biuret_conc=(stream3_biuret_total/stream3_mass)*100 if stream3_mass>0 else 0
cp1=cp_from_conc_pre_evap(STREAM1_CONC+STREAM1_BIURET); cp2=cp_from_conc_pre_evap(STREAM2_CONC+STREAM2_BIURET)
Q1=total_mass_1*cp1*110.0; Q2=total_mass_2*cp2*40.0
stream3_temp=(Q1+Q2)/(total_mass_1*cp1+total_mass_2*cp2) if (total_mass_1*cp1+total_mass_2*cp2)>0 else 100.0

evap_feed_mass_kgph=stream3_mass; evap_feed_conc_pct=stream3_conc; evap_feed_temp_c=stream3_temp; evap_feed_biuret_pct=stream3_biuret_conc
m_gps=evap_feed_mass_kgph/3600; m_urea_gps=m_gps*(evap_feed_conc_pct/100); cp0=cp_from_conc_pre_evap(evap_feed_conc_pct)*1000
T_current = st.session_state.pre_evap_state['Tprod_c']; Psh_current = st.session_state.pre_evap_state['Psh_barg']
C_current = st.session_state.pre_evap_state['Cprod_pct']; B_current = st.session_state.pre_evap_state['Bprod_pct']
pct_current = st.session_state.pre_evap_state['pct_valve']; Pvac_current = st.session_state.pre_evap_state['Pvac_barg']
shell_steam_mass_current = st.session_state.pre_evap_state['shell_steam_mass_kg']

if mode == "Auto": pct_sp = st.session_state.pid_pressure_pre_evap.update(Psh_current, DT)
elif mode == "Cascade": pct_sp = st.session_state.pid_temp_pre_evap.update(T_current, DT)
else: pct_sp = pct_current

# *** UPDATE VALVE PHYSICS ***
pct_new = update_valve_physics(pct_sp, pct_current, 10.0, DT)
if mode == "Manual": pct_new = st.session_state.pre_evap_state['pct_valve']

T_shell_steam_c, Lh_shell_steam_j=get_steam_properties_pre_evap(Psh_current)
steam_flow_kgps=compute_steam_flow_kgps_pre_evap(pct_new, MASTER_P_EVAPORATOR_PRE_EVAP, Psh_current)
delta_T1=T_shell_steam_c-T_current; delta_T2=T_shell_steam_c-evap_feed_temp_c
LMTD=(delta_T1-delta_T2)/np.log(delta_T1/delta_T2) if abs(delta_T1-delta_T2)>1e-6 and delta_T1>0 and delta_T2>0 else (delta_T1 if delta_T1>0 else 0.0)
U_value = calculate_U_value_pre_evap(m_gps)
Q_actual_transferred_kw=(U_value * A_HEAT_TRANSFER_AREA_PRE_EVAP * LMTD)/1000
Q_effective=Q_actual_transferred_kw
Psh_target, shell_steam_mass_target=calculate_shell_pressure_ideal_gas_pre_evap(steam_flow_kgps, SHELL_VOL_M3_PRE_EVAP, T_shell_steam_c, Q_effective, Lh_shell_steam_j, shell_steam_mass_current, DT)
Pvac_target=calculate_vacuum_pressure_tuned_pre_evap(pct_vac_valve, st.session_state.last_evap_rate_pre_evap)
Tprod_target=solve_for_T_enhanced_pre_evap(Q_effective, evap_feed_temp_c, cp0, m_gps, m_urea_gps, Pvac_target, evap_feed_conc_pct, T_shell_steam_c)
Cprod_target=conc_from_TP_evap_pre_evap(Tprod_target, Pvac_target, C_current) if evap_feed_conc_pct > 0.1 else 0.0
temp_streams=calculate_streams_corrected_pre_evap(m_gps, m_urea_gps, Tprod_target, Cprod_target)
rho_prod=rho_from_conc_pre_evap(C_current); prod_mass_flow_kgps=temp_streams['s15']['mass']/3600
vol_flow_out_m3ps=prod_mass_flow_kgps/rho_prod if rho_prod>0 else 0.01
concentration_tc=TUBE_VOL_M3_PRE_EVAP/vol_flow_out_m3ps if vol_flow_out_m3ps>0 else 120.0
evap_vapor_density = get_steam_density_pre_evap(Pvac_current, T_current)
vol_flow_evap_m3ps = (st.session_state.last_evap_rate_pre_evap / 3600) / evap_vapor_density if evap_vapor_density > 0 else 0.1
vacuum_tc = FLASH_VESSEL_VOL_M3_PRE_EVAP / vol_flow_evap_m3ps if vol_flow_evap_m3ps > 0 else 50.0
steam_density = get_steam_density_pre_evap(Psh_current, T_shell_steam_c)
vol_flow_steam_in_m3ps = steam_flow_kgps / steam_density if steam_density > 0 else 0.1
pressure_tc = SHELL_VOL_M3_PRE_EVAP / vol_flow_steam_in_m3ps if vol_flow_steam_in_m3ps > 0 else 30.0
Bprod_target=calculate_biuret_kinetic_pre_evap(evap_feed_biuret_pct,evap_feed_conc_pct,Tprod_target,Cprod_target,concentration_tc)
thermal_tc = 20.0
d_Pvac_dt=(Pvac_target-Pvac_current)/vacuum_tc; Pvac_new=Pvac_current+d_Pvac_dt*DT; d_Psh_dt=(Psh_target-Psh_current)/pressure_tc; Psh_new=Psh_current+d_Psh_dt*DT
d_T_dt=(Tprod_target-T_current)/thermal_tc; Tprod_new=T_current+d_T_dt*DT; d_C_dt=(Cprod_target-C_current)/concentration_tc; Cprod_new=C_current+d_C_dt*DT
d_B_dt=(Bprod_target-B_current)/concentration_tc; Bprod_new=B_current+d_B_dt*DT
d_mass_dt=(shell_steam_mass_target-shell_steam_mass_current)/pressure_tc; shell_steam_mass_new=shell_steam_mass_current+d_mass_dt*DT
st.session_state.pre_evap_state={'Tprod_c':Tprod_new,'Psh_barg':Psh_new,'Cprod_pct':Cprod_new,'Bprod_pct':Bprod_new,'pct_valve':pct_new,'Pvac_barg':Pvac_new,'shell_steam_mass_kg':shell_steam_mass_new}
final_streams=calculate_streams_corrected_pre_evap(m_gps, m_urea_gps, Tprod_new, Cprod_new); st.session_state.last_evap_rate_pre_evap=final_streams['s16']['mass']
stream4_mass_kgph=final_streams['s15']['mass']; stream4_conc=Cprod_new; stream4_biuret_conc=Bprod_new; stream4_temp=Tprod_new
stream4_rho=rho_from_conc_pre_evap(stream4_conc+stream4_biuret_conc); stream4_vol=stream4_mass_kgph/stream4_rho if stream4_rho>0 else 0
stream44_mass_kgph=final_streams['s16']['mass']

# =============================================================================
# ===== MIXER LOGIC (Stream 20 + Stream 21 -> Stream 23 -> Stream 5) ==========
# =============================================================================
# 1. Stream 21 Properties (Process Condensate)
s21_vol_flow = stream21_flow
s21_rho = 960.0  # Hot water approximation
s21_mass_kgph = s21_vol_flow * s21_rho
s21_temp = 100.0 # Condensate temp
s21_conc = 0.0
s21_biuret = 0.0
s21_cp = 4.18

# 2. Stream 20 Properties (Recycle from Melt Tank - From Previous Loop)
s20_props = st.session_state.stream20_props
s20_mass_kgph = s20_props['mass_kgph']
s20_temp = s20_props['temp_c']
s20_conc = s20_props['conc_pct']
s20_biuret = s20_props['biuret_pct']
s20_cp = cp_from_conc_pre_evap(s20_conc + s20_biuret)

# 3. Mixer Calculations (Stream 23)
s23_mass_kgph = s21_mass_kgph + s20_mass_kgph

if s23_mass_kgph > 0.01:
    # Mass Balance for components
    s23_urea_flow = (s21_mass_kgph * s21_conc/100.0) + (s20_mass_kgph * s20_conc/100.0)
    s23_biuret_flow = (s21_mass_kgph * s21_biuret/100.0) + (s20_mass_kgph * s20_biuret/100.0)
    
    s23_conc = (s23_urea_flow / s23_mass_kgph) * 100.0
    s23_biuret = (s23_biuret_flow / s23_mass_kgph) * 100.0
    
    # Energy Balance
    Q21 = s21_mass_kgph * s21_cp * s21_temp
    Q20 = s20_mass_kgph * s20_cp * s20_temp
    
    s23_cp = cp_from_conc_pre_evap(s23_conc + s23_biuret)
    s23_temp = (Q21 + Q20) / (s23_mass_kgph * s23_cp)
else:
    s23_conc = 0.0; s23_biuret = 0.0; s23_temp = 25.0

# 4. Map Stream 23 to Stream 5 (Input to Tank System)
stream5_mass_kgph = s23_mass_kgph
stream5_temp_c = s23_temp
stream5_conc_pct = s23_conc
stream5_biuret_pct = s23_biuret

# --- End of Mixer Logic ---

stream4_urea_kgph = stream4_mass_kgph * (stream4_conc / 100.0)
stream4_biuret_kgph = stream4_mass_kgph * (stream4_biuret_conc / 100.0)
stream4_cp = cp_from_conc_pre_evap(stream4_conc + stream4_biuret_conc)
stream4_Q = stream4_mass_kgph * stream4_cp * stream4_temp
stream5_urea_kgph = stream5_mass_kgph * (stream5_conc_pct / 100.0)
stream5_biuret_kgph = stream5_mass_kgph * (stream5_biuret_pct / 100.0)
stream5_cp = cp_from_conc_pre_evap(stream5_conc_pct + stream5_biuret_pct)
stream5_Q = stream5_mass_kgph * stream5_cp * stream5_temp_c
stream6_mass_kgph = stream4_mass_kgph + stream5_mass_kgph
stream6_urea_kgph = stream4_urea_kgph + stream5_urea_kgph
stream6_biuret_kgph = stream4_biuret_kgph + stream5_biuret_kgph
stream6_Q_total = stream4_Q + stream5_Q
stream6_mass_cp_total = (stream4_mass_kgph * stream4_cp) + (stream5_mass_kgph * stream5_cp)
if stream6_mass_kgph > 0.01:
    stream6_conc = (stream6_urea_kgph / stream6_mass_kgph) * 100.0
    stream6_biuret_conc = (stream6_biuret_kgph / stream6_mass_kgph) * 100.0
else: stream6_conc = 0.0; stream6_biuret_conc = 0.0
if stream6_mass_cp_total > 0.01: stream6_temp = stream6_Q_total / stream6_mass_cp_total
else: stream6_temp = 25.0
stream6_rho = rho_from_conc_pre_evap(stream6_conc + stream6_biuret_conc)
if stream6_rho > 0: stream6_vol_m3ph = stream6_mass_kgph / stream6_rho
else: stream6_vol_m3ph = 0.0
stream6_vol = stream6_vol_m3ph
tank_max_vol=80.0; tank_diameter=2*np.sqrt(tank_max_vol/(np.pi*5.0))
tank_level_percent = (st.session_state.tank_volume / tank_max_vol) * 100 if tank_max_vol > 0 else 0
VALVE_RESPONSE_TIME_SEC=15.0; dt_loop=1.0
st.session_state.fv_324401_actual+=(st.session_state.fv_324401_sp-st.session_state.fv_324401_actual)*dt_loop/VALVE_RESPONSE_TIME_SEC
st.session_state.fv_324401_actual = np.clip(st.session_state.fv_324401_actual, 0, 100)
max_flow_out_valve = st.session_state.fv_324401_actual * 1.1
if close_323P003:
    stream10_vol_flow = 0.0
    if close_323P003_suction: stream7_vol = 0.0
    else: stream7_vol = -stream8_condensate
elif close_323P003_suction:
    stream7_vol = 0.0
    stream10_vol_flow = min(stream8_condensate, max_flow_out_valve) 
elif st.session_state.pump_running and not pump_fault:
    stream10_vol_flow = max_flow_out_valve
    stream7_vol = stream10_vol_flow - stream8_condensate
else:
    if stream8_condensate > 0:
        flow_to_discharge_demand = 0.34 * stream8_condensate
        if max_flow_out_valve >= flow_to_discharge_demand:
            stream10_vol_flow = flow_to_discharge_demand
            stream7_vol = - (stream8_condensate - stream10_vol_flow)
        else:
            stream10_vol_flow = max_flow_out_valve
            stream7_vol = -(stream8_condensate - stream10_vol_flow) 
    else: stream10_vol_flow = 0.0; stream7_vol = 0.0

if stream7_vol<0: stream_backflow_vol=abs(stream7_vol); stream7_vol=0.0
else: stream_backflow_vol=0.0
stream10_vol_flow_m3ps = stream10_vol_flow / 3600.0
if stream10_vol_flow_m3ps > 1e-6:
    V_total_mix=stream10_vol_flow_m3ps*DT; V_pipe_section=0.3
    C_urea_pipe=st.session_state.stream10_conc_last; C_biuret_pipe=st.session_state.stream10_biuret_last; T_pipe=st.session_state.stream10_temp_last
    if stream7_vol > 0: C_in = st.session_state.tank_conc; B_in = st.session_state.tank_biuret_conc; T_in = st.session_state.tank_temp
    else: C_in = 0.0; B_in = 0.0; T_in = 100.0
    V_in_total = (stream7_vol + stream8_condensate) / 3600.0 * DT
    if V_in_total > 0:
        stream10_conc=(C_urea_pipe*V_pipe_section + C_in*V_in_total)/(V_pipe_section+V_in_total)
        stream10_biuret=(C_biuret_pipe*V_pipe_section + B_in*V_in_total)/(V_pipe_section+V_in_total)
        stream10_temp=(T_pipe*V_pipe_section + T_in*V_in_total)/(V_pipe_section+V_in_total)
    else: stream10_conc=C_urea_pipe; stream10_biuret=C_biuret_pipe; stream10_temp=T_pipe
else: 
    stream10_conc=st.session_state.stream10_conc_last; stream10_biuret=st.session_state.stream10_biuret_last; stream10_temp=max(25.0,st.session_state.stream10_temp_last-(1.5/60.0)*DT)
st.session_state.stream10_conc_last=stream10_conc
st.session_state.stream10_biuret_last=stream10_biuret
st.session_state.stream10_temp_last=stream10_temp
stream10_rho = rho_from_conc_pre_evap(stream10_conc)
stream10_mass_flow_kgph = stream10_vol_flow * stream10_rho
outflow_m3ps=stream7_vol/3600.0; inflow_m3ps=stream6_vol/3600.0
backflow_m3ps = stream_backflow_vol / 3600.0
backflow_rho = rho_from_conc_pre_evap(0.0); backflow_mass_in = backflow_m3ps * backflow_rho
tank_rho=rho_from_conc_pre_evap(st.session_state.tank_conc+st.session_state.tank_biuret_conc)
mass_tank=st.session_state.tank_volume*tank_rho; urea_tank=mass_tank*st.session_state.tank_conc/100; biuret_tank=mass_tank*st.session_state.tank_biuret_conc/100
mass_out=outflow_m3ps*tank_rho; urea_out=mass_out*st.session_state.tank_conc/100; biuret_out=mass_out*st.session_state.tank_biuret_conc/100
mass_in_1=inflow_m3ps*stream6_rho; urea_in_1=mass_in_1*stream6_conc/100; biuret_in_1=mass_in_1*stream6_biuret_conc/100
new_mass_tank=mass_tank+(mass_in_1 + backflow_mass_in - mass_out)*DT
new_urea_tank=urea_tank+(urea_in_1 - urea_out)*DT
new_biuret_tank=biuret_tank+(biuret_in_1 - biuret_out)*DT
if new_mass_tank>0:
    new_total_conc = ((new_urea_tank+new_biuret_tank)/new_mass_tank)*100
    new_rho = rho_from_conc_pre_evap(new_total_conc)
    st.session_state.tank_conc=(new_urea_tank/new_mass_tank)*100; st.session_state.tank_biuret_conc=(new_biuret_tank/new_mass_tank)*100
    st.session_state.tank_volume=new_mass_tank/new_rho if new_rho > 0 else 0
else: st.session_state.tank_conc=0; st.session_state.tank_biuret_conc=0; st.session_state.tank_volume=0
st.session_state.tank_volume = np.clip(st.session_state.tank_volume, 0, tank_max_vol)
cp_tank=cp_from_conc_pre_evap(st.session_state.tank_conc+st.session_state.tank_biuret_conc); Q_tank=mass_tank*cp_tank*st.session_state.tank_temp
cp_in_1=cp_from_conc_pre_evap(stream6_conc+stream6_biuret_conc); Q_in=(mass_in_1*cp_in_1*stream6_temp)
cp_backflow=cp_from_conc_pre_evap(0.0); Q_backflow = backflow_mass_in * cp_backflow * 100.0
Q_out=mass_out*cp_tank*st.session_state.tank_temp; Q_new=Q_tank+(Q_in + Q_backflow - Q_out)*DT
if new_mass_tank>0 and cp_from_conc_pre_evap(st.session_state.tank_conc)>0:
    new_cp_tank=cp_from_conc_pre_evap(st.session_state.tank_conc+st.session_state.tank_biuret_conc)
    st.session_state.tank_temp = Q_new/(new_mass_tank*new_cp_tank) if (new_mass_tank*new_cp_tank)>0 else 25.0
else: st.session_state.tank_temp=25.0
active_alarms = []
if tank_level_percent > 95.0: active_alarms.append(f"🔴 HIGH-HIGH TANK LEVEL: {tank_level_percent:.1f}% (SP: < 95%)")
if tank_level_percent < 5.0: active_alarms.append(f"🔴 LOW-LOW TANK LEVEL: {tank_level_percent:.1f}% (SP: > 5%)")
if T_current > 115.0: active_alarms.append(f"🟠 PRE-EVAP HIGH TEMP: {T_current:.1f}°C (SP: < 115°C)")
if Psh_current > 2.0: active_alarms.append(f"🟠 PRE-EVAP HIGH SHELL PRESS: {Psh_current:.2f} barg (SP: < 2.0 barg)")
if B_current > 0.9: active_alarms.append(f"🟡 PRE-EVAP HIGH BIURET: {B_current:.3f}% (SP: < 0.9%)")
if not st.session_state.pump_running and fic_sp > 0 and valve_mode == 'Auto': active_alarms.append("⚫ PUMP IS OFF but flow is required.")

# *** PIPE LAG SIMULATION (Stream 10 -> Stream 11) using new function ***
s10_props_target = {
    'mass_flow_kgph': stream10_mass_flow_kgph, 'temp_c': stream10_temp, 
    'conc_pct': stream10_conc, 'biuret_pct': stream10_biuret, 
    'rho_kg_m3': stream10_rho, 'vol_flow_m3ph': stream10_vol_flow
}
st.session_state.stream11_props_lagged = update_pipe_lag(
    st.session_state.stream11_props_lagged, 
    s10_props_target, 
    pipe_vol_m3=PIPE_VOL_M3, 
    dt_s=DT
)

# =============================================================================
# ===== (CODE 2) EVAP 1 CALCULATION LOOP ======================================
# =============================================================================
s11_inputs = st.session_state.stream11_props_lagged
F_evap1 = s11_inputs['vol_flow_m3ph']; C0_evap1 = s11_inputs['conc_pct']; T0_evap1 = s11_inputs['temp_c']; B0_total_evap1 = s11_inputs['biuret_pct']
rho_evap1 = rho_from_conc_evap(C0_evap1); m_h_evap1 = F_evap1*rho_evap1; m_gps_evap1 = m_h_evap1/3600.0
m_urea_gps_evap1 = m_gps_evap1*(C0_evap1/100.0)

T_current_evap1 = st.session_state.evap1_state['Tprod_c']; Psh_current_evap1 = st.session_state.evap1_state['Psh_barg']
Psh_display_current_evap1 = st.session_state.evap1_state['Psh_display_barg']; C_current_evap1 = st.session_state.evap1_state['Cprod_pct']
B_current_evap1 = st.session_state.evap1_state['Bprod_pct']; pct_current_evap1 = st.session_state.evap1_state['pct_valve']
Pvac_current_evap1 = st.session_state.evap1_state['Pvac_barg']; shell_steam_mass_current_evap1 = st.session_state.evap1_state['shell_steam_mass_kg']

if mode_evap1 == "Auto": pct_sp_evap1 = st.session_state.pid_pressure_evap1.update(Psh_display_current_evap1, DT)
elif mode_evap1 == "Cascade": pct_sp_evap1 = st.session_state.pid_temp_evap1.update(T_current_evap1, DT)
else: pct_sp_evap1 = pct_current_evap1

# *** UPDATE VALVE PHYSICS ***
pct_new_evap1 = update_valve_physics(pct_sp_evap1, pct_current_evap1, 15.0, DT)
if mode_evap1 == "Manual": pct_new_evap1 = st.session_state.evap1_state['pct_valve']

effective_max_steam_kgph_evap1 = max_steam_flow_kgph_evap1 * (iso_valve_steam_evap1 / 100.0)
effective_vac_pct_evap1 = pct_vac_valve_evap1 * (iso_valve_vac_evap1 / 100.0)
steam_flow_kgps_evap1 = compute_steam_flow_kgps_evap(pct_new_evap1, effective_max_steam_kgph_evap1, master_p_evap1, Psh_current_evap1)
U_actual_evap1 = U_HEAT_TRANSFER_COEFF_EVAP1 
# *** UPDATED PHYSICS CALL WITH WIDGET VALUES ***
Psh_new_evap1, shell_steam_mass_new_evap1, Q_actual_transferred_kw_evap1 = calculate_shell_physics_evap(steam_flow_kgps_evap1, shell_vol_m3_EVAP1, Psh_current_evap1, T_current_evap1, T0_evap1, U_actual_evap1, A_HEAT_TRANSFER_AREA_EVAP1, DT)
if shell_vent_evap1: Psh_new_evap1 = max(Psh_new_evap1, 0.0)
cp_flow_avg = ((cp_from_conc_evap(C0_evap1) + cp_from_conc_evap(C_current_evap1))/2.0) * 1000.0
conc_equilibrium = conc_from_TP_evap_main(T_current_evap1, Pvac_current_evap1, C0_evap1)
if conc_equilibrium < C0_evap1: conc_equilibrium = C0_evap1
if conc_equilibrium > C0_evap1 and m_gps_evap1 > 0:
    m_out_liquid = (m_gps_evap1 * C0_evap1) / conc_equilibrium
    m_evap_rate_actual = m_gps_evap1 - m_out_liquid
else: m_evap_rate_actual = 0.0
Q_input_j = Q_actual_transferred_kw_evap1 * 1000.0
Q_sensible = m_gps_evap1 * cp_flow_avg * (T_current_evap1 - T0_evap1)
Q_evap_loss = m_evap_rate_actual * 2260000.0 
Net_Power_J = Q_input_j - Q_sensible - Q_evap_loss
thermal_mass_tubes = tube_vol_m3_EVAP1 * rho_from_conc_evap(C_current_evap1) * cp_flow_avg 
dT_dt_evap1 = Net_Power_J / max(1.0, thermal_mass_tubes)
Tprod_new_evap1 = T_current_evap1 + dT_dt_evap1 * DT
m_out_kgps = max(0, m_gps_evap1 - m_evap_rate_actual)
rho_out = rho_from_conc_evap(C_current_evap1)
vol_flow_out_m3ps = m_out_kgps / rho_out if rho_out > 0 else 0.01
concentration_tc_evap1 = tube_vol_m3_EVAP1 / vol_flow_out_m3ps if vol_flow_out_m3ps > 0 else 120.0

# *** UPDATE MIXING PHYSICS ***
effective_liquid_vol_1 = tube_vol_m3_EVAP1 + (0.2 * FLASH_VESSEL_VOL_M3_EVAP1)
Cprod_new_evap1 = update_mixing_lag(C_current_evap1, conc_equilibrium, effective_liquid_vol_1, vol_flow_out_m3ps*3600, DT)

evap_vapor_density_evap1 = get_steam_density_evap(Pvac_current_evap1, T_current_evap1)
vol_flow_evap_m3ps_evap1 = (m_evap_rate_actual) / evap_vapor_density_evap1 if evap_vapor_density_evap1 > 0 else 0.1
vacuum_tc_evap1 = FLASH_VESSEL_VOL_M3_EVAP1 / vol_flow_evap_m3ps_evap1 if vol_flow_evap_m3ps_evap1 > 0 else 50.0
Pvac_target_evap1 = calculate_vacuum_pressure_evap(effective_vac_pct_evap1, master_p_evap1, m_evap_rate_actual * 3600, vac_params)
d_Pvac_dt_evap1 = (Pvac_target_evap1 - Pvac_current_evap1) / vacuum_tc_evap1
Pvac_new_evap1 = Pvac_current_evap1 + d_Pvac_dt_evap1 * DT
residence_time_s_evap1 = concentration_tc_evap1
Bprod_target_evap1 = calculate_biuret_kinetic_evap_generic(B0_total_evap1, C0_evap1, Tprod_new_evap1, Cprod_new_evap1, residence_time_s_evap1, A_BIURET_EVAP1, Ea_BIURET_EVAP1, R_GAS_EVAP)

# *** UPDATE MIXING PHYSICS FOR BIURET ***
Bprod_new_evap1 = update_mixing_lag(B_current_evap1, Bprod_target_evap1, effective_liquid_vol_1, vol_flow_out_m3ps*3600, DT)

d_Psh_display_dt_evap1 = (Psh_new_evap1 - Psh_display_current_evap1) / max(5.0, DT)
Psh_new_display_evap1 = Psh_display_current_evap1 + d_Psh_display_dt_evap1 * DT
st.session_state.evap1_state = { 'Tprod_c': Tprod_new_evap1, 'Psh_barg': Psh_new_evap1, 'Psh_display_barg': Psh_new_display_evap1, 'Cprod_pct': Cprod_new_evap1, 'Bprod_pct': Bprod_new_evap1, 'pct_valve': pct_new_evap1, 'Pvac_barg': Pvac_new_evap1, 'shell_steam_mass_kg': shell_steam_mass_new_evap1 }
final_streams_evap1 = calculate_streams_corrected_evap_generic(m_gps_evap1, m_urea_gps_evap1, Tprod_new_evap1, Cprod_new_evap1, T0_evap1, C0_evap1, Q_actual_transferred_kw_evap1)
st.session_state.evap1_last_evap_rate = final_streams_evap1['mass_vapor']

# =============================================================================
# ===== (CODE 3) EVAP 2 CALCULATION LOOP (UPDATED) ============================
# =============================================================================
# Input: Output of Evap 1 (Stream 12 became Stream 14)
# Stream 14 properties (Feed to Evap 2)
m_gps_evap2 = final_streams_evap1['mass_out'] / 3600
C0_evap2 = final_streams_evap1['conc_out']
T0_evap2 = final_streams_evap1['temp_out']
B0_total_evap2 = st.session_state.evap1_state['Bprod_pct']
m_urea_gps_evap2 = m_gps_evap2 * (C0_evap2 / 100.0)

# State retrieval
T_current_evap2 = st.session_state.evap2_state['Tprod_c']; Psh_current_evap2 = st.session_state.evap2_state['Psh_barg']
Psh_display_current_evap2 = st.session_state.evap2_state['Psh_display_barg']; C_current_evap2 = st.session_state.evap2_state['Cprod_pct']
B_current_evap2 = st.session_state.evap2_state['Bprod_pct']; pct_current_evap2 = st.session_state.evap2_state['pct_valve']
Pvac_current_evap2 = st.session_state.evap2_state['Pvac_barg']; shell_steam_mass_current_evap2 = st.session_state.evap2_state['shell_steam_mass_kg']

# PID Control
if mode_evap2 == "Auto": pct_sp_evap2 = st.session_state.pid_pressure_evap2.update(Psh_display_current_evap2, DT)
elif mode_evap2 == "Cascade": pct_sp_evap2 = st.session_state.pid_temp_evap2.update(T_current_evap2, DT)
else: pct_sp_evap2 = pct_current_evap2

# *** UPDATE VALVE PHYSICS ***
pct_new_evap2 = update_valve_physics(pct_sp_evap2, pct_current_evap2, 15.0, DT)
if mode_evap2 == "Manual": pct_new_evap2 = st.session_state.evap2_state['pct_valve']

# Physics
effective_max_steam_kgph_evap2 = max_steam_flow_kgph_evap2 * (iso_valve_steam_evap2 / 100.0)
effective_vac_pct_evap2 = pct_vac_valve_evap2 * (iso_valve_vac_evap2 / 100.0)
steam_flow_kgps_evap2 = compute_steam_flow_kgps_evap(pct_new_evap2, effective_max_steam_kgph_evap2, master_p_evap2, Psh_current_evap2)
U_actual_evap2 = U_HEAT_TRANSFER_COEFF_EVAP2 
# *** UPDATED PHYSICS CALL WITH WIDGET VALUES ***
Psh_new_evap2, shell_steam_mass_new_evap2, Q_actual_transferred_kw_evap2 = calculate_shell_physics_evap(steam_flow_kgps_evap2, shell_vol_m3_EVAP2, Psh_current_evap2, T_current_evap2, T0_evap2, U_actual_evap2, A_HEAT_TRANSFER_AREA_EVAP2, DT)
if shell_vent_evap2: Psh_new_evap2 = max(Psh_new_evap2, 0.0)

cp_flow_avg_2 = ((cp_from_conc_evap(C0_evap2) + cp_from_conc_evap(C_current_evap2))/2.0) * 1000.0
conc_equilibrium_2 = conc_from_TP_evap_main(T_current_evap2, Pvac_current_evap2, C0_evap2)
if conc_equilibrium_2 < C0_evap2: conc_equilibrium_2 = C0_evap2
if conc_equilibrium_2 > C0_evap2 and m_gps_evap2 > 0:
    m_out_liquid_2 = (m_gps_evap2 * C0_evap2) / conc_equilibrium_2
    m_evap_rate_actual_2 = m_gps_evap2 - m_out_liquid_2
else: m_evap_rate_actual_2 = 0.0
Q_input_j_2 = Q_actual_transferred_kw_evap2 * 1000.0
Q_sensible_2 = m_gps_evap2 * cp_flow_avg_2 * (T_current_evap2 - T0_evap2)
Q_evap_loss_2 = m_evap_rate_actual_2 * 2260000.0 
Net_Power_J_2 = Q_input_j_2 - Q_sensible_2 - Q_evap_loss_2
thermal_mass_tubes_2 = tube_vol_m3_EVAP2 * rho_from_conc_evap(C_current_evap2) * cp_flow_avg_2
dT_dt_evap2 = Net_Power_J_2 / max(1.0, thermal_mass_tubes_2)
Tprod_new_evap2 = T_current_evap2 + dT_dt_evap2 * DT

m_out_kgps_2 = max(0, m_gps_evap2 - m_evap_rate_actual_2)
rho_out_2 = rho_from_conc_evap(C_current_evap2)
vol_flow_out_m3ps_2 = m_out_kgps_2 / rho_out_2 if rho_out_2 > 0 else 0.01
concentration_tc_evap2 = tube_vol_m3_EVAP2 / vol_flow_out_m3ps_2 if vol_flow_out_m3ps_2 > 0 else 120.0

# *** UPDATE MIXING PHYSICS ***
effective_liquid_vol_2 = tube_vol_m3_EVAP2 + (0.2 * FLASH_VESSEL_VOL_M3_EVAP2)
Cprod_new_evap2 = update_mixing_lag(C_current_evap2, conc_equilibrium_2, effective_liquid_vol_2, vol_flow_out_m3ps_2*3600, DT)

evap_vapor_density_evap2 = get_steam_density_evap(Pvac_current_evap2, T_current_evap2)
vol_flow_evap_m3ps_evap2 = (m_evap_rate_actual_2) / evap_vapor_density_evap2 if evap_vapor_density_evap2 > 0 else 0.1
vacuum_tc_evap2 = FLASH_VESSEL_VOL_M3_EVAP2 / vol_flow_evap_m3ps_evap2 if vol_flow_evap_m3ps_evap2 > 0 else 50.0

# *** CHANGE: Call the new Special Physics Function with MASTER_P_EVAP1 linkage ***
Pvac_target_evap2 = calculate_vacuum_pressure_evap2_special(effective_vac_pct_evap2, m_evap_rate_actual_2 * 3600, master_p_evap1)
d_Pvac_dt_evap2 = (Pvac_target_evap2 - Pvac_current_evap2) / vacuum_tc_evap2
Pvac_new_evap2 = Pvac_current_evap2 + d_Pvac_dt_evap2 * DT

residence_time_s_evap2 = concentration_tc_evap2
Bprod_target_evap2 = calculate_biuret_kinetic_evap_generic(B0_total_evap2, C0_evap2, Tprod_new_evap2, Cprod_new_evap2, residence_time_s_evap2, A_BIURET_EVAP2, Ea_BIURET_EVAP2, R_GAS_EVAP)

# *** UPDATE MIXING PHYSICS FOR BIURET ***
Bprod_new_evap2 = update_mixing_lag(B_current_evap2, Bprod_target_evap2, effective_liquid_vol_2, vol_flow_out_m3ps_2*3600, DT)

d_Psh_display_dt_evap2 = (Psh_new_evap2 - Psh_display_current_evap2) / max(5.0, DT)
Psh_new_display_evap2 = Psh_display_current_evap2 + d_Psh_display_dt_evap2 * DT

st.session_state.evap2_state = { 'Tprod_c': Tprod_new_evap2, 'Psh_barg': Psh_new_evap2, 'Psh_display_barg': Psh_new_display_evap2, 'Cprod_pct': Cprod_new_evap2, 'Bprod_pct': Bprod_new_evap2, 'pct_valve': pct_new_evap2, 'Pvac_barg': Pvac_new_evap2, 'shell_steam_mass_kg': shell_steam_mass_new_evap2 }
final_streams_evap2 = calculate_streams_corrected_evap_generic(m_gps_evap2, m_urea_gps_evap2, Tprod_new_evap2, Cprod_new_evap2, T0_evap2, C0_evap2, Q_actual_transferred_kw_evap2)
st.session_state.evap2_last_evap_rate = final_streams_evap2['mass_vapor']

# =============================================================================
# ===== (CODE 4) MELT TANK SYSTEM CALCULATION LOOP ============================
# =============================================================================

# 1. Gather Inputs
# Stream 15 (From Evap 2)
s15_in = final_streams_evap2
m_s15_kgph = s15_in['mass_out']
T_s15 = s15_in['temp_out']
C_s15 = s15_in['conc_out'] 
B_s15 = st.session_state.evap2_state['Bprod_pct'] 

# Stream 17 (Condensate on Pump)
if not st.session_state.melt_system['suction_open_cmd'] and not st.session_state.melt_system['discharge_open_cmd']:
    flow_s17_m3ph = 0.0
else:
    flow_s17_m3ph = 10.0 * (st.session_state.melt_system['valve_condensate_pos'] / 100.0)
rho_s17 = 958.0 # Water at 100C
m_s17_kgph = flow_s17_m3ph * rho_s17
T_s17 = 100.0; C_s17 = 0.0; B_s17 = 0.0

# 2. Pump Logic & Interlocks
# Trip Condition: Drain Open OR Suction Closed
trip_condition_active = st.session_state.melt_system['drain_open_cmd'] or (not st.session_state.melt_system['suction_open_cmd'])

if trip_condition_active:
    st.session_state.melt_system['pump_trip_timer'] += DT
    if st.session_state.melt_system['pump_trip_timer'] >= 10.0:
        if st.session_state.melt_system['pump_running']:
            st.session_state.active_alarms.append("🔴 MELT PUMP TRIPPED (Suction Closed or Drain Open)")
            st.session_state.melt_system['pump_running'] = False
            st.session_state.melt_system['pump_trip_active'] = True
else:
    st.session_state.melt_system['pump_trip_timer'] = 0.0
    st.session_state.melt_system['pump_trip_active'] = False

# Low Level Trip
if st.session_state.melt_system['level_pct'] < 5.0:
    if st.session_state.melt_system['pump_running']:
        st.session_state.active_alarms.append("🔴 MELT PUMP TRIPPED (Low Level < 5%)")
        st.session_state.melt_system['pump_running'] = False

# =============================================================================
# 3. Main Header Pressure Calculation (New Physics Model)
# =============================================================================
# Flow from Pump Discharge
current_s19_flow_m3ph = st.session_state.melt_system['stream19_flow_m3ph']

# Retrieve System Configuration
hv_335602_pos = st.session_state.melt_system.get('hv_335602_pos', 0.0)
active_headers = st.session_state.melt_system.get('active_headers', 12)

# Calculate Equivalent Headers
valve_capacity_equivalent = (hv_335602_pos / 100.0) * 7.0
total_equivalent_headers = active_headers + valve_capacity_equivalent

# Calculate Load per Header
if total_equivalent_headers > 0:
    load_per_header = current_s19_flow_m3ph / total_equivalent_headers
else:
    load_per_header = 0.0

# Pressure Equation: P = 1.5 + (0.333 * Load)
calculated_pressure = 1.5 + (0.3333 * load_per_header)

if current_s19_flow_m3ph <= 0.1:
    calculated_pressure = 0.0 

# Apply Max Limit (6 barg) and Lag Filter
instant_pressure = min(calculated_pressure, 6.0)

# First-Order Lag Filter (Smoothing)
tau_pressure = 2.0 
prev_pressure = st.session_state.melt_system.get('header_pressure_filtered', 3.6)
filtered_pressure = prev_pressure + (instant_pressure - prev_pressure) * (DT / tau_pressure)
st.session_state.melt_system['header_pressure_filtered'] = filtered_pressure

# Use filtered pressure for control logic
est_header_pressure = filtered_pressure 

# =============================================================================
# 4. Control Logic (LIC-324501) - CORRECTED & STABLE
# =============================================================================
current_level = st.session_state.melt_system['level_pct']

# Direct Acting Fix (PV - SP) for discharge
sp = st.session_state.pid_level_melt.setpoint
inverted_pv = (2.0 * sp) - current_level

# Auto-Range Fix
st.session_state.pid_level_melt.max_output = 200.0
pid_demand = st.session_state.pid_level_melt.update(inverted_pv, DT)

mode = st.session_state.lic_324501_mode

# Smooth Override Logic
if est_header_pressure >= 3.8:
    pressure_limit_factor = 0.0
elif est_header_pressure <= 3.6:
    pressure_limit_factor = 1.0
else:
    pressure_limit_factor = (3.8 - est_header_pressure) / (3.8 - 3.6)

if mode == "Manual":
    final_A = st.session_state.melt_system['valve_A_pos']
    final_B = st.session_state.melt_system['valve_B_pos']

elif mode == "Auto":
    user_set_A = st.session_state.melt_system['valve_A_pos']
    final_A = min(user_set_A, user_set_A * pressure_limit_factor)
    final_B = np.clip(pid_demand, 0.0, 100.0) 

elif mode == "Cascade":
    raw_demand_A = pid_demand 
    target_A_unlimited = np.clip(raw_demand_A, 0.0, 100.0)
    final_A = target_A_unlimited * pressure_limit_factor
    deficit = pid_demand - final_A
    final_B = deficit
    st.session_state.melt_system['high_pressure_override_active'] = (pressure_limit_factor < 1.0)

# Safety Clamps
final_A = np.clip(final_A, 0.0, 100.0)
final_B = np.clip(final_B, 0.0, 100.0)

st.session_state.melt_system['valve_A_pos'] = final_A
st.session_state.melt_system['valve_B_pos'] = final_B

target_A = final_A
target_B = final_B

# 5. Output Flows Calculation (Stream 19, 20, 18)
# Stream 18 (Drain)
flow_s18_m3ph = 0.0
if st.session_state.melt_system['drain_open_cmd'] and st.session_state.melt_system['suction_open_cmd']:
    h_liquid = MELT_TANK_HEIGHT_M * (current_level / 100.0)
    if h_liquid > 0:
        v_eff = 0.6 * math.sqrt(2 * 9.81 * h_liquid)
        flow_s18_m3s = DRAIN_AREA_M2 * v_eff
        flow_s18_m3ph = flow_s18_m3s * 3600.0
    else: flow_s18_m3ph = 0.0

# Stream 19 (Prod) & 20 (Recycle)
can_pump_flow = st.session_state.melt_system['pump_running'] and \
                st.session_state.melt_system['suction_open_cmd'] and \
                st.session_state.melt_system['discharge_open_cmd']

if can_pump_flow:
    req_flow_A = target_A * 1.1 # 100% = 110 m3/h
    req_flow_B = target_B * 1.1
    total_req = req_flow_A + req_flow_B
    
    if total_req > MELT_PUMP_MAX_CAPACITY:
        ratio = MELT_PUMP_MAX_CAPACITY / total_req
        flow_s19_m3ph = req_flow_A * ratio
        flow_s20_m3ph = req_flow_B * ratio
    else:
        flow_s19_m3ph = req_flow_A
        flow_s20_m3ph = req_flow_B
else:
    flow_s19_m3ph = 0.0
    flow_s20_m3ph = 0.0

st.session_state.melt_system['stream19_flow_m3ph'] = flow_s19_m3ph

# 6. Mass & Energy Balance (Tank)
# Densities
rho_mix_tank = rho_from_conc_evap(st.session_state.melt_system['conc_pct'])
rho_s15 = rho_from_conc_evap(C_s15)

mass_in_s15 = m_s15_kgph * DT / 3600.0
mass_in_s17 = m_s17_kgph * DT / 3600.0

mass_out_s18 = (flow_s18_m3ph * rho_mix_tank) * DT / 3600.0
mass_out_s19 = (flow_s19_m3ph * rho_mix_tank) * DT / 3600.0
mass_out_s20 = (flow_s20_m3ph * rho_mix_tank) * DT / 3600.0

current_mass_tank = MELT_TANK_VOL_M3 * (current_level/100.0) * rho_mix_tank

# Component Balance (Urea)
urea_mass_tank = current_mass_tank * (st.session_state.melt_system['conc_pct'] / 100.0)
urea_in_s15 = mass_in_s15 * (C_s15 / 100.0)
urea_in_s17 = 0 # Water
urea_out_total = (mass_out_s18 + mass_out_s19 + mass_out_s20) * (st.session_state.melt_system['conc_pct'] / 100.0)

new_urea_mass = urea_mass_tank + urea_in_s15 + urea_in_s17 - urea_out_total

# Component Balance (Biuret)
biuret_mass_tank = current_mass_tank * (st.session_state.melt_system['biuret_pct'] / 100.0)
biuret_in_s15 = mass_in_s15 * (B_s15 / 100.0)
biuret_out_total = (mass_out_s18 + mass_out_s19 + mass_out_s20) * (st.session_state.melt_system['biuret_pct'] / 100.0)

new_biuret_mass = biuret_mass_tank + biuret_in_s15 - biuret_out_total

# Total Mass
new_total_mass = current_mass_tank + mass_in_s15 + mass_in_s17 - (mass_out_s18 + mass_out_s19 + mass_out_s20)

# Energy Balance
cp_tank = cp_from_conc_evap(st.session_state.melt_system['conc_pct'])
cp_s15 = cp_from_conc_evap(C_s15)
cp_s17 = 4.18 # Water

Q_tank = current_mass_tank * cp_tank * st.session_state.melt_system['temp_c']
Q_in_s15 = mass_in_s15 * cp_s15 * T_s15
Q_in_s17 = mass_in_s17 * cp_s17 * T_s17
Q_out_total = (mass_out_s18 + mass_out_s19 + mass_out_s20) * cp_tank * st.session_state.melt_system['temp_c']

new_Q_tank = Q_tank + Q_in_s15 + Q_in_s17 - Q_out_total

# Update State Variables
if new_total_mass > 0.1:
    new_conc = (new_urea_mass / new_total_mass) * 100.0
    new_biuret = (new_biuret_mass / new_total_mass) * 100.0
    new_rho = rho_from_conc_evap(new_conc)
    new_vol = new_total_mass / new_rho
    new_level = (new_vol / MELT_TANK_VOL_M3) * 100.0
    
    new_cp = cp_from_conc_evap(new_conc)
    new_temp = new_Q_tank / (new_total_mass * new_cp)
    
    st.session_state.melt_system['conc_pct'] = new_conc
    st.session_state.melt_system['biuret_pct'] = new_biuret
    st.session_state.melt_system['level_pct'] = np.clip(new_level, 0.0, 100.0)
    st.session_state.melt_system['temp_c'] = new_temp
else:
    st.session_state.melt_system['level_pct'] = 0.0

# Alarm Checks
if st.session_state.melt_system['level_pct'] > 95.0: st.session_state.active_alarms.append(f"🔴 MELT TANK HIGH LEVEL: {st.session_state.melt_system['level_pct']:.1f}%")
if est_header_pressure > 3.8: st.session_state.active_alarms.append(f"🟠 HIGH HEADER PRESSURE: {est_header_pressure:.2f} barg (Override Active)")


# =============================================================================
# UPDATE RECYCLE STREAM STATE (For Next Frame Mixer Calculation)
# =============================================================================
# Stream 20 is the Recycle Output. We save its properties for the next iteration.
rho_s20_actual = rho_from_conc_evap(st.session_state.melt_system['conc_pct'])
mass_flow_s20_actual = flow_s20_m3ph * rho_s20_actual

st.session_state.stream20_props = {
    'mass_kgph': mass_flow_s20_actual,
    'temp_c': st.session_state.melt_system['temp_c'], # Same as tank temp
    'conc_pct': st.session_state.melt_system['conc_pct'], # Same as tank conc
    'biuret_pct': st.session_state.melt_system['biuret_pct']
}

# =============================================================================
# ===== UI DISPLAY (FLICKER-FREE METHOD) ======================================
# =============================================================================
if mode == "Cascade": mode_display = "E1"
else: mode_display = mode[0]
tank_valve_mode_display = valve_mode[0]
if mode_evap1 == "Cascade": mode_display_evap1 = "E1"
else: mode_display_evap1 = mode_evap1[0]
if mode_evap2 == "Cascade": mode_display_evap2 = "E1"
else: mode_display_evap2 = mode_evap2[0]

s11 = st.session_state.stream11_props_lagged
s_evap1 = st.session_state.evap1_state; s_evap2 = st.session_state.evap2_state
s12 = final_streams_evap1
s15 = final_streams_evap2 # Output of Evap 2

current_values = {
    "General_Plant Load": effective_plant_load,
    "Pre-evaporator_Product Temperature": T_current, "Pre-evaporator_Shell Pressure": Psh_current,
    "Pre-evaporator_Vacuum": Pvac_current, "Pre-evaporator_Product Concentration": C_current,
    "Tank_Level (%)": tank_level_percent, "Tank_Temperature": st.session_state.tank_temp,
    "Tank_Urea Concentration": st.session_state.tank_conc, "Tank_Biuret Concentration": st.session_state.tank_biuret_conc,
    "Tank_Level Bar": tank_level_percent,
    "Valves & Pump_Pump Status": "RUN" if st.session_state.pump_running else "STOP",
    "Valves & Pump_Suction Valve Status": "OPEN" if not close_323P003_suction else "CLOSED",
    "Valves & Pump_Discharge Valve Status": "OPEN" if not close_323P003 else "CLOSED",
    "Valves & Pump_Tank Outlet Valve Opening": st.session_state.fv_324401_actual,
    "Valves & Pump_Vacuum Valve Opening": pct_vac_valve,
    "Valves & Pump_Steam Valve Opening": pct_new, "Valves & Pump_Steam Valve Mode": mode_display, "Valves & Pump_Tank Outlet Valve Mode": tank_valve_mode_display,
    "Stream 1 (Plant Feed)_Mass Flow (kg/h)": total_mass_1, "Stream 1 (Plant Feed)_Volume Flow (m³/h)": stream1_vol_flow, "Stream 1 (Plant Feed)_Urea Conc": STREAM1_CONC, "Stream 1 (Plant Feed)_Temperature": 110.0, "Stream 1 (Plant Feed)_Biuret Conc": STREAM1_BIURET,
    "Stream 2 (Dissolving)_Volume Flow (m³/h)": stream2_vol_flow, "Stream 2 (Dissolving)_Urea Conc": STREAM2_CONC, "Stream 2 (Dissolving)_Temperature": 40.0, "Stream 2 (Dissolving)_Biuret Conc": STREAM2_BIURET,
    "Stream 3 (Evap Feed)_Volume Flow (m³/h)": evap_feed_mass_kgph / rho_from_conc_pre_evap(evap_feed_conc_pct) if rho_from_conc_pre_evap(evap_feed_conc_pct) > 0 else 0, "Stream 3 (Evap Feed)_Temperature": stream3_temp, "Stream 3 (Evap Feed)_Urea Conc": stream3_conc,
    "Stream 10 (Pump Out)_Volume Flow (m³/h)": stream10_vol_flow, "Stream 10 (Pump Out)_Urea Conc": stream10_conc,

    # *** Evap 1 Values ***
    "1st Evaporator KPIs_Product Temp (S12)": s_evap1['Tprod_c'], "1st Evaporator KPIs_Product Conc (S12)": s_evap1['Cprod_pct'], "1st Evaporator KPIs_Product Biuret (S12)": s_evap1['Bprod_pct'],
    "1st Evaporator KPIs_Product Mass (S12)": s12['mass_out'], "1st Evaporator KPIs_Vapor Flow (S13)": s12['mass_vapor'],
    "1st Evaporator KPIs_Steam Consumption": steam_flow_kgps_evap1 * 3600, "1st Evaporator KPIs_Heat Duty (Q)": Q_actual_transferred_kw_evap1,
    
    "1st Evaporator Valves_Steam Valve Mode": mode_display_evap1, "1st Evaporator Valves_Steam Valve Opening": s_evap1['pct_valve'], 
    "1st Evaporator Valves_Steam ISO Valve": iso_valve_steam_evap1, "1st Evaporator Valves_Vacuum ISO Valve": iso_valve_vac_evap1, "1st Evaporator Valves_Shell Vent": "OPEN" if shell_vent_evap1 else "CLOSED",
    "1st Evaporator Valves_Vacuum Valve Opening": pct_vac_valve_evap1, "1st Evaporator Valves_Shell Pressure": s_evap1['Psh_display_barg'], "1st Evaporator Valves_Flash Vacuum": s_evap1['Pvac_barg'],
    "Tank (From Code 1)_Level": tank_level_percent, "Tank (From Code 1)_Pump Status": "RUN" if st.session_state.pump_running else "STOP", "Tank (From Code 1)_FV-324401 Mode": tank_valve_mode_display, "Tank (From Code 1)_FV-324401 Opening": st.session_state.fv_324401_actual, "Tank (From Code 1)_FV-324401 SP": fic_sp,

    # *** Evap 2 Values ***
    "2nd Evaporator KPIs_Product Temp (S15)": s_evap2['Tprod_c'], "2nd Evaporator KPIs_Product Conc (S15)": s_evap2['Cprod_pct'], "2nd Evaporator KPIs_Product Biuret (S15)": s_evap2['Bprod_pct'],
    "2nd Evaporator KPIs_Product Mass (S15)": s15['mass_out'], "2nd Evaporator KPIs_Vapor Flow (S16)": s15['mass_vapor'],
    "2nd Evaporator KPIs_Steam Consumption": steam_flow_kgps_evap2 * 3600, "2nd Evaporator KPIs_Heat Duty (Q)": Q_actual_transferred_kw_evap2,
    
    "2nd Evaporator Valves_Steam Valve Mode": mode_display_evap2, "2nd Evaporator Valves_Steam Valve Opening": s_evap2['pct_valve'], 
    "2nd Evaporator Valves_Steam ISO Valve": iso_valve_steam_evap2, "2nd Evaporator Valves_Vacuum ISO Valve": iso_valve_vac_evap2, "2nd Evaporator Valves_Shell Vent": "OPEN" if shell_vent_evap2 else "CLOSED",
    "2nd Evaporator Valves_Vacuum Valve Opening": pct_vac_valve_evap2, "2nd Evaporator Valves_Shell Pressure": s_evap2['Psh_display_barg'], "2nd Evaporator Valves_Flash Vacuum": s_evap2['Pvac_barg'],
}
current_values.update({
    "Melt Tank System_Level": st.session_state.melt_system['level_pct'],
    "Melt Tank System_Level Bar": st.session_state.melt_system['level_pct'], # هذا السطر هو المسؤول عن ملء البار
    "Melt Tank System_Temperature": st.session_state.melt_system['temp_c'],
    "Melt Tank System_Concentration": st.session_state.melt_system['conc_pct'],
    "Melt Tank System_Pump Status": "RUN" if st.session_state.melt_system['pump_running'] else "STOP",
    "Melt Tank System_Prod. Flow (S19)": flow_s19_m3ph,
    "Melt Tank System_Recycle Flow (S20)": flow_s20_m3ph,
    "Melt Tank System_Header Pressure": est_header_pressure,
    "Melt Tank System_Valve A (Prod)": st.session_state.melt_system['valve_A_pos'],
    "Melt Tank System_Valve B (Recyc)": st.session_state.melt_system['valve_B_pos'],
    "Melt Tank System_HV-335602": st.session_state.melt_system.get('hv_335602_pos', 0.0),
    "Melt Tank System_Stream 23 (Mixer Out/S5)": stream5_conc_pct, 
})

BASE_DIR = Path(__file__).resolve().parent
IMAGE_PATH_TANK = BASE_DIR / "tank.png"
IMAGE_PATH_EVAP = BASE_DIR / "Evaporation.png" 

if selected_view == "Tank System":
    try:
        with open(IMAGE_PATH_TANK, "rb") as f: image_base64 = f"data:image/png;base64,{base64.b64encode(f.read()).decode()}"
    except FileNotFoundError: image_base64 = ""
    dynamic_css, html_elements = "", ""
    for entity_name, properties in st.session_state.all_display_items.items():
        for prop_name, settings in properties.items():
            if settings.get('visible', False):
                value_key = f"{entity_name}_{prop_name}"; element_id = "disp_" + re.sub(r'[^A-Za-z0-9_]', '', value_key)
                if 'format' in settings:
                    value = current_values.get(value_key, 0.0); formatted_value = settings['format'].format(value)
                    html_elements += f'<div id="{element_id}" class="metric-box">{formatted_value}</div>'
                    dynamic_css += f"#{element_id} {{ top: {settings['top']:.1f}%; left: {settings['left']:.1f}%; width: {settings['width']:.1f}%; height: {settings['height']:.1f}%; color: {settings['color']}; font-size: {settings['size']:.1f}vw; }}"
                else:
                    html_elements += f'<div id="{element_id}" class="metric-box">&nbsp;</div>'; level = current_values.get(value_key, 0.0); background_css = f"linear-gradient(to top, {settings['color']} {level:.1f}%, #333333 {level:.1f}%)"
                    dynamic_css += f"#{element_id} {{ top: {settings['top']:.1f}%; left: {settings['left']:.1f}%; width: {settings['width']:.1f}%; height: {settings['height']:.1f}%; background: {background_css}; }}"
    st.markdown(f"""<style>.image-container {{ position: relative; width: 100%; height: 85vh; background-image: url("{image_base64}"); background-size: contain; background-repeat: no-repeat; background-position: center; }}.metric-box {{ position: absolute; background-color: rgba(0, 0, 0, 0.0); border: none; padding: 5px; display: flex; align-items: center; justify-content: center; text-align: center; font-family: monospace; font-weight: bold; border-radius: 5px; white-space: nowrap; }}#disp_Tank_Level_Bar {{ border: 1.5px solid white; }}{dynamic_css}</style><div class="image-container">{html_elements}</div>""", unsafe_allow_html=True)

elif selected_view == "Alarms":
    st.title("🚨 Active Alarms Panel")
    if not st.session_state.active_alarms: st.success("✅ System stable. No active alarms.")
    else:
        for alarm in st.session_state.active_alarms:
            if "🔴" in alarm: st.error(alarm)
            elif "🟠" in alarm: st.warning(alarm)
            else: st.info(alarm)

elif selected_view == "Performance KPIs":
    st.title("🚀 Key Performance Indicators (KPIs)")
    st.subheader("1st Evaporator Performance")
    colE, colF, colG, colH = st.columns(4)
    colE.metric("1st Evap Heat Duty (kW)", f"{Q_actual_transferred_kw_evap1:.1f}")
    colF.metric("1st Evap U-Value (W/m²K)", f"{U_actual_evap1:.1f}")
    colG.metric("1st Evap Steam (kg/h)", f"{steam_flow_kgps_evap1*3600:.0f}")
    colH.metric("1st Evap Rate (kg/h)", f"{st.session_state.evap1_last_evap_rate:.0f}")
    st.subheader("2nd Evaporator Performance")
    colI, colJ, colK, colL = st.columns(4)
    colI.metric("2nd Evap Heat Duty (kW)", f"{Q_actual_transferred_kw_evap2:.1f}")
    colJ.metric("2nd Evap U-Value (W/m²K)", f"{U_actual_evap2:.1f}")
    colK.metric("2nd Evap Steam (kg/h)", f"{steam_flow_kgps_evap2*3600:.0f}")
    colL.metric("2nd Evap Rate (kg/h)", f"{st.session_state.evap2_last_evap_rate:.0f}")

elif selected_view == "Streams Data":
    st.title("📊 Process Streams Data")
    streams_data_full = {
        "Stream 10 (Pump Out)": {"Mass Flow (kg/h)": stream10_mass_flow_kgph, "Volume Flow (m³/h)": stream10_vol_flow, "Temperature (°C)": stream10_temp, "Urea Conc (%)": stream10_conc},
        "Stream 11 (1st Evap Feed)": {"Mass Flow (kg/h)": s11['mass_flow_kgph'], "Volume Flow (m³/h)": s11['vol_flow_m3ph'], "Temperature (°C)": s11['temp_c'],"Urea Conc (%)": s11['conc_pct']},
        "Stream 12 (1st Evap Prod)": {"Mass Flow (kg/h)": s12['mass_out'], "Temperature (°C)": s12['temp_out'],"Urea Conc (%)": s12['conc_out']},
        "Stream 13 (1st Evap Vapor)": {"Mass Flow (kg/h)": s12['mass_vapor'], "Temperature (°C)": s12['temp_out'],"Urea Conc (%)": 0.0},
        "Stream 14 (2nd Evap Feed)": {"Mass Flow (kg/h)": s12['mass_out'], "Temperature (°C)": s12['temp_out'],"Urea Conc (%)": s12['conc_out']},
        "Stream 15 (2nd Evap Prod)": {"Mass Flow (kg/h)": s15['mass_out'], "Temperature (°C)": s15['temp_out'],"Urea Conc (%)": s15['conc_out']},
        "Stream 16 (2nd Evap Vapor)": {"Mass Flow (kg/h)": s15['mass_vapor'], "Temperature (°C)": s15['temp_out'],"Urea Conc (%)": 0.0},
        "Stream 20 (Recycle)": {"Mass Flow (kg/h)": st.session_state.stream20_props['mass_kgph'], "Temperature (°C)": st.session_state.stream20_props['temp_c'], "Urea Conc (%)": st.session_state.stream20_props['conc_pct']},
        "Stream 21 (Proc Cond)": {"Mass Flow (kg/h)": s21_mass_kgph, "Temperature (°C)": s21_temp, "Urea Conc (%)": 0.0},
        "Stream 23 (Mixer Out/S5)": {"Mass Flow (kg/h)": stream5_mass_kgph, "Temperature (°C)": stream5_temp_c, "Urea Conc (%)": stream5_conc_pct},
    }
    df = pd.DataFrame(streams_data_full).T
    st.dataframe(df.style.format("{:.2f}", na_rep="-"), use_container_width=True)

elif selected_view == "Future View 2":
    st.title("1st & 2nd Evaporator Dashboard")
    try:
        with open(IMAGE_PATH_EVAP, "rb") as f: image_base64 = f"data:image/png;base64,{base64.b64encode(f.read()).decode()}"
    except FileNotFoundError: image_base64 = ""
    dynamic_css, html_elements = "", ""
    for entity_name, properties in st.session_state.evap_view_display_items.items():
        for prop_name, settings in properties.items():
            if settings.get('visible', False):
                value_key = f"{entity_name}_{prop_name}"; element_id = "disp_evap2_" + re.sub(r'[^A-Za-z0-9_]', '', f"{entity_name}_{prop_name}")
                
                # --- المنطق الصحيح لرسم النصوص والبارات معاً ---
                if 'format' in settings:
                    value = current_values.get(value_key, 0.0); formatted_value = settings['format'].format(value)
                    html_elements += f'<div id="{element_id}" class="metric-box">{formatted_value}</div>'
                    dynamic_css += f"#{element_id} {{ top: {settings['top']:.1f}%; left: {settings['left']:.1f}%; width: {settings['width']:.1f}%; height: {settings['height']:.1f}%; color: {settings['color']}; font-size: {settings['size']:.1f}vw; }}"
                else:
                    # رسم البارات (يستخدم التدرج اللوني)
                    html_elements += f'<div id="{element_id}" class="metric-box">&nbsp;</div>'; level = current_values.get(value_key, 0.0); background_css = f"linear-gradient(to top, {settings['color']} {level:.1f}%, #333333 {level:.1f}%)"
                    dynamic_css += f"#{element_id} {{ top: {settings['top']:.1f}%; left: {settings['left']:.1f}%; width: {settings['width']:.1f}%; height: {settings['height']:.1f}%; background: {background_css}; }}"
                    
    st.markdown(f"""<style>.image-container {{ position: relative; width: 100%; height: 85vh; background-image: url("{image_base64}"); background-size: contain; background-repeat: no-repeat; background-position: center; }}.metric-box {{ position: absolute; background-color: rgba(0, 0, 0, 0.0); border: none; padding: 5px; display: flex; align-items: center; justify-content: center; text-align: center; font-family: monospace; font-weight: bold; border-radius: 5px; white-space: nowrap; }}#disp_evap2_MeltTankSystem_LevelBar {{ border: 1.5px solid white; }}{dynamic_css}</style><div class="image-container">{html_elements}</div>""", unsafe_allow_html=True)

time.sleep(0.1) 
st.rerun()