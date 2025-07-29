import math
import numpy as np
from scipy.integrate import quad

"""
Referenced Literature

Duffie and Beckman - Solar Engineering of Thermal Processes
Niklas and Spatz, 2004,
Enquist, West, Brown., 2009
Kempes et al., 2011

"""


def arrh(T_C, E_a):
    T_K = T_C + 273.15
    k_b = 8.314 #J/K*mol
    return np.exp(-E_a/(k_b*T_K))/np.exp(-E_a/(k_b*273.15))

def hT(T_C):
    T_K = T_C + 273.15
    k_b = 8.314 #J/K*mol
    a_s = 668.39 #K*mol
    b_s = 1.07 # J/mol K^2
    dS = a_s - b_s*T_K
    H_d = 200000 # J/mol
    H_v = 71513 # J/mol
    farrh = np.exp(-H_v/(k_b*T_K))/np.exp(-H_v/(k_b*273.15))
    mod = (1+np.exp((273.15*dS - H_d)/(k_b*273.15)))/(1+np.exp((T_K*dS - H_d)/(k_b*T_K)))
    return farrh*mod



def R_abs(x, day, eSLA=20, h=10, lat=0, R_inc=1367, alt = 0, LDMC = 1.0):
    """ Calculates instantaneous canopy absorbed irradiance and PAR for a given set of geographic conditions (latitude and elevation), trait values. Takes two required parameters solar hour angle (x),
    and calendar day (day). Height (h) is in meters, specific leaf area (eSLA) is in m^2*kg, and R_inc (solar radiation constant) is in Watts/m^2"""
    
    # Atmosferic loss corrections (as per Duffie and Beckman, chapter 2.8)
    azz = 0.4237 - 0.00821*(6-alt/1000)**2
    aoo = 0.5055 + 0.00595*(6.5-alt/1000)**2
    kayy = 0.2711 + 0.01858*(2.5-alt/1000)**2
    rz = 0.97
    ro = 0.99
    rk = 1.01
    az = azz*rz
    ao = aoo*ro
    kay = kayy*rk

    # Instantaneous corrected R_inc (same as above)
    dec = np.radians(24.45*np.sin(2*np.pi*((284+day)/365))) # solar declination angle
    psi = np.arccos(np.cos(lat)*np.cos(dec)*np.cos(x) + np.sin(lat)*np.sin(dec)) # Solar zenith angle
    psi = np.where(psi < np.radians(96.07), psi, np.radians(96.07))
    Gon = R_inc*(1 + (1/30)*(np.cos((2*np.pi*day)/365)))
    am = 1/(np.cos(psi) + 0.50572*(96.07995-np.degrees(psi))**(-1.634)) # Air mass with curvature correction
    taub = az + ao*np.exp(-1*kay*am)
    
    In = Gon*taub # Clear sky beam irradiance
    
    ####################################################################################################################################################################################################
    # Scaling exponents (various sources)
    
    bet_6 = 1 # Proportionality factor between tree height and canopy height - (Kempes et al., 2011)
    bet_5 = 0.3524 # Scaling intercept between tree height and root radial extent - (Enquist, West, Brown., 2009)
    eta_5 = 1.14 # Scaling exponent between tree height and canopy radius - (Enquist, West, Brown., 2009)
    K_2 = 136.8 # +-0.04 kg/m**2 -(Niklas and Spatz, 2004)
    k6 = 0.475 # m - (Niklas and Spatz, 2004)
    k5 = 34.64 # m^(1/3) - (Niklas and Spatz, 2004)
    
    D_tree = ((h+k6)/k5)**(3/2) # Tree diameter
    M_L = K_2*D_tree**2 # Photosynthetic mass
    a_L = M_L*eSLA/LDMC # Total leaf area
  
    
    # Canopy geometry 
    r_can = bet_5*h**(eta_5) # Canopy radial extent in m
    h_can = bet_6*h # Canopy height in m
    zeta_dc = 0.06 # Deep canopy refelction coefficient
    zeta_s = 0.30 # Soil reflection coefficient
    alph = 0.5 # leaf absorptivity
    
    # Canopy projections - Kempes et al., 2011
    theta = np.arctan(-h_can/(2*r_can*np.tan(psi)))
    P_can = np.pi*r_can*(r_can*np.cos(theta) + 0.5*h_can*np.sin(theta)*np.tan(-psi)) # Canopy projection
    S_arc = np.arcsin(2*np.sqrt(abs((0.5*h_can)**(2) - r_can**(2)))/h_can) 
    S_can = 2*np.pi*(r_can**(2) + (r_can**(2)*(h_can*0.5)**(2))*S_arc/(np.sqrt(abs((0.5*h_can)**(2) - r_can**(2))))) # Canopy surface area
    K = 2*P_can/S_can
    
    LAI = (a_L/(np.pi*r_can**(2))) # Leaf area index
    f = (zeta_dc-zeta_s)/(zeta_dc*zeta_s - 1) 
    zeta_c = (zeta_dc + f*np.exp(-2*K*LAI))/(1 + zeta_dc*f*np.exp(-2*K*LAI)) # Canopy reflectance
    tau = np.exp(-np.sqrt(alph)*K*LAI) # Canopy transmission ceofficient
    alph_can = 1 - zeta_c - (1-zeta_s)*tau # Total solar radiaition absorption coefficient for canopy

    # Return values
    R_abs = alph_can*P_can*In #Total solar radiation absorbed by canopy (Watts)
    I_c = (1-zeta_c)*(1-np.exp(-K*LAI)) # PAR absorption coefficient for canopy 
    I_abs = I_c*P_can*In #Total PAR absorbed by canopy (Watts)
    
    return R_abs, I_abs, In



def g_compensation(eSLA, T_A = 20, h=10, p_inc=0.940, uw=4, RH=50, lat=0, alt=0, gsday_s=180, gsday_e=270, LDMC = 1.0):
    
    s_length = gsday_e - gsday_s # Growing season length
    f_g = s_length/365 # s_length as a proportion of whole year
    
    T_D = T_A - (100-RH)/5 # Dew point
    
    p_a = 101.3*np.exp(-alt/8200) # Atmospheric pressure accounting for elevation
    
    #constants
    JtoMuMol = 4.6 # Joules to micromol conversion
    fPAR = 0.5 # Proportion of solar radiation which is photosynthetically active
    mu_w = 0.01801528 # molar mass of water - kg/mol 
    rho_w = 998 # Water density kg/m^{3}
    gamma = 1 # WUE
    D_H = 18.46*10**(-6) # Thermal heat diffusivity of air (m^{2}/s)
    D_nu = 24.9*10**(-6) # Thermal heat diffusivity of water in air (m^{2}/s)
    nu_a = 1.52*10**(-5) # kinematic viscosity of air (m^2/s) at 20C
    S_ca = nu_a/D_nu # Dimensionless Schmidt number 
    P_ra = nu_a/D_H # Dimensionless Prandtl number for air
    c_p = 29.10 # Specific heat of air (isobaric) (J/mol K)
    sig = 5.67*10**(-8) # Stefan-Boltzmann Constant (W/m^2K^4)
    # C_c = 0.1 leaf construction costs
    
    # Scaling
    n = 2 # Branching ratio
    bet_hm = 2.95
    eta_hm = 1.29
    bet_3 = 0.423
    
    bet_mr = 0.421 # micro mol C/s*kg^eta_mr scaling coefficient for total plant respiration
    eta_mr = 0.78 # scaling exponent for total plant respiration
    K_2  = 136.8 # +-0.04 kg/m**2 -(Niklas and Spatz, 2004)
    k6 = 0.475 # m - (Niklas and Spatz, 2004)
    k5 = 34.64 # m^(1/3) - (Niklas and Spatz, 2004)
    
    M = eta_hm*h**(bet_hm)

    D_tree= ((h+k6)/k5)**(3/2) # Tree diameter
    M_L = K_2 *D_tree**2 # Photosynthetic mass
    a_L = M_L*eSLA/LDMC # Total leaf area
    
    # 
    delta_s = 220 #stomatal density stomata/mm^2 (averaged over both sides of leaf)
    a_s = 235.1*10**(-6) # stomatal area mm^2
    z_s = 10*10**(-6) # stomatal depth m
    eps_l = 0.95# leaf emissivity ()
    b_1 = 0.611 # KPa - Tetans formula constant
    b_2 = 17.502 # Dimensionless - Tetans formula constant
    b_3 = 240.97 # deg C - Tetans formula constant
    lamb = 40660 # latent heat of water evaporation (J/mol)
    
    rN = 0.44*10**(-3) # Terminal branch radius
    r0 = D_tree/2 # Basal branch radius
    N = 2*np.log(r0/rN)/np.log(n) # Branching number
    nN = n**N # Total number of terminal branches (number of leaves)
    a_l = a_L/nN # Individual leaf area
    
    # Environmental dependencies of leaf conductance
    rho_a = (44.6*p_a*273.15)/(101.3*(T_A+273)) # molar density of air (mol/m^3)
    d = 1.62*np.sqrt(a_l/(np.pi)) # individual leaf characteristic dimension (assumes circular leaf)
    R_ea = uw*d/nu_a # Dimensionless Reynolds number for air
    e_a = b_1*np.exp(b_2*T_A/(b_3+T_A)) # - Tetans formula for saturation vapor pressure
    de_s = (b_1*b_2*b_3/(b_3+T_A)**(2))*np.exp(b_2*T_A/(b_3+T_A))
    D_v = b_1*np.exp(b_2*T_A/(b_3+T_A)) - b_1*np.exp(b_2*T_D/(b_3+T_D)) # Vapor pressure deficit
    
    # Conductances
    g_ua = (0.664*rho_a*D_v*(R_ea**(1/2))*(S_ca**(1/3)))/d # Boundary layer conductance
    g_us = rho_a*D_nu/z_s # Single stoma conductance
    g_ul = g_us*a_s*delta_s # Leaf conductance (per m^2)
    g_ups = 1/((1/g_ul) + (1/g_ua)) # Canopy conductance
    g_Ha = (0.664*rho_a*D_H*(R_ea**(1/2))*(P_ra**(1/3)))/d # Heat conductance of air
    g_r = 4*eps_l*sig*(T_A+273)**(3)/c_p # Radiative conductance
    
    
    # Flux coefficients
    g_1 = eps_l*sig*(T_A+273)**(4) # Leaf emissivity
    g_2 = g_r*c_p # Leaf emissivity
    j_1 = c_p*g_Ha # Sensible heat loss
    f_1ast = lamb*de_s/p_a # Latent heat loss
    f_2ast = lamb*D_v/p_a #latent heat loss
    
    # Areas
    a_f = 2*a_L*delta_s*a_s # Total area of stomatal openings
    a_g = a_L # One sided leaf area
    a_j = 2*a_L # Two sided leaf area

    
    # Integrate canopy absorbed irradiance and PAR over durnal cycle and average it across season length (computationally expensive, ideally would replace with data)
    R_avg = 0
    I_tot = 0
    step = math.floor(s_length/3)
    for i in range(gsday_s, gsday_e, step):
        day = ((i-1) % 365) + 1
        dec = np.radians(24.45*np.sin(np.radians(360*((284+day)/365)))) # solar declination angle
        sol_set = np.arccos(-np.tan(lat)*np.tan(dec)) # sunrise and sunset times
        R = lambda x:R_abs(x, eSLA=eSLA, h=h,  day=day, lat=lat, R_inc=1367, alt=alt)[0]
        I = lambda x:R_abs(x, eSLA=eSLA, h=h,  day=day, lat=lat, R_inc=1367, alt=alt)[1] 
        # We integrate R_abs and I_abs over a diurnal cycle and average over day length to get average R_abs
        R_temp = quad(R, -sol_set, sol_set)
        I_temp = quad(I, -sol_set, sol_set)
        R_avg += (R_temp[0])/(2*np.pi*3) # Average solar irradiance in Watts, averaged over day and night (2*pi) and evenly spaced portions of growing season (3)
        I_tot += (I_temp[0])/(3*2*np.pi) # Averaged absorbed PAR in Watts, averaged over day and night (2*pi) and evenly spaced portions of growing season (3)
        
    # Hydraulic traits
    r_roo = bet_3**(1/4)*h # Radial root extent
    pre_s = p_inc/(3600*24*s_length) #convert incoming precipitation into m/season to m/s
    Q_p = gamma*(np.pi*r_roo**2)*pre_s # Available flow rate
    #E = (f_1/lamb)*((R_avg - a_f*f_2 - a_g*g_2)/(f_1*a_f + g_1*a_g + j_1*a_j)) + f_2/lamb # Latent heat loss
    #Q_e = (mu_w/rho_w)*a_f*E # Maximum flow rate

    L_1 = R_avg - g_2*a_g
    L_2 = g_1*a_g - j_1*a_j
    H_p = Q_p*lamb*rho_w/(a_f*mu_w)

    omega = L_2*H_p/(f_1ast*(L_1-H_p*a_f) + L_2*f_2ast)

    if omega > g_ups:
        omega = g_ups
    elif omega < 0:
        omega = g_ups

    g_crit = g_ul/(g_ul - omega)

    if g_ua < g_crit:
        p = 1
    else:
        p = g_ua*omega/(g_ua*g_ul - g_ul)

    return p



def G_de(eSLA, T_A = 20, h=10, p_inc=0.940, uw=4, RH=50, kappa=0.5, kap=10.586, lat=0, alt=0, gsday_s=180, gsday_e=270, LDMC = 1.0):
    
    s_length = gsday_e - gsday_s # Growing season length
    f_g = s_length/365 # s_length as a proportion of whole year
    
    T_D = T_A - (100-RH)/5 # Dew point
    
    p_a = 101.3*np.exp(-alt/8200) # Atmospheric pressure accounting for elevation
    
    #constants
    JtoMuMol = 4.6 # Joules to micromol conversion
    fPAR = 0.5 # Proportion of solar radiation which is photosynthetically active
    mu_w = 0.01801528 # molar mass of water - kg/mol 
    rho_w = 998 # Water density kg/m^{3}
    gamma = 1 # WUE
    D_H = 18.46*10**(-6) # Thermal heat diffusivity of air (m^{2}/s)
    D_nu = 24.9*10**(-6) # Thermal heat diffusivity of water in air (m^{2}/s)
    nu_a = 1.52*10**(-5) # kinematic viscosity of air (m^2/s) at 20C
    S_ca = nu_a/D_nu # Dimensionless Schmidt number 
    P_ra = nu_a/D_H # Dimensionless Prandtl number for air
    c_p = 29.10 # Specific heat of air (isobaric) (J/mol K)
    sig = 5.67*10**(-8) # Stefan-Boltzmann Constant (W/m^2K^4)
    # C_c = 0.1 leaf construction costs
    
    # Scaling
    n = 2 # Branching ratio
    bet_lm = 0.16 # scaling coefficient for leaf mass from total biomass
    eta_lm = 0.779 #scaling exponent for leaf mass from total biomass
    bet_hm = 2.95
    eta_hm = 1.29
    bet_3 = 0.423
    
    bet_mr = 0.421 # micro mol C/s*kg^eta_mr scaling coefficient for total plant respiration
    eta_mr = 0.78 # scaling exponent for total plant respiration
    K_2  = 136.8 # +-0.04 kg/m**2 -(Niklas and Spatz, 2004)
    k6 = 0.475 # m - (Niklas and Spatz, 2004)
    k5 = 34.64 # m^(1/3) - (Niklas and Spatz, 2004)
    
    M = eta_hm*h**(bet_hm)

    D_tree= ((h+k6)/k5)**(3/2) # Tree diameter
    M_L = K_2 *D_tree**2 # Photosynthetic mass
    a_L = M_L*eSLA/LDMC # Total leaf area
    
    # 
    delta_s = 220 #stomatal density stomata/mm^2 (averaged over both sides of leaf)
    a_s = 235.1*10**(-6) # stomatal area mm^2
    z_s = 10*10**(-6) # stomatal depth m
    eps_l = 0.95# leaf emissivity ()
    b_1 = 0.611 # KPa - Tetans formula constant
    b_2 = 17.502 # Dimensionless - Tetans formula constant
    b_3 = 240.97 # deg C - Tetans formula constant
    lamb = 40660 # latent heat of water evaporation (J/mol)
    
    rN = 0.44*10**(-3) # Terminal branch radius
    r0 = D_tree/2 # Basal branch radius
    N = 2*np.log(r0/rN)/np.log(n) # Branching number
    nN = n**N # Total number of terminal branches (number of leaves)
    a_l = a_L/nN # Individual leaf area
    
    # Environmental dependencies of leaf conductance
    rho_a = (44.6*p_a*273.15)/(101.3*(T_A+273)) # molar density of air (mol/m^3)
    d = 1.62*np.sqrt(a_l/(np.pi)) # individual leaf characteristic dimension (assumes circular leaf)
    R_ea = uw*d/nu_a # Dimensionless Reynolds number for air
    e_a = b_1*np.exp(b_2*T_A/(b_3+T_A)) # - Tetans formula for saturation vapor pressure
    de_s = (b_1*b_2*b_3/(b_3+T_A)**(2))*np.exp(b_2*T_A/(b_3+T_A))
    D_v = b_1*np.exp(b_2*T_A/(b_3+T_A)) - b_1*np.exp(b_2*T_D/(b_3+T_D)) # Vapor pressure deficit
    
    # Conductances
    g_ua = (0.664*rho_a*D_v*(R_ea**(1/2))*(S_ca**(1/3)))/d # Boundary layer conductance
    g_us = rho_a*D_nu/z_s # Single stoma conductance
    g_ul = g_us*a_s*delta_s # Leaf conductance (per m^2)
    g_ups = 1/((1/g_ul) + (1/g_ua)) # Canopy conductance
    g_Ha = (0.664*rho_a*D_H*(R_ea**(1/2))*(P_ra**(1/3)))/d # Heat conductance of air
    g_r = 4*eps_l*sig*(T_A+273)**(3)/c_p # Radiative conductance
    
    
    # Flux coefficients
    g_1 = eps_l*sig*(T_A+273)**(4) # Leaf emissivity
    g_2 = g_r*c_p # Leaf emissivity
    j_1 = c_p*g_Ha # Sensible heat loss
    f_1 = lamb*g_ups*de_s/p_a # Latent heat loss
    f_2 = lamb*g_ups*D_v/p_a #latent heat loss
    
    # Areas
    a_f = 2*a_L*delta_s*a_s # Total area of stomatal openings
    a_g = a_L # One sided leaf area
    a_j = 2*a_L # Two sided leaf area

    
    # Integrate canopy absorbed irradiance and PAR over durnal cycle and average it across season length (computationally expensive, ideally would replace with data)
    R_avg = 0
    I_tot = 0
    step = math.floor(s_length/3)
    for i in range(gsday_s, gsday_e, step):
        day = ((i-1) % 365) + 1
        dec = np.radians(24.45*np.sin(np.radians(360*((284+day)/365)))) # solar declination angle
        sol_set = np.arccos(-np.tan(lat)*np.tan(dec)) # sunrise and sunset times
        R = lambda x:R_abs(x, eSLA=eSLA, h=h,  day=day, lat=lat, R_inc=1367, alt=alt)[0]
        I = lambda x:R_abs(x, eSLA=eSLA, h=h,  day=day, lat=lat, R_inc=1367, alt=alt)[1] 
        # We integrate R_abs and I_abs over a diurnal cycle and average over day length to get average R_abs
        R_temp = quad(R, -sol_set, sol_set)
        I_temp = quad(I, -sol_set, sol_set)
        R_avg += (R_temp[0])/(2*np.pi*3) # Average solar irradiance in Watts, averaged over day and night (2*pi) and evenly spaced portions of growing season (3)
        I_tot += (I_temp[0])/(3*2*np.pi) # Averaged absorbed PAR in Watts, averaged over day and night (2*pi) and evenly spaced portions of growing season (3)
        
    # Hydraulic traits
    r_roo = bet_3**(1/4)*h # Radial root extent
    pre_s = p_inc/(3600*24*s_length) #convert incoming precipitation into m/season to m/s
    Q_p = gamma*(np.pi*r_roo**2)*pre_s # Available flow rate
    E = (f_1/lamb)*((R_avg - a_f*f_2 - a_g*g_2)/(f_1*a_f + g_1*a_g + j_1*a_j)) + f_2/lamb # Latent heat loss
    Q_e = (mu_w/rho_w)*a_f*E # Maximum flow rate
    
    # Leaf photosynthesis
    u = 768 # Dimensionless +- 71
    k1 = 0.030 # Scaling factor (Kg of biomass/molC)
    k2 = 0.0864 # Scaling factor (s*molC/day*micromolC) = (molC/day)/(micromolC/s)
    Kc_25 = 0.03997 # kPa
    E_aKc = 79430 # J/mol
    Ko_25 = 27.480 # kPa
    E_aKo = 36380 # J/mol
    po2 = 0.2095*p_a
    c_a = 0.0004*p_a
    T_K = T_A + 273.15
    k_b = 8.314 #J/K*mol
    a_a = 0.8 #leaf absorptance
    b_l = 0.010875# fraction of a_a that reaches photosystem II
    T_crt = 647.096 # Critical temperature of water at sea level K
    H0 = 1.677
    H1 = 2.204
    H2 = 0.636
    H3 = -0.241


    Kc = Kc_25*arrh(T_C=T_A, E_a=E_aKc)
    Ko = Ko_25*arrh(T_C=T_A, E_a=E_aKo)
    K_s = Kc*(1+(po2/Ko)) # Effective Michaelis-Menten Coefficient
    beta_carb = 146 # ratio of unit costs for the maintenance of carboxylation and water transport capacities at 25C

    eta_s = 100*np.sqrt(T_K/T_crt)/(H0 + H1*(T_crt/T_K) + H2*(T_crt/T_K)**2 + H3*(T_crt/T_K)**3) # viscosity of water depends on temp
    phi_0 = 0.08718*(0.352 + 0.022*T_A - 0.00034*T_A**2)# intrinsic quantum efficiency of photosynthesis (dep T)
    I_abs = I_tot*fPAR*JtoMuMol # Leaf absorbed PPFD micromols per day
    gam_s = po2*np.exp(6.779 - 37830/(T_K*k_b)) # Photorespiratory compensation point (depT)

    zet = np.sqrt(beta_carb*(gam_s+K_s)/(1.6*eta_s))
    c_i = (zet*c_a +gam_s*np.sqrt(D_v))/(zet + np.sqrt(D_v)) # Leaf internal partial pressure of CO2
    m = (c_i - gam_s)/(c_i + 2*gam_s)
    m_c = (c_i - gam_s)/(c_i + K_s)
    Vcmax_gt = phi_0*I_abs*m/(m_c*a_L)
    Vcmax_25 = Vcmax_gt/(hT(T_C=T_A))
    A_0 = k1*k2*Vcmax_gt*m_c
    
    B_0 = bet_mr*M**(eta_mr)*k1*k2
    gamma0 = A_0*f_g
    gamma1 = (A_0*f_g**(2))*365*k1*k2*Vcmax_25/(2*u)

    G_0 = (gamma0 - gamma1*eSLA)*a_L
    
    Q_r = Q_p/Q_e
    
    G_de = (1/(1+np.exp(-kappa*(Q_r - kap))))*G_0 - B_0
    
    return D_tree, Q_r, Q_e, G_de/M



def G_de_alt(eSLA, T_A = 20, h=10, p_inc=0.940, uw=4, RH=50, lat=0, alt=0, gsday_s=180, gsday_e=270, LDMC = 1.0, gamma=1/3):
    
    s_length = gsday_e - gsday_s # Growing season length
    f_g = s_length/365 # s_length as a proportion of whole year
    
    T_D = T_A - (100-RH)/5 # Dew point
    
    p_a = 101.3*np.exp(-alt/8200) # Atmospheric pressure accounting for elevation
    
    #constants
    JtoMuMol = 4.6 # Joules to micromol conversion
    fPAR = 0.5 # Proportion of solar radiation which is photosynthetically active
    mu_w = 0.01801528 # molar mass of water - kg/mol 
    rho_w = 998 # Water density kg/m^{3}
    D_H = 18.46*10**(-6) # Thermal heat diffusivity of air (m^{2}/s)
    D_nu = 24.9*10**(-6) # Thermal heat diffusivity of water in air (m^{2}/s)
    nu_a = 1.52*10**(-5) # kinematic viscosity of air (m^2/s) at 20C
    S_ca = nu_a/D_nu # Dimensionless Schmidt number 
    P_ra = nu_a/D_H # Dimensionless Prandtl number for air
    c_p = 29.10 # Specific heat of air (isobaric) (J/mol K)
    sig = 5.67*10**(-8) # Stefan-Boltzmann Constant (W/m^2K^4)
    # C_c = 0.1 leaf construction costs
    
    # Scaling
    n = 2 # Branching ratio
    bet_lm = 0.16 # scaling coefficient for leaf mass from total biomass
    eta_lm = 0.779 #scaling exponent for leaf mass from total biomass
    bet_hm = 2.95
    eta_hm = 1.29
    bet_3 = 0.423
    
    bet_mr = 0.421 # micro mol C/s*kg^eta_mr scaling coefficient for total plant respiration
    eta_mr = 0.78 # scaling exponent for total plant respiration
    K_2  = 136.8 # +-0.04 kg/m**2 -(Niklas and Spatz, 2004)
    k6 = 0.475 # m - (Niklas and Spatz, 2004)
    k5 = 34.64 # m^(1/3) - (Niklas and Spatz, 2004)
    
    M = eta_hm*h**(bet_hm)

    D_tree= ((h+k6)/k5)**(3/2) # Tree diameter
    M_L = K_2 *D_tree**2 # Photosynthetic mass
    a_L = M_L*eSLA/LDMC # Total leaf area
    
    # 
    delta_s = 220 #stomatal density stomata/mm^2 (averaged over both sides of leaf)
    a_s = 235.1*10**(-6) # stomatal area mm^2
    z_s = 10*10**(-6) # stomatal depth m
    eps_l = 0.95# leaf emissivity ()
    b_1 = 0.611 # KPa - Tetans formula constant
    b_2 = 17.502 # Dimensionless - Tetans formula constant
    b_3 = 240.97 # deg C - Tetans formula constant
    lamb = 40660 # latent heat of water evaporation (J/mol)
    
    rN = 0.44*10**(-3) # Terminal branch radius
    r0 = D_tree/2 # Basal branch radius
    N = 2*np.log(r0/rN)/np.log(n) # Branching number
    nN = n**N # Total number of terminal branches (number of leaves)
    a_l = a_L/nN # Individual leaf area
    
    # Environmental dependencies of leaf conductance
    rho_a = (44.6*p_a*273.15)/(101.3*(T_A+273)) # molar density of air (mol/m^3)
    d = 1.62*np.sqrt(a_l/(np.pi)) # individual leaf characteristic dimension (assumes circular leaf)
    R_ea = uw*d/nu_a # Dimensionless Reynolds number for air
    e_a = b_1*np.exp(b_2*T_A/(b_3+T_A)) # - Tetans formula for saturation vapor pressure
    de_s = (b_1*b_2*b_3/(b_3+T_A)**(2))*np.exp(b_2*T_A/(b_3+T_A))
    D_v = b_1*np.exp(b_2*T_A/(b_3+T_A)) - b_1*np.exp(b_2*T_D/(b_3+T_D)) # Vapor pressure deficit
    
    # Conductances
    g_ua = (0.664*rho_a*D_v*(R_ea**(1/2))*(S_ca**(1/3)))/d # Boundary layer conductance
    g_us = rho_a*D_nu/z_s # Single stoma conductance
    g_ul = g_us*a_s*delta_s # Leaf conductance (per m^2)
    g_ups = 1/((1/g_ul) + (1/g_ua)) # Canopy conductance
    g_Ha = (0.664*rho_a*D_H*(R_ea**(1/2))*(P_ra**(1/3)))/d # Heat conductance of air
    g_r = 4*eps_l*sig*(T_A+273)**(3)/c_p # Radiative conductance
    
    
    # Flux coefficients
    g_1 = eps_l*sig*(T_A+273)**(4) # Leaf emissivity
    g_2 = g_r*c_p # Leaf emissivity
    j_1 = c_p*g_Ha # Sensible heat loss
    f_1 = lamb*g_ups*de_s/p_a # Latent heat loss
    f_2 = lamb*g_ups*D_v/p_a #latent heat loss
    
    # Areas
    a_f = 2*a_L*delta_s*a_s # Total area of stomatal openings
    a_g = a_L # One sided leaf area
    a_j = 2*a_L # Two sided leaf area

    
    # Integrate canopy absorbed irradiance and PAR over durnal cycle and average it across season length (computationally expensive, ideally would replace with data)
    R_avg = 0
    I_tot = 0
    step = math.floor(s_length/3)
    for i in range(gsday_s, gsday_e, step):
        day = ((i-1) % 365) + 1
        dec = np.radians(24.45*np.sin(np.radians(360*((284+day)/365)))) # solar declination angle
        sol_set = np.arccos(-np.tan(lat)*np.tan(dec)) # sunrise and sunset times
        R = lambda x:R_abs(x, eSLA=eSLA, h=h,  day=day, lat=lat, R_inc=1367, alt=alt)[0]
        I = lambda x:R_abs(x, eSLA=eSLA, h=h,  day=day, lat=lat, R_inc=1367, alt=alt)[1] 
        # We integrate R_abs and I_abs over a diurnal cycle and average over day length to get average R_abs
        R_temp = quad(R, -sol_set, sol_set)
        I_temp = quad(I, -sol_set, sol_set)
        R_avg += (R_temp[0])/(2*np.pi*3) # Average solar irradiance in Watts, averaged over day and night (2*pi) and evenly spaced portions of growing season (3)
        I_tot += (I_temp[0])/(3*2*np.pi) # Averaged absorbed PAR in Watts, averaged over day and night (2*pi) and evenly spaced portions of growing season (3)
        
    # Hydraulic traits
    r_roo = bet_3**(1/4)*h # Radial root extent
    pre_s = p_inc/(3600*24*s_length) #convert incoming precipitation into m/season to m/s
    Q_p = gamma*(np.pi*r_roo**2)*pre_s # Available flow rate
    E = (f_1/lamb)*((R_avg - a_f*f_2 - a_g*g_2)/(f_1*a_f + g_1*a_g + j_1*a_j)) + f_2/lamb # Latent heat loss
    Q_e = (mu_w/rho_w)*a_f*E # Maximum flow rate
    LpD_to_M3pS = (10**(-6))/(86400) # Liters per day to cubic meters per second conversion factor
    bet_2 = 9.2*LpD_to_M3pS # basal flow rate intercept
    Q_0 = bet_2*h**(2.7)
    
    # Leaf photosynthesis
    u = 768 # Dimensionless +- 71
    k1 = 0.030 # Scaling factor (Kg of biomass/molC)
    k2 = 0.0864 # Scaling factor (s*molC/day*micromolC) = (molC/day)/(micromolC/s)
    Kc_25 = 0.03997 # kPa
    E_aKc = 79430 # J/mol
    Ko_25 = 27.480 # kPa
    E_aKo = 36380 # J/mol
    po2 = 0.2095*p_a
    c_a = 0.0004*p_a
    T_K = T_A + 273.15
    k_b = 8.314 #J/K*mol
    a_a = 0.8 #leaf absorptance
    b_l = 0.010875# fraction of a_a that reaches photosystem II
    T_crt = 647.096 # Critical temperature of water at sea level K
    H0 = 1.677
    H1 = 2.204
    H2 = 0.636
    H3 = -0.241


    Kc = Kc_25*arrh(T_C=T_A, E_a=E_aKc)
    Ko = Ko_25*arrh(T_C=T_A, E_a=E_aKo)
    K_s = Kc*(1+(po2/Ko)) # Effective Michaelis-Menten Coefficient
    beta_carb = 146 # ratio of unit costs for the maintenance of carboxylation and water transport capacities at 25C

    eta_s = 100*np.sqrt(T_K/T_crt)/(H0 + H1*(T_crt/T_K) + H2*(T_crt/T_K)**2 + H3*(T_crt/T_K)**3) # viscosity of water depends on temp
    phi_0 = 0.08718*(0.352 + 0.022*T_A - 0.00034*T_A**2)# intrinsic quantum efficiency of photosynthesis (dep T)
    I_abs = I_tot*fPAR*JtoMuMol # Leaf absorbed PPFD micromols per day
    gam_s = po2*np.exp(6.779 - 37830/(T_K*k_b)) # Photorespiratory compensation point (depT)

    zet = np.sqrt(beta_carb*(gam_s+K_s)/(1.6*eta_s))
    c_i = (zet*c_a +gam_s*np.sqrt(D_v))/(zet + np.sqrt(D_v)) # Leaf internal partial pressure of CO2
    m = (c_i - gam_s)/(c_i + 2*gam_s)
    m_c = (c_i - gam_s)/(c_i + K_s)
    Vcmax_gt = phi_0*I_abs*m/(m_c*a_L)
    Vcmax_25 = Vcmax_gt/(hT(T_C=T_A))
    A_0 = k1*k2*Vcmax_gt*m_c
    
    B_0 = bet_mr*M**(eta_mr)*k1*k2
    gamma0 = A_0*f_g
    gamma1 = (A_0*f_g**(2))*365*k1*k2*Vcmax_25/(2*u)

    G_0 = (gamma0 - gamma1*eSLA)*a_L


    G_corr = g_compensation(eSLA, T_A = T_A, h=h, p_inc=p_inc, uw=uw, RH=RH,
                    lat=lat, alt=alt, gsday_s=gsday_s, gsday_e=gsday_e, LDMC = LDMC)

    G_de = (G_corr*G_0 - B_0)/m
    
    
    return G_de, G_corr