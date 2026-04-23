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

####################################################################################################################################################################################################
# Scaling exponents (various sources)
    
bet_mr = 0.421 # micro mol C/s*kg^eta_mr scaling coefficient for total plant respiration
eta_mr = 0.78 # scaling exponent for total plant respiration
bet_6 = 1 # Proportionality factor between tree height and canopy height - (Kempes et al., 2011)
bet_3 = 0.423**(1/4) #+-0.01 Scaling intercept for root radial extent in terms of tree height (dimensionless), (Niklas 2004) 
bet_5 = 0.3524 # Scaling intercept between tree height and canopy radius - (Enquist, West, Brown., 2009)
eta_5 = 1.14 # Scaling exponent between tree height and canopy radius - (Enquist, West, Brown., 2009)
# K_2 = 136.8 # +-0.04 kg/m**2 -(Niklas and Spatz, 2004)
C_L = 12000 # unitless, West, Brown, Enquist; Nature; 1999
# k6 = 0.475 # m - (Niklas and Spatz, 2004)
# k5 = 34.64 # m^(1/3) - (Niklas and Spatz, 2004)
n = 2 # Branching ratio
bet_hm = 2.95
eta_hm = 1.29
K_0 = 2.05 #+- 0.02 1/yr - (Niklas and Spatz, 2004)
K_1 = 0.281 #+- 0.02 kg^(1/4)/yr
K_3 = 0.423 # unitless - (Niklas and Spatz, 2004)
K_4 = 202.3 #+- 0.01 kg/m^3 - (Niklas and Spatz, 2004)

# Photosynthesis constants
k1 = 0.030 # Scaling factor (Kg of biomass/molC)
k2 = 0.0864 # Scaling factor (s*molC/day*micromolC) = (molC/day)/(micromolC/s)
u = 768 # Dimensionless +- 71
C_c = 13.23 # 13.23 +- 4.07 leaf construction costs prentice et al
muMol_to_mol = 1/(10**6)

# Heat flux constants
b_1 = 0.611 # KPa - Tetans formula constant
b_2 = 17.502 # Dimensionless - Tetans formula constant
b_3 = 240.97 # deg C - Tetans formula constant
lamb = 40660 # latent heat of water evaporation (J/mol)

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


#########################################################################################################################################################################################################################
# Radiative transfer equations
#########################################################################################################################################################################################################################

def Elev_angle(lat, day, o):
    dec = np.radians(24.45*np.sin(2*np.pi*((284+day)/365)))
    inv = np.sin(lat)*np.sin(dec) + np.cos(lat)*np.cos(dec)*np.cos(o)
    Elev = np.arcsin(inv)
    return Elev



def P_can_dif(r, h):
    """Calculates effective horizontal canopy projection for diffuse radiation from the ratio of canopy radius (r) and height (h). Based on a hyperbolic approximation
    for average canopy projection integrated from (0.2-pi/2)rads."""

    b = 2*r/h
    res = (np.pi*r**2)*((0.83/b) + 0.79) ## Hyperbolic fit between effective canopy projection and aspect ratio (b)

    return res

def P_can_dir(r, h, elev_eff):
    """Calculates effective canopy projection given canopy radius (r), height (h), and effective elevation angle"""

    b = 2*r/h

    res = 2*np.pi*r**2*np.sqrt(1 + (1/(np.tan(elev_eff)*b)**2))

    return res



def S_can(r, h):
    """Calculates canopy surface area as an ellipsoid with horizontal radius (r) and vertical semimajor axis (h/2)"""

    b = 2*r/h

    if b < 1:
        e_2 = np.sqrt(1-b**2)
        res = 2*np.pi*r**2*(1 + np.arcsin(e_2)/(b*e_2))
    elif b == 1:
        res = 4*np.pi*r**2
    else:
        e_1 = np.sqrt(1 - (1/b)**2)
        res = 2*np.pi*r**2*(1 + np.log((1+e_1)/(1-e_1))/(2*e_1*b) )
        
    return res



def I_abs_dif(R_dif, LAI, P_can_dif, S_can, rho_c, zeta_s, alph):
    """Calculates total absorbed diffuse radiation flux from incoming diffuse radiation (R_dif), leaf area index (LAI), effective canopy projection for diffuse light (P_can_dif), deep canopy reflection coefficient (rho_c),
    soil reflectance (zeta_s), and leaf absorptivity coefficient for longwave and shortwave radiation (alph)"""

    f = (rho_c-zeta_s)/(rho_c*zeta_s - 1) 
    zeta_c = (rho_c + f*np.exp(-2*(P_can_dif/S_can)*LAI))/(1 + rho_c*f*np.exp(-2*(P_can_dif/S_can)*LAI))

    res = R_dif*P_can_dif*(1-zeta_c - (1-zeta_s)*np.exp(-np.sqrt(alph)*(P_can_dif/S_can)*LAI))

    return res



def I_abs_dir(R_dir, LAI, P_can_dir, S_can, rho_c, zeta_s, alph):
    """Calculates total absorbed direct radiation flux from incoming diffuse radiation (R_dif), leaf area index (LAI), effective canopy projection for diffuse light (P_can_dif), deep canopy reflection coefficient (rho_c),
    soil reflectance (zeta_s), and leaf absorptivity coefficient for longwave and shortwave radiation (alph)"""


    f = (rho_c-zeta_s)/(rho_c*zeta_s - 1) 
    zeta_c = (rho_c + f*np.exp(-2*(P_can_dir/S_can)*LAI))/(1 + rho_c*f*np.exp(-2*(P_can_dir/S_can)*LAI))

    res = R_dir*P_can_dir*(1-zeta_c - (1-zeta_s)*np.exp(-np.sqrt(alph)*(P_can_dir/S_can)*LAI))

    return res



def I_PAR_dif(R_PAR_dif, LAI, P_can_dif, S_can, rho_c, alph_p):
    """Calculates intercepted diffuse PAR from incoming diffuse PAR (R_dif), leaf area index (LAI), effective canopy projection for diffuse light (P_can_dif), deep canopy reflection coefficient (rho_c), and leaf absorptivity coefficient for PAR (alph_p)"""


    res = R_PAR_dif*P_can_dif*(1-rho_c)*(1-np.exp(-np.sqrt(alph_p)*(P_can_dif/S_can)*LAI))

    return res


def I_PAR_dir(R_PAR_dir, LAI, P_can_dir, S_can, rho_c, alph_p):
    """Calculates intercepted direct PAR from incoming direct PAR (R_PAR_dir), leaf area index (LAI), effective canopy projection for diffuse light (P_can_dir), deep canopy reflection coefficient (rho_c), and leaf absorptivity coefficient for PAR (alph_p)"""


    res = R_PAR_dir*P_can_dir*(1-rho_c)*(1 - np.exp(-np.sqrt(alph_p)*(P_can_dir/S_can)*LAI))

    return res

def Rad_transfer(LMA, R_dir, R_dif, R_PAR_dif, R_PAR_dir, h=10, lat=0, gsday_s=180, gsday_e=270, bins=5):

    s_length = gsday_e - gsday_s # Growing season length
    lat =np.radians(lat)

    # Canopy geometry 
    r_can = bet_5*h**(eta_5) # Canopy radial extent in m
    h_can = bet_6*h # Canopy height in m
    rho_c = 0.06 # Deep canopy refelction coefficient
    zeta_s = 0.30 # Soil reflection coefficient
    alph = 0.5 # leaf absorptivity
    alph_p = 0.8 # leaf absoptivity in the PAR waveband

    K_2 = (LMA*C_L)/4
    K_6 = K_2/((1+K_3)*K_4)
    K_5 = (((K_0*K_2)/K_1)**(4/3))*(K_6/K_2)

    D_tree= ((h+K_6)/K_5)**(3/2) # Tree diameter
    M_L = K_2*D_tree**2 # Photosynthetic mass
    a_L = M_L/LMA # Total leaf area
    LAI = a_L/(np.pi*r_can**2)
    S_canopy = S_can(r=r_can, h=h_can)

    I_abs = 0
    I_PAR = 0
    sol_set_avg = 0
    step = math.floor(s_length/bins)
    for day in range(gsday_s, gsday_e, step):

        dec = np.radians(24.45*np.sin(np.radians(360*((284+day)/365)))) # solar declination angle
        sol_set = np.arccos(-np.tan(lat)*np.tan(dec)) # sunrise and sunset times
        o_eff = 0.5*sol_set
        theta_eff = Elev_angle(lat=lat, day=day, o=o_eff)

        P_can_df = P_can_dif(r=r_can, h=h_can)
        P_can_dr = P_can_dir(r=r_can, h=h_can, elev_eff=theta_eff)

        I_abs += (I_abs_dif(R_dif=R_dif, LAI=LAI, P_can_dif=P_can_df, S_can=S_canopy, rho_c=rho_c, zeta_s=zeta_s, alph=alph) + I_abs_dir(R_dir=R_dir, LAI=LAI, P_can_dir=P_can_dr, S_can=S_canopy, rho_c=rho_c, zeta_s=zeta_s, alph=alph))/bins
        I_PAR += (I_PAR_dir(R_PAR_dir=R_PAR_dir, LAI=LAI, P_can_dir=P_can_dr, S_can=S_canopy, rho_c=rho_c, alph_p=alph_p) + I_PAR_dif(R_PAR_dif=R_PAR_dif, LAI=LAI, P_can_dif=P_can_df, S_can=S_canopy, rho_c=rho_c, alph_p=alph_p))/bins
        sol_set_avg += sol_set/bins

    return I_abs, I_PAR, sol_set_avg


#########################################################################################################################################################################################################################
# Photosynthetic assimilation equations
#########################################################################################################################################################################################################################

def g_de(a_L, I_PAR, sol_set_avg, T_A = 20, RH=50, alt=0):
    """Calculates growing season averaged assimilation rate per unit leaf area"""
    
    #constants
    JtoMuMol = 4.6 # Joules to micromol conversion

    T_D = T_A - (100-RH)/5 # Dew point
    p_a = 101.3*np.exp(-alt/8200) # Atmospheric pressure accounting for elevation kPa


    ####################################################################################
    # Leaf photosynthesis
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
    b_l = 0.010875# fraction of a_a that reaches photosystem 
    
    # Water viscosity
    T_crt = 647.096 # Critical temperature of water at sea level K
    H0 = 1.677
    H1 = 2.204
    H2 = 0.636
    H3 = -0.241
    eta_s = 100*np.sqrt(T_K/T_crt)/(H0 + H1*(T_crt/T_K) + H2*(T_crt/T_K)**2 + H3*(T_crt/T_K)**3) # viscosity of water depends on temp


    Kc = Kc_25*arrh(T_C=T_A, E_a=E_aKc)
    Ko = Ko_25*arrh(T_C=T_A, E_a=E_aKo)
    K_s = Kc*(1+(po2/Ko)) # Effective Michaelis-Menten Coefficient
    beta_carb = 146 # ratio of unit costs for the maintenance of carboxylation and water transport capacities at 25C
    D_v = b_1*np.exp(b_2*T_A/(b_3+T_A)) - b_1*np.exp(b_2*T_D/(b_3+T_D)) # Vapor pressure deficit

    
    phi_0 = (a_a*b_l/4)*(0.352 + 0.022*T_A - 0.00034*T_A**2)# intrinsic quantum yield of photosynthesis (dep T)
    I_PAR_day = I_PAR*(24*sol_set_avg)*(3600)/np.pi # Net PAR absorbed in a day (joules)
    PPFD_abs_day = (I_PAR_day*JtoMuMol)/(a_L)# Average absorbed PPFD per unit leaf area integrated over diurnal cycle (micromols/(day*m^2))
    PPFD_abs = (I_PAR*JtoMuMol)/(a_L) ## Average absorbed PPFD per unit leaf area i (micromols/(day*m^2))

    gam_s = po2*np.exp(6.779 - 37830/(T_K*k_b)) # Photorespiratory compensation point (depT)
    zet = np.sqrt(beta_carb*(gam_s+K_s)/(1.6*eta_s))
    c_i = (zet*c_a +gam_s*np.sqrt(D_v))/(zet + np.sqrt(D_v)) # Leaf internal partial pressure of CO2
    m = (c_i - gam_s)/(c_i + 2*gam_s)
    m_c = (c_i - gam_s)/(c_i + K_s)

    Vcmax_gt = phi_0*PPFD_abs*m*(muMol_to_mol)/(m_c)

    A_0 = k1*phi_0*PPFD_abs_day*m*(muMol_to_mol)

    delt_c_a = np.abs(c_i - c_a)

    g_ul = 1.6*p_a*(Vcmax_gt*m_c)*(1 + (zet/np.sqrt(D_v)))/(c_a - gam_s)
    
    
    return A_0, Vcmax_gt, g_ul

##################################################################################################################################################
# Radiative Balance and Evapotranspiration
##################################################################################################################################################

def g_compensation(LMA, a_l, g_ul, I_abs, sol_set_avg, T_A = 20, h=10, p_inc=0.940, uw=4, RH=50, alt=0, gsday_s=180, gsday_e=270):
    
    s_length = gsday_e - gsday_s # Growing season length
    
    T_D = T_A - (100-RH)/5 # Dew point
    
    p_a = 101.3*np.exp(-alt/8200) # Atmospheric pressure accounting for elevation
    
    #constants
    
    mu_w = 0.01801528 # molar mass of water - kg/mol 
    rho_w = 998 # Water density kg/m^{3}
    gamma = 0.33 # Root water absorption efficiency
    D_H = 18.46*10**(-6) # Thermal heat diffusivity of air (m^{2}/s)
    D_nu = 24.9*10**(-6) # Thermal heat diffusivity of water in air (m^{2}/s)
    nu_a = 1.52*10**(-5) # kinematic viscosity of air (m^2/s) at 20C
    S_ca = nu_a/D_nu # Dimensionless Schmidt number 
    P_ra = nu_a/D_H # Dimensionless Prandtl number for air
    c_p = 29.10 # Specific heat of air (isobaric) (J/mol K)
    sig = 5.67*10**(-8) # Stefan-Boltzmann Constant (W/m^2K^4)
    
    # Scaling
    n = 2 # Branching ratio
    bet_3 = 0.423
    K_2 = (LMA*C_L)/4
    K_6 = K_2/((1+K_3)*K_4)
    K_5 = (((K_0*K_2)/K_1)**(4/3))*(K_6/K_2)

    D_tree= ((h+K_6)/K_5)**(3/2) # Tree diameter
    M_L = K_2 *D_tree**2 # Photosynthetic mass
    a_L = M_L/LMA # Total one sided leaf area
    
    
    eps_l = 0.95# leaf emissivity ()
    
    # Environmental dependencies of leaf conductance
    rho_a = (44.6*p_a*273.15)/(101.3*(T_A+273)) # molar density of air (mol/m^3)
    d = 1.62*np.sqrt(a_l/(np.pi)) # individual leaf characteristic dimension (assumes circular leaf)
    R_ea = uw*d/nu_a # Dimensionless Reynolds number for air
    e_a = b_1*np.exp(b_2*T_A/(b_3+T_A)) # - Tetans formula for saturation vapor pressure
    de_s = (b_1*b_2*b_3/(b_3+T_A)**(2))*np.exp(b_2*T_A/(b_3+T_A))
    D_v = e_a - b_1*np.exp(b_2*T_D/(b_3+T_D)) # Vapor pressure deficit
    
    # Conductances
    g_ua = (0.664*rho_a*D_v*(R_ea**(1/2))*(S_ca**(1/3)))/d # Boundary layer conductance
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
    a_g = a_L # One sided leaf area
    a_j = 2*a_L # Two sided leaf area

    
    r_roo = bet_3**(1/4)*h # Radial root extent
    pre_s = p_inc/(3600*24*s_length) #convert incoming precipitation into m/season to m/s
    Q_p = gamma*(np.pi*r_roo**2)*pre_s # Available flow rate
    

    I_abs_day = (sol_set_avg/np.pi)*I_abs
    L_1 = I_abs_day - g_2*a_g
    L_2 = g_1*a_g - j_1*a_j
    H_p = Q_p*lamb*rho_w/(a_j*mu_w)

    omega = L_2*H_p/(f_1ast*(L_1-H_p*a_j) + L_2*f_2ast)

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

##################################################################################################################################################
# Net assimilation rate
##################################################################################################################################################

def G_de(LMA, h, a_l, T_A, p_inc, uw, RH, lat, alt, R_dir, R_dif, R_PAR_dir, R_PAR_dif, gsday_s, gsday_e):
    """Calculates net growing season carbon assimilation"""


    s_length = gsday_e - gsday_s # Growing season length

    K_2 = (LMA*C_L)/4
    K_6 = K_2/((1+K_3)*K_4)
    K_5 = (((K_0*K_2)/K_1)**(4/3))*(K_6/K_2)

    D_tree= ((h+K_6)/K_5)**(3/2) # Tree diameter
    M_L = K_2 *D_tree**2 # Photosynthetic mass
    M_T = (((K_0*K_2)/K_1)**(4/3))*D_tree**(8/3)
    a_L = M_L/LMA # Total one sided leaf area

    RADs = Rad_transfer(LMA=LMA, R_dir=R_dir, R_dif=R_dif, R_PAR_dir=R_PAR_dir, R_PAR_dif=R_PAR_dif, h=10, lat=lat, gsday_s=gsday_s, gsday_e=gsday_e)

    I_abs = RADs[0]
    I_PAR= RADs[1]
    sol_set_avg = RADs[2]

    PHOTs = g_de(a_L=a_L, I_PAR=I_PAR, sol_set_avg=sol_set_avg, T_A=T_A, RH=RH, alt=alt)

    A_0 = PHOTs[0]
    Vcmax_gt = PHOTs[1]
    g_ul = PHOTs[2]
    Vcmax_25 = Vcmax_gt/(hT(T_C=T_A))
    sen_rate = (u*LMA)/(k1*k2*Vcmax_25)

    p_g = g_compensation(LMA=LMA, a_l=a_l, g_ul=g_ul, I_abs=I_abs, sol_set_avg=sol_set_avg, T_A=T_A, h=h, p_inc=p_inc, uw=uw, RH=RH, alt=alt, gsday_s=gsday_s, gsday_e=gsday_e)

    LL_ev = np.sqrt(2*sen_rate*C_c*LMA*365/(s_length*A_0))



    g_area = s_length*A_0*p_g*(1 - s_length/(2*sen_rate)) - C_c*LMA - (bet_mr*M_T**(eta_mr)*k1*k2)/a_L
    #g_area = s_length*A_0*p_g*(1 - LL_ev/(2*sen_rate)) - s_length*C_c*LMA/LL_ev
    G_tot = g_area*a_L

    return g_area, G_tot, g_ul





