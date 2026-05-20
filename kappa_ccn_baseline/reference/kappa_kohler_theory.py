import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy
import miepython
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
from scipy.optimize import root_scalar

def kappa_petter_and_Kreidenweis_2010_EQ10(critical_diameter, critical_saturation, surface_tension = 0.072 #J/m2
                                                                                 , Mw = 18.01528           #g/mol
                                                                                 , T = 298.15              #K
                                                                                 , density = 997048        #g/m3
                                                                                 , R=8.3145):              #J/(mol K)
    A = (4 * surface_tension * Mw) / (R * T * density)
    kappa = (4 * A**3) / (27 * critical_diameter**3 * np.square(np.log(critical_saturation)))

    return kappa

def Sc_petter_and_Kreidenweis_2010_EQ10(critical_diameter, kappa, surface_tension = 0.072 #J/m2
                                              , Mw = 18.01528           #g/mol
                                              , T = 298.15              #K
                                              , density = 997048        #g/m3
                                              , R=8.3145):              #J/(mol K)
    
    A = (4 * surface_tension * Mw) / (R * T * density)
    Sc = np.exp(np.sqrt((4 * A**3)/(27 * critical_diameter**3 * kappa)))
    return Sc

def Dd_petter_and_Kreidenweis_2010_EQ10(kappa, critical_saturation, surface_tension = 0.072 #J/m2
                                              , Mw = 18.01528           #g/mol
                                              , T = 298.15              #K
                                              , density = 997048        #g/m3
                                              , R=8.3145):              #J/(mol K)
    
    A = (4 * surface_tension * Mw) / (R * T * density)
    critical_diameter = np.power((4 * A**3) / (27 * kappa * np.square(np.log(critical_saturation))), 1/3)
    return critical_diameter


def S_petter_and_Kreidenweis_2010_EQ6(wet_diameter, dry_diameter, kappa, surface_tension = 0.072 #J/m2
                                                                       , Mw = 18.01528           #g/mol
                                                                       , T = 298.15              #K
                                                                       , density = 997048        #g/m3
                                                                       , R=8.3145):              #J/(mol K)
    
    B = (4 * surface_tension * Mw) / (R * T * density * wet_diameter)
    S = (wet_diameter**3 - dry_diameter**3) / (wet_diameter**3 - (1-kappa) * dry_diameter**3) * np.exp(B)
    return S

def find_peak_S_D_binary_search(Dd_input, kappa, surface_tension=0.072, Mw=18.01528, T=298.15, density=997048, R=8.3145, tol=1e-12):
    """
    Find the peak of the S(D) function using a binary search approach for a single dry diameter or an array of dry diameters.
    
    Parameters:
    Dd_input (float or np.array): Single dry diameter or array of dry diameters in meters
    kappa (float): Hygroscopicity parameter
    surface_tension (float): Surface tension in J/m^2
    Mw (float): Molar mass of water in g/mol
    T (float): Temperature in Kelvin
    density (float): Density of water in g/m^3
    R (float): Universal gas constant in J/(mol*K)
    tol (float): Tolerance for the convergence of the binary search

    Returns:
    If Dd_input is a single value: Returns two single values (peak diameter and peak S(D)).
    If Dd_input is an array: Returns two arrays (array of peak diameters and array of peak S(D) values).
    """

    # Check if the input is a single value or an array
    if isinstance(Dd_input, (float, np.float64)):
        Dd_array = [Dd_input]  # Convert to a list for consistency
        single_value_input = True
    else:
        Dd_array = Dd_input
        single_value_input = False

    peak_diameters = []
    peak_S_D_values = []

    for Dd in Dd_array:
        left = Dd
        right = Dd * 100

        while right - left > tol:
            mid = (left + right) / 2
            mid_left = (left + mid) / 2
            mid_right = (mid + right) / 2

            if S_petter_and_Kreidenweis_2010_EQ6(mid_left, Dd, kappa, surface_tension, Mw, T, density, R) < S_petter_and_Kreidenweis_2010_EQ6(mid, Dd, kappa, surface_tension, Mw, T, density, R):
                left = mid_left
            elif S_petter_and_Kreidenweis_2010_EQ6(mid_right, Dd, kappa, surface_tension, Mw, T, density, R) > S_petter_and_Kreidenweis_2010_EQ6(mid, Dd, kappa, surface_tension, Mw, T, density, R):
                left = mid
            else:
                right = mid_right

        peak_D = (left + right) / 2
        peak_S_D = S_petter_and_Kreidenweis_2010_EQ6(peak_D, Dd, kappa, surface_tension, Mw, T, density, R)

        peak_diameters.append(peak_D)
        peak_S_D_values.append(peak_S_D)

    if single_value_input:
        return peak_diameters[0], peak_S_D_values[0]
    else:
        return np.array(peak_diameters), np.array(peak_S_D_values)


# def calculate_critical_diameter_interpolated(aerosol_sizes, aerosol_size_dist, super_saturations, ccn_concentrations, tolerence = 1E-12):
#     critical_diameters = []

#     # get rid off nan values
#     non_nan_indices = np.where(~np.isnan(aerosol_size_dist))
#     aerosol_sizes = aerosol_sizes[non_nan_indices]
#     aerosol_size_dist = aerosol_size_dist[non_nan_indices]

#     # Create an interpolation of the aerosol size distribution
#     original_interpolation = scipy.interpolate.interp1d(aerosol_sizes, aerosol_size_dist, kind='linear', bounds_error=False, fill_value="extrapolate")

#     print('calculate critical dry diameter: Debug Information')

#     for ss_i, an_ss in enumerate(super_saturations):
#         a_ccnc = ccn_concentrations[ss_i]

#         lower_bound = min(aerosol_sizes)
#         upper_bound = max(aerosol_sizes)
#         crit_diameter = None

#         while upper_bound - lower_bound > tolerence:  # Tolerance for the search
#             mid_point = 10**((np.log10(lower_bound) + np.log10(upper_bound)) / 2)

#             fine_grid = np.linspace(mid_point, aerosol_sizes[-1], 10000)
#             interpolated_dist = original_interpolation(fine_grid)

#             if len(interpolated_dist) > 0 and len(fine_grid) > 0:

#                 integral = scipy.integrate.simpson(y = interpolated_dist, x = np.log10(fine_grid))
#                 if integral < a_ccnc:
#                     upper_bound = mid_point
#                 else:
#                     lower_bound = mid_point
#             else:
#                 # Can not do simpson integration
#                 upper_bound = lower_bound

#         crit_diameter = 10**((np.log10(lower_bound) + np.log10(upper_bound)) / 2)
#         critical_diameters.append(crit_diameter)

#         # Debugging information
#         print(f"Super Saturation: {an_ss:.4f}, CCN Concentration: {a_ccnc:.4f}, Critical Diameter: {crit_diameter:.4f}, Integral: {integral:.4f}")

#     critical_diameters = np.array(critical_diameters)

#     return critical_diameters

import numpy as np
import scipy.interpolate
import scipy.integrate

def calculate_critical_diameter_interpolated(aerosol_sizes, aerosol_size_dist, super_saturations, ccn_concentrations, tolerance=1E-6):
    critical_diameters = []

    # Remove NaN values from the aerosol size distribution
    non_nan_indices = np.where(~np.isnan(aerosol_size_dist))
    aerosol_sizes = aerosol_sizes[non_nan_indices]
    aerosol_size_dist = aerosol_size_dist[non_nan_indices]

    # Create an interpolation of the aerosol size distribution
    original_interpolation = scipy.interpolate.interp1d(
        aerosol_sizes, aerosol_size_dist, kind='linear', bounds_error=False, fill_value="extrapolate"
    )

    #print('Calculating critical dry diameter: Debug Information')

    for ss_i, an_ss in enumerate(super_saturations):
        a_ccnc = ccn_concentrations[ss_i]

        lower_bound = min(aerosol_sizes)
        upper_bound = max(aerosol_sizes)
        crit_diameter = None

        max_iterations = 1000  # To prevent infinite loops
        iteration_count = 0

        while upper_bound - lower_bound > tolerance and iteration_count < max_iterations:
            iteration_count += 1
            # Compute the midpoint in log scale
            mid_point = 10**((np.log10(lower_bound) + np.log10(upper_bound)) / 2)

            # Create a fine grid for integration
            fine_grid = np.linspace(mid_point, aerosol_sizes[-1], 2000)
            interpolated_dist = original_interpolation(fine_grid)

            # Handle invalid interpolation values
            interpolated_dist = np.nan_to_num(interpolated_dist, nan=0.0, posinf=0.0, neginf=0.0)

            if len(interpolated_dist) > 0 and len(fine_grid) > 0:
                # Perform numerical integration
                integral = scipy.integrate.simpson(y=interpolated_dist, x=np.log10(fine_grid))

                # Update bounds based on the integral value
                if integral < a_ccnc:
                    upper_bound = mid_point
                else:
                    lower_bound = mid_point
            else:
                # Interpolation failed, stop the loop
                #print(f"Interpolation failed at iteration {iteration_count}. Breaking out of loop.")
                break

        # Check for convergence
        if iteration_count < max_iterations and upper_bound - lower_bound <= tolerance:
            crit_diameter = 10**((np.log10(lower_bound) + np.log10(upper_bound)) / 2)
        else:
            #print(f"Reached max iterations or failed to converge for SS={an_ss:.4f}. Returning None.")
            crit_diameter = None

        critical_diameters.append(crit_diameter)

        # Debugging information
        if crit_diameter is not None:
            #print(f"Super Saturation: {an_ss:.4f}, CCN Concentration: {a_ccnc:.4f}, "
            #      f"Critical Diameter: {crit_diameter:.4f}, Iterations: {iteration_count}")
            pass
        else:
            #print(f"Super Saturation: {an_ss:.4f}, CCN Concentration: {a_ccnc:.4f}, "
            #      f"Critical Diameter: None, Iterations: {iteration_count}")
            pass

    critical_diameters = np.array(critical_diameters, dtype=object)  # Use dtype=object to accommodate None

    return critical_diameters


def calculate_kappa_fitting(Dc, Sc):
    '''
    Sc is the critical super saturation
    Dc is the critical diameter
    '''
    def func_to_fit(Dd, kappa):
        _, peak_S_D = find_peak_S_D_binary_search(Dd, kappa)
        return peak_S_D

    kappa_list = []
    for i in range(len(Sc)):
        popt, pcov = curve_fit(lambda Dd, kappa: func_to_fit(Dd, kappa), Dc[i], Sc[i], p0=0.1, bounds=(0, 2))
        kappa_list.append(popt[0])
    return kappa_list


def calculate_kappa(Dc, Sc, x0=0.001, x1=2.0, max_expand=5):
    kappa_list = []
    for i in range(len(Sc)):
        def f(k): 
            _, peak_S_D = find_peak_S_D_binary_search(Dc[i], k)
            return peak_S_D - Sc[i]

        a, b = x0, x1
        fa, fb = f(a), f(b)
        # expand b until f(a) and f(b) have opposite sign
        for _ in range(max_expand):
            if fa*fb < 0:
                break
            b *= 2
            fb = f(b)
        else:
            raise RuntimeError(f"Couldn't bracket root around {x0}–{x1}")

        sol = root_scalar(f, bracket=[a, b], method='brentq',
                          xtol=1e-8, rtol=1e-8, maxiter=1000)
        if not sol.converged:
            raise RuntimeError(f"Root solve failed at index {i}")
        kappa_list.append(sol.root)

    return kappa_list

def calculate_critical_diameter(kappa_list, Sc): #
    '''
    Finding the smallest particle diameter with a certain kappa that will activate under an Sc.
    Given a list of kappa values and the corresponding critical supersaturations (Sc),
    this function returns the calculated critical diameters (Dc).
    '''
    Dc_list = []
    for i in range(len(Sc)):
        def func_to_solve(Dd):
            _, peak_S_D = find_peak_S_D_binary_search(Dd, kappa_list[i])
            return peak_S_D - Sc[i]
        
        # Use a numerical solver to find the root (Dc) where func_to_solve equals zero
        Dc_solution = root_scalar(func_to_solve, bracket=[1e-9, 1e-5], method='brentq')  # Adjust bracket range as needed
        Dc_list.append(Dc_solution.root)
        
    return Dc_list



def plot_Sc_Dd_base(fig = None, axis = None, figsize=(5,5)):
    
    if (fig is None) or (axis is None):
        fig, axis = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    
    ddry = np.logspace(1, 3, 20)  # 0.01 um to 1 um

    Dw_k_1, Sc_k_1         = find_peak_S_D_binary_search(ddry*1E-9, 1)
    Dw_k_01, Sc_k_01       = find_peak_S_D_binary_search(ddry*1E-9, 0.1)
    Dw_k_001, Sc_k_001     = find_peak_S_D_binary_search(ddry*1E-9, 0.01)
    Dw_k_0001, Sc_k_0001   = find_peak_S_D_binary_search(ddry*1E-9, 0.001)
    Dw_k_00001, Sc_k_00001 = find_peak_S_D_binary_search(ddry*1E-9, 0.0001)
    Dw_k_0, Sc_k_0         = find_peak_S_D_binary_search(ddry*1E-9, 0)

    axis.plot(ddry*1E-3, (Sc_k_1-1)*100, c = 'k', ls = 'solid', alpha = 0.75)
    axis.plot(ddry*1E-3, (Sc_k_01-1)*100, c = 'k', ls = 'dashed', alpha = 0.75)
    axis.plot(ddry*1E-3, (Sc_k_001-1)*100, c = 'k', ls = 'dashdot', alpha = 0.75)
    axis.plot(ddry*1E-3, (Sc_k_0001-1)*100, c = 'k', ls = 'dotted', alpha = 0.75)
    axis.plot(ddry*1E-3, (Sc_k_0-1)*100, c = 'k', ls = 'solid', lw = 3, alpha = 0.75)

    axis.set_xlabel('Dry Diameter (µm)')
    axis.set_xscale('log')

    axis.set_ylabel('Critical Supersaturation (%)')
    axis.set_yscale('log')
    axis.text(0.016, 1.9, r'$\kappa=1$' , ha='center')
    axis.text(0.034, 1.9, r'$0.1$' , ha='right')
    axis.text(0.064, 1.9, r'$0.01$' , ha='right')
    axis.text(0.093, 1.9, r'$0.001$' , ha='center')

    axis.text(0.3, 0.7, r'$\kappa=0$', ha='center', rotation=-56, fontweight='bold')

    axis.set_ylim(0.08,1.8)
    axis.set_xlim(0.01,1)

    return fig, axis

def calculate_wet_diameter(S, dry_diameter, kappa, surface_tension=0.072, Mw=18.01528, T=298.15, density=997, R=8.3145):
    """
        Either S or the dry_diameter could be an 1d array. Not both.
    """

    def f(wet_diameter, S_element):
        return S_petter_and_Kreidenweis_2010_EQ6(wet_diameter, dry_diameter, kappa, surface_tension, Mw, T, density, R) - S_element
    initial_value = dry_diameter * 1
    if np.isscalar(S):
        return scipy.optimize.fsolve(f, initial_value, args=(S))
    else:
        wet_diameters = np.zeros_like(S)
        for i, S_element in enumerate(S):
            wet_diameters[i] = scipy.optimize.fsolve(f, initial_value, args=(S_element))
        return wet_diameters


def calculate_humidification_factor(dry_sizes, dndlogdps, rh, kappa, wavelength, dry_ri_n, dry_ri_k, water_ri_n = 1.33):
    '''
        dry_sizes : an array in nanometers
        dndlogdps : an array
        rh        : a scalar
        kappa     : a scalar
        Wavelength: in nanometers
    '''

    wet_sizes      = calculate_wet_diameter(rh, dry_sizes, kappa)
    dry_v_raitos   = dry_sizes**3 / wet_sizes**3
    water_v_ratios = 1 - dry_v_raitos
    wet_ri_ns      = dry_ri_n * dry_v_raitos + water_ri_n * water_v_ratios
    #wet_ri_ns      = dry_ri_n
    wet_ris    = wet_ri_ns - dry_ri_k * 1j
    dry_ri     = dry_ri_n  - dry_ri_k * 1j

    wet_extinction_coeff, wet_scattering_coeff, wet_bckscatter_coeff = calculate_coefficients(wet_sizes, dndlogdps, wet_ris, wavelength)
    dry_extinction_coeff, dry_scattering_coeff, dry_bckscatter_coeff = calculate_coefficients(dry_sizes, dndlogdps, dry_ri , wavelength)

    ext_humidificaiton_factor = wet_extinction_coeff/dry_extinction_coeff
    sca_humidificaiton_factor = wet_scattering_coeff/dry_scattering_coeff
    bck_humidificaiton_factor = wet_bckscatter_coeff/dry_bckscatter_coeff

    if rh == 1.0:
        print(ext_humidificaiton_factor)

    return ext_humidificaiton_factor, sca_humidificaiton_factor, bck_humidificaiton_factor


def calculate_humidification_factor_ammonium_sulfate(dry_sizes, dndlogdps, rh, kappa, wavelength):
    '''
        Cotterell er al. 2017
        
        dry_sizes : an array
        dndlogdps : an array
        rh        : a scalar
        kappa     : a scalar
        Wavelength: in nanometers
    '''

    as_rh     = np.array([0.4  , 0.5 , 0.6  , 0.7  , 0.8  , 0.9  , 1.0  ])
    as_realri = np.array([1.453, 1.44, 1.428, 1.417, 1.403, 1.379, 1.335])
    cs = CubicSpline(as_rh, as_realri, bc_type='not-a-knot')

    wet_sizes = calculate_wet_diameter(rh, dry_sizes, kappa)
    dry_ri_n = cs(0.3)
    dry_ri_k = 0

    wet_ri_ns = cs(rh)
    
    wet_ris    = wet_ri_ns - dry_ri_k * 1j
    dry_ri     = dry_ri_n  - dry_ri_k * 1j

    wet_extinction_coeff, wet_scattering_coeff, wet_bckscatter_coeff = calculate_coefficients(wet_sizes, dndlogdps, wet_ris, wavelength)
    dry_extinction_coeff, dry_scattering_coeff, dry_bckscatter_coeff = calculate_coefficients(dry_sizes, dndlogdps, dry_ri , wavelength)


    ext_humidificaiton_factor = wet_extinction_coeff/dry_extinction_coeff
    sca_humidificaiton_factor = wet_scattering_coeff/dry_scattering_coeff
    bck_humidificaiton_factor = wet_bckscatter_coeff/dry_bckscatter_coeff


    return ext_humidificaiton_factor, sca_humidificaiton_factor, bck_humidificaiton_factor



def calculate_coefficients(sizes, dndlogdps, ri, wavelength):
    '''
        Calculate extinction coefficient, scattering coefficient, or backscatter coefficient
    '''

    size_parameters = np.pi * sizes/wavelength
    qexts, qscas, qbcks, gs = miepython.mie(ri, size_parameters)


    exts_times_dndlogdp = np.pi * (sizes**2)/4 * qexts * dndlogdps
    scas_times_dndlogdp = np.pi * (sizes**2)/4 * qscas * dndlogdps
    bcks_times_dndlogdp = np.pi * (sizes**2)/4 * qbcks * dndlogdps

    non_nan_index = np.where(~np.isnan(dndlogdps))[0]

    extinction_coeff = scipy.integrate.simpson(y = exts_times_dndlogdp[non_nan_index], x = np.log10(sizes[non_nan_index]))
    scattering_coeff = scipy.integrate.simpson(y = scas_times_dndlogdp[non_nan_index], x = np.log10(sizes[non_nan_index]))
    bckscatter_coeff = scipy.integrate.simpson(y = bcks_times_dndlogdp[non_nan_index], x = np.log10(sizes[non_nan_index]))

    return extinction_coeff, scattering_coeff, bckscatter_coeff
