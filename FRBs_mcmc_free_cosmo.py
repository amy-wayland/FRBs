import sacc
import numpy as np
import pyccl as ccl
import HaloProfiles as hp
import matplotlib.pyplot as plt
from getdist import plots
from cobaya import run
from scipy.integrate import cumulative_trapezoid

#%%

s = sacc.Sacc.load_fits('frb-wl_3x2.fits')

keep_types = []
for t1, t2 in s.get_tracer_combinations():
    if t1 == 'frb' and t2 == 'frb':
        keep_types.append(('cl_ee', t1, t2))
    elif ('frb' in (t1, t2)) and ('wl_' in t1 or 'wl_' in t2):
        keep_types.append(('cl_ee', t1, t2))

s.remove_selection(lambda t, t1, t2, l: (t, t1, t2) not in keep_types)

d = s.mean
cov = s.covariance.dense
icov = np.linalg.pinv(cov)
ells = s.get_ell_cl('cl_00', 'frb', 'frb')[0]
n_ell = len(ells)

tracer_names = list(s.tracers.keys())
name_to_index = {name: i for i, name in enumerate(tracer_names)}
n_tracers = len(s.tracers)

#%%

k_arr = np.logspace(-3, 3, 48)
lk_arr = np.log(k_arr)

# Halo model implementation
hmd_200c = ccl.halos.MassDef200c
cM = ccl.halos.ConcentrationDuffy08(mass_def=hmd_200c)
nM = ccl.halos.MassFuncTinker08(mass_def=hmd_200c)
bM = ccl.halos.HaloBiasTinker10(mass_def=hmd_200c)
hmc = ccl.halos.HMCalculator(mass_function=nM, halo_bias=bM, mass_def=hmd_200c, log10M_max=15.0, log10M_min=10.0, nM=32)

#%%

# Prefactor for FRB kernel
G_m3_per_kg_per_s2 = ccl.physical_constants.GNEWT
G_cm3_per_kg_per_s2 = 1e6 * G_m3_per_kg_per_s2
G = G_cm3_per_kg_per_s2
mp_kg = 1.67262e-27
mp = mp_kg
pc = 3.0857e13
km_to_Mpc = 1/(1e6*pc)
xH = 0.75

#%%

pM = hp.HaloProfileNFWBaryon(mass_def=hmd_200c, concentration=cM, lMc=14.0, beta=0.6, A_star=0.03, eta_b=0.5)
pE = hp.HaloProfileDensityHE(mass_def=hmd_200c, concentration=cM, lMc=14.0, beta=0.6, A_star=0.03, eta_b=0.5)

def lnprob(Om_m, s8, lMc, eta_b, DM_DM=True, DM_WL=True, WL_WL=False):
    
    if eta_b < 0:
        return -np.inf
    
    cosmo = ccl.Cosmology(Omega_c=Om_m-0.048974682,
                      Omega_b=0.048974682,
                      h=0.6766,
                      n_s=0.9665,
                      sigma8=s8,
                      matter_power_spectrum='halofit')
    
    cosmo.compute_growth()
    
    H0_per_s = cosmo['H0'] * km_to_Mpc
    H0 = H0_per_s
    A = (3*cosmo['Omega_b']*H0**2)/(8*np.pi*G*mp) * (1+xH)/2
    
    pM.update_parameters(lMc=lMc, eta_b=eta_b)
    pE.update_parameters(lMc=lMc, eta_b=eta_b)
    cls_matrix = np.zeros((n_tracers, n_tracers, n_ell))
    
    pk_ee = None
    pk_em = None
    pk_mm = None
    
    if DM_DM or DM_WL:
        z_arr = s.tracers['frb'].z
        nz_arr = s.tracers['frb'].nz
        nz_arr /= np.trapz(nz_arr, z_arr)
        a_arr = 1/(1+z_arr)
        chis = ccl.comoving_radial_distance(cosmo, a_arr)
        nz_integrated = 1 - cumulative_trapezoid(nz_arr, z_arr, initial=0)
        W_chi = A * (1+z_arr) * nz_integrated * 1e6
        t_dm = ccl.Tracer()
        t_dm.add_tracer(cosmo, kernel=(chis, W_chi))
        
    if DM_WL or WL_WL:
        tracers_wl = []
        for tracer_name in s.tracers:
            if 'wl_' in tracer_name:
                tracer = s.tracers[tracer_name]
                tracers_wl.append(ccl.WeakLensingTracer(cosmo, dndz=(tracer.z, tracer.nz)))
    
    if DM_DM:
        pk_ee = ccl.halos.halomod_Pk2D(cosmo, hmc, pE, lk_arr=lk_arr, a_arr=a_arr[::-1])
        
    if DM_WL:
        pk_em = ccl.halos.halomod_Pk2D(cosmo, hmc, pE, prof2=pM, lk_arr=lk_arr, a_arr=a_arr[::-1])
        
    if WL_WL:
        pk_mm = ccl.halos.halomod_Pk2D(cosmo, hmc, pM, lk_arr=lk_arr, a_arr=a_arr[::-1])
        
    for t1, t2 in s.get_tracer_combinations():
        i = name_to_index[t1]
        j = name_to_index[t2]
        if i == j == 0: # DM-DM
            ti = tj = t_dm
            pk = pk_ee
        elif (i == 0 and j != 0) or (i != 0 and j == 0): # DM-WL
            if j == 0:
                ti, tj = tj, ti
                i, j = j, i
            ti = t_dm
            tj = tracers_wl[j-1]
            pk = pk_em
        else:
            continue
        #if i != 0 and j != 0: # WL-WL
        #    ti = tracers_wl[i-1]
        #    tj = tracers_wl[j-1]
        #    pk = pk_mm
        cl = ccl.angular_cl(cosmo, ti, tj, ells, p_of_k_a=pk)
        cls_matrix[i, j, :] = cl
        cls_matrix[j, i, :] = cl
            
    # Flatten the upper triangle of cls_matrix to match the SACC ordering
    m = [cls_matrix[i, j, :] for i in range(n_tracers) for j in range(i, n_tracers)]
    m = np.concatenate(m)
    
    r = d - m
    return -0.5 * np.dot(r, np.dot(icov, r))

#%%

import time
start = time.time()
p = np.array([0.26066676+0.048974682, 0.8102, 14.0, 0.5])
stop = time.time()
print(stop-start)
print(round(-lnprob(*p),2))

#%%

info = {"likelihood": {"logprob": lnprob}}

info["params"] = {
    "Om_m": {"prior": {"dist": "norm", "loc": p[0], "scale": 0.1}, "ref": p[0], "proposal": 0.01, "latex": r"\Omega_{\mathrm{m}}"},
    "s8": {"prior": {"dist": "norm", "loc": p[1], "scale": 0.1}, "ref": p[1], "proposal": 0.01, "latex": r"\sigma_8"},
    "lMc": {"prior": {"dist": "norm", "loc": p[2], "scale": 0.5}, "ref": p[2], "proposal": 0.5, "latex": r"\log_{10} M_{\mathrm{c}}"},
    "eta_b": {"prior": {"dist": "norm", "loc": p[3], "scale": 0.5}, "ref": p[3], "proposal": 0.5, "latex": r"\eta_{\rm b}"}}

info["sampler"] = {"mcmc": {"Rminus1_stop": 0.03, "max_tries": 1000}}

info["output"] = "wl-frb"

#%%

updated_info, sampler = run(info)

#%%

gd_sample = sampler.products(to_getdist=True, skip_samples=0.3)["sample"]

mean = gd_sample.getMeans()[:4]
covmat = gd_sample.getCovMat().matrix[:4, :4]
print("Mean:", mean)
print("Covariance matrix:", covmat)

g = plots.get_subplot_plotter(subplot_size=3)
g.settings.legend_fontsize = 32
g.settings.axes_labelsize = 30
g.settings.axes_fontsize = 20
g.settings.figure_legend_frame = False

g.triangle_plot(gd_sample, ["Om_m", "s8", "lMc", "eta_b"], filled=True)
plt.savefig('posteriors_frb.png', dpi=100)
plt.show()
