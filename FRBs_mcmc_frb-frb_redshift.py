import sacc
import numpy as np
import pyccl as ccl
import HaloProfiles as hp
import matplotlib.pyplot as plt
from getdist import plots
from cobaya import run
from scipy.integrate import cumulative_trapezoid

#%%

COSMO_P18 = {"Omega_c": 0.26066676,
             "Omega_b": 0.048974682,
             "h": 0.6766,
             "n_s": 0.9665,
             "sigma8": 0.8102,
             "matter_power_spectrum": "halofit"}
             
cosmo = ccl.Cosmology(**COSMO_P18)
cosmo.compute_growth()

#%%

s = sacc.Sacc.load_fits('frb-frb.fits')
d = s.mean
cov = s.covariance.dense
icov = np.linalg.pinv(cov)
ells = s.get_ell_cl('cl_00', 'frb', 'frb')[0]

#%%

# Halo model implementation
hmd_200c = ccl.halos.MassDef200c
cM = ccl.halos.ConcentrationDuffy08(mass_def=hmd_200c)
nM = ccl.halos.MassFuncTinker08(mass_def=hmd_200c)
bM = ccl.halos.HaloBiasTinker10(mass_def=hmd_200c)
pM = ccl.halos.HaloProfileNFW(mass_def=hmd_200c, concentration=cM)
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
H0_per_s = cosmo['H0'] * km_to_Mpc
H0 = H0_per_s
xH = 0.75
A = (3*cosmo['Omega_b']*H0**2)/(8*np.pi*G*mp) * (1+xH)/2

#%%

k_arr = np.logspace(-3, 3, 48)
lk_arr = np.log(k_arr)

tr = s.tracers["frb"]
zz = tr.z
aa = 1/(1+zz)
chis = ccl.comoving_radial_distance(cosmo, aa)

#%%

pE = hp.HaloProfileDensityHE(mass_def=hmd_200c, concentration=cM, lMc=14.0, beta=0.6, A_star=0.03, eta_b=0.5)

def lnprob(lMc, eta_b, alpha):
    
    if eta_b < 0 or alpha < 0:
        return -np.inf
    
    nz = zz**2 * np.exp(-alpha*zz)
    nz /= np.trapz(nz, zz)
    nz_integrated = 1 - cumulative_trapezoid(nz, zz, initial=0)
    W_chi = A * (1+zz) * nz_integrated * 1e6
    t_frb = ccl.Tracer()
    t_frb.add_tracer(cosmo, kernel=(chis, W_chi))
    
    pE.update_parameters(lMc=lMc, eta_b=eta_b)
    pk_ee = ccl.halos.halomod_Pk2D(cosmo, hmc, pE, prof2=pE, lk_arr=lk_arr, a_arr=aa[::-1])
    Cl = ccl.angular_cl(cosmo, t_frb, t_frb, ells, p_of_k_a=pk_ee)
    r = d - Cl
    return -0.5 * np.dot(r, np.dot(icov, r))

#%%

import time
start = time.time()
nrepeat = 10
for i in range(nrepeat):
    p = np.array([14.0, 0.5, 3.5])
stop = time.time()
print((stop-start)/nrepeat)
print(np.round(-lnprob(*p), 2))

#%%

info = {"likelihood": {"logprob": lnprob}}

info["params"] = {
    "lMc": {"prior": {"dist": "norm", "loc": p[0], "scale": 0.5}, "ref": p[0], "proposal": 0.5, "latex": r"\log_{10} M_{\mathrm{c}}"},
    "eta_b": {"prior": {"dist": "norm", "loc": p[1], "scale": 0.5}, "ref": p[1], "proposal": 0.5, "latex": r"\eta_{\rm b}"},
    "alpha": {"prior": {"dist": "norm", "loc": p[2], "scale": 0.5}, "ref": p[2], "proposal": 0.5, "latex": r"\alpha"}}

info["sampler"] = {"mcmc": {"Rminus1_stop": 0.03, "max_tries": 1000}}

info["output"] = "wl-frb"

#%%

updated_info, sampler = run(info)

#%%

gd_sample = sampler.products(to_getdist=True, skip_samples=0.3)["sample"]

mean = gd_sample.getMeans()[:3]
covmat = gd_sample.getCovMat().matrix[:3, :3]
print("Mean:", mean)
print("Covariance matrix:", covmat)

g = plots.get_subplot_plotter(subplot_size=3)
g.settings.legend_fontsize = 32
g.settings.axes_labelsize = 30
g.settings.axes_fontsize = 20
g.settings.figure_legend_frame = False

g.triangle_plot(gd_sample, ["lMc", "eta_b"], filled=True)
plt.savefig('posteriors_frb.png', dpi=100)
plt.show()
