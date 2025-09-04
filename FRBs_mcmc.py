import sacc
import numpy as np
import pyccl as ccl
import matplotlib.pyplot as plt
from getdist import plots
from cobaya import run

#%%

s = sacc.Sacc.load_fits('wl-frb.fits')
s.get_tracer_combinations()
d = s.mean
cov = s.covariance.dense
icov = np.linalg.pinv(cov)

# Get the l values from the data
ells, cells = s.get_ell_cl('cl_ee', 'wl_0', 'wl_0')

#%%

# Define the k and a arrays used to compute the non-linear power spectrum
a_array = np.linspace(1/(1+4.0), 1, 64)
l10k_array = np.linspace(-3, 1, 255)
k_array = 10**l10k_array

# Find the index corresponding to the minimum a value of BACCO
a_min_index = np.where(a_array >= 1/(1+1.5))[0][0]

# Array of a values for which BACCO is calibrated
a_subset = a_array[a_min_index:]

# Intrinsic alignment model
def A_IA(amp, eta, z, z_star=0.62):
    return amp * ((1+z)/(1+z_star))**eta

#%%

hfix = 0.6766
logh = np.log10(hfix)

def lnprob(Om_m, s8, A_IA0, eta_IA, log_Mc, log_eta, log_beta, log_M1,
           log_theta_inn, log_theta_out, log_M_inn):
    
    if ((Om_m < 0.23) or
        (Om_m > 0.40) or
        (s8 < 0.73) or
        (s8 > 0.90) or
        (A_IA0 < -5.0) or 
        (A_IA0 > 5.0) or
        (eta_IA < -5.0) or
        (eta_IA > 5.0) or
        (log_Mc+logh < 9.0) or
        (log_Mc+logh > 15.0) or
        (log_eta < -0.70) or
        (log_eta > 0.70) or
        (log_beta < -1.0) or
        (log_beta > 0.70) or
        (log_M1+logh < 9.0) or
        (log_M1+logh > 13.0) or
        (log_theta_inn < -2.0) or
        (log_theta_inn > -0.5) or
        (log_theta_out < 0.0) or
        (log_theta_out > 0.47) or
        (log_M_inn+logh < 9.0) or
        (log_M_inn+logh > 13.5)):
        return -np.inf
    
    cosmo = ccl.Cosmology(Omega_c=Om_m-0.048974682,
                      Omega_b=0.048974682,
                      h=hfix,
                      n_s=0.9665,
                      sigma8=s8,
                      transfer_function="eisenstein_hu")

    cosmo.compute_growth()
    
    pk_a = np.array([ccl.nonlin_matter_power(cosmo, k_array, a) for a in a_array])
    
    bar = ccl.BaccoemuBaryons(log10_M_c=log_Mc, log10_eta=log_eta,
                              log10_beta=log_beta, log10_M1_z0_cen=log_M1,
                              log10_theta_out=log_theta_out,
                              log10_theta_inn=log_theta_inn,
                              log10_M_inn=log_M_inn)
    
    boost = np.zeros([64, 255])
    bacco_boost = bar.boost_factor(cosmo, k_array, a_subset)
    boost[a_min_index:, :] = bacco_boost
    boost[:a_min_index, :] = bacco_boost[0][None, :]
    pk_bar = pk_a * boost
    Pk2D = ccl.Pk2D(a_arr=a_array, lk_arr=np.log(k_array), pk_arr=np.log(pk_bar))
    
    m = []
    for t1, t2 in s.get_tracer_combinations():
        t_wl = ccl.WeakLensingTracer(cosmo, dndz=(s.tracers[t1].z, s.tracers[t2].nz), 
                                      ia_bias=(s.tracers[t1].z, A_IA(amp=A_IA0, eta=eta_IA, z=s.tracers[t1].z)))
        t_frb = ccl.NumberCountsTracer(cosmo, dndz=(s.tracers[t2].z, s.tracers[t2].nz), has_rsd=False)
        Cl = ccl.angular_cl(cosmo, t_wl, t_frb, ells, p_of_k_a=Pk2D)
        m.append(Cl)
    m = np.concatenate(m)
    r = d - m
    return -0.5 * np.dot(r, np.dot(icov, r))

#%%

# True values of the cosmological and baryonic parameters used to generate the mock data
p = np.array([0.26066676+0.048974682, 0.8102, 0, 0, 14.0, -0.3, -0.22, 10.674, -0.86, 0.25, 12.6])

#%%

info = {"likelihood": {"logprob": lnprob}}

info["params"] = {
    "Om_m": {"prior": {"dist": "norm", "loc": p[0], "scale": 0.1}, "ref": p[0], "proposal": 0.01, "latex": r"\Omega_{\mathrm{m}}"},
    "s8": {"prior": {"dist": "norm", "loc": p[1], "scale": 0.1}, "ref": p[1], "proposal": 0.01, "latex": r"\sigma_8"},
    "A_IA0" : {"prior": {"dist": "norm", "loc": p[2], "scale": 1.0}, "ref": p[2], "proposal": 0.05, "latex": r"A_{\mathrm{IA}}"},
    "eta_IA" : {"prior": {"dist": "norm", "loc": p[3], "scale": 1.0}, "ref": p[3], "proposal": 0.05, "latex": r"\eta_{\mathrm{IA}}"},
    "log_Mc": {"prior": {"dist": "norm", "loc": p[4], "scale": 0.5}, "ref": p[4], "proposal": 0.5, "latex": r"\log M_{\mathrm{c}}"},
    "log_eta": {"prior": {"dist": "norm", "loc": p[5], "scale": 0.5}, "ref": p[5], "proposal": 0.5, "latex": r"\log \eta"},
    "log_beta": {"prior": {"dist": "norm", "loc": p[6], "scale": 0.5}, "ref": p[6], "proposal": 0.5, "latex": r"\log \beta"},
    "log_M1": {"prior": {"dist": "norm", "loc": p[7], "scale": 0.5}, "ref": p[7], "proposal": 0.5, "latex": r"\log M_1"}, 
    "log_theta_inn": {"prior": {"dist": "norm", "loc": p[8], "scale": 0.5}, "ref": p[8], "proposal": 0.5, "latex": r"\log \theta_{\mathrm{inn}}"}, 
    "log_theta_out": {"prior": {"dist": "norm", "loc": p[9], "scale": 0.5}, "ref": p[9], "proposal": 0.5, "latex": r"\log \theta_{\mathrm{out}}"}, 
    "log_M_inn": {"prior": {"dist": "norm", "loc": p[10], "scale": 0.5}, "ref": p[10], "proposal": 0.5, "latex": r"\log M_{\mathrm{inn}}"}}

info["sampler"] = {"mcmc": {"Rminus1_stop": 0.03, "max_tries": 1000}}

info["output"] = "wl-frb"

#%%

updated_info, sampler = run(info)

#%%

gd_sample = sampler.products(to_getdist=True, skip_samples=0.3)["sample"]

mean = gd_sample.getMeans()[:11]
covmat = gd_sample.getCovMat().matrix[:11, :11]
print("Mean:", mean)
print("Covariance matrix:", covmat)

g = plots.get_subplot_plotter(subplot_size=3)
g.settings.legend_fontsize = 32
g.settings.axes_labelsize = 30
g.settings.axes_fontsize = 20
g.settings.figure_legend_frame = False

g.triangle_plot(gd_sample, ["Om_m", "s8", "log_Mc", "log_eta", "log_beta", "log_M1", "log_theta_inn", "log_theta_out", "log_M_inn"], filled=True)
plt.savefig('posteriors_wl_frb.png', dpi=100)
plt.show()
