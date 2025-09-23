import numpy as np
import pyccl as ccl
import matplotlib.pyplot as plt
import HaloProfiles as hp

#%%
# FRB auto-correlations

#%%

COSMO_P18 = {"Omega_c": 0.26066676,
             "Omega_b": 0.048974682,
             "h": 0.6766,
             "n_s": 0.9665,
             "sigma8": 0.8102,
             "matter_power_spectrum": "halofit"}
             
cosmo = ccl.Cosmology(**COSMO_P18)
cosmo.compute_growth()

# FRB redshift distribution
alpha = 3.5
zz = np.linspace(0, 2, 128)
aa = 1/(1+zz)
nz = zz**2 * np.exp(-alpha*zz)
nz = nz/np.trapz(nz, zz)

# Alonso 2021 redshift distribution
nz_a = np.exp(-0.5*((zz-1.0)/0.15)**2)/np.sqrt(2*np.pi*0.15**2)

#%%

# Want [G] = [cm^3 kg^{-1} s^{-2}]
G_m3_per_kg_per_s2 = ccl.physical_constants.GNEWT
G_cm3_per_kg_per_s2 = 1e6 * G_m3_per_kg_per_s2
G = G_cm3_per_kg_per_s2

# Want [m_p] = [kg]
mp_kg = 1.67262e-27
mp = mp_kg

# Want [H_0] = [Mpc s^{-1} Mpc^{-1}]
pc = 3.0857e13 # 1pc in km
km_to_Mpc = 1/(1e6*pc) # 1 km = 3.24078e-20 Mpc
H0_per_s = cosmo['H0'] * km_to_Mpc
H0 = H0_per_s

# Prefactor in units of [A] = [cm^{-3}]
xH = 0.75
A = (3*cosmo['Omega_b']*H0**2)/(8*np.pi*G*mp) * (1+xH)/2

#%%

from scipy.integrate import cumulative_trapezoid

# Cumulative integral of n(z)
nz_integrated = 1 - cumulative_trapezoid(nz, zz, initial=0)
nz_integrated_a = 1 - cumulative_trapezoid(nz_a, zz, initial=0)

#%%

# [W_{\chi}] = [A] = [cm^{-3}]
# Factor of 1e6 so that Cl is in units of [pc cm^{-3}]
h = cosmo['H0'] / 100
chis = ccl.comoving_radial_distance(cosmo, aa)
W_chi = A * (1+zz) * nz_integrated * 1e6
W_chi_a = A * (1+zz) * nz_integrated_a * 1e6
#W_chi[chis<400] = 0.0
#W_chi_a[chis<400] = 0.0

# W(chi) d(chi) = W(z) dz x d(chi)/dz
dchi_dz = np.gradient(chis, zz)
dz_dchi = 1/dchi_dz
W_z = W_chi * dchi_dz
W_z_a = W_chi_a * dchi_dz

t_frb = ccl.Tracer()
t_frb.add_tracer(cosmo, kernel=(chis, W_chi))

t_frb_a = ccl.Tracer()
t_frb_a.add_tracer(cosmo, kernel=(chis, W_chi_a))

#%%

plt.rcParams.update({
    "mathtext.fontset": "stix",
    "font.family": "serif",
    "font.size": 16})

# Plot radial kernels as a function of chi
plt.figure(figsize=(8, 6))
plt.plot(chis, W_chi_a, linewidth=2, color='crimson', label=r'$n_{\rm Alonso}(z)$')
plt.plot(chis, W_chi, linewidth=2, color='mediumblue', label=r'$n_{\rm Reischke}(z)$')
plt.xlabel(r'$\chi$', fontsize=24)
plt.ylabel(r'$W(\chi)$', fontsize=24)
plt.xlim(0, 5000)
plt.tick_params(which='both', top=True, right=True, direction='in', width=1, length=5)
plt.legend(fontsize=20, frameon=False, ncol=1, loc="upper right")
#plt.savefig('dm_radial_kernel.pdf', format="pdf", bbox_inches="tight")
plt.show()

#%%

k_arr = np.logspace(-3, 3, 48)
lk_arr = np.log(k_arr)
a_arr = aa[::-1]

# Halo model implementation
hmd_200c = ccl.halos.MassDef200c
cM = ccl.halos.ConcentrationDuffy08(mass_def=hmd_200c)
nM = ccl.halos.MassFuncTinker08(mass_def=hmd_200c)
bM = ccl.halos.HaloBiasTinker10(mass_def=hmd_200c)
pM = ccl.halos.HaloProfileNFW(mass_def=hmd_200c, concentration=cM)
pE = hp.HaloProfileDensityHE(mass_def=hmd_200c, concentration=cM, lMc=14.0, beta=0.6, A_star=0.03, eta_b=0.5)
hmc = ccl.halos.HMCalculator(mass_function=nM, halo_bias=bM, mass_def=hmd_200c, log10M_max=15.0, log10M_min=10.0, nM=32)

# m-m, e-e, and e-m power spectra
pk_mm = ccl.halos.halomod_Pk2D(cosmo, hmc, pM, lk_arr=lk_arr, a_arr=a_arr)
pk_ee = ccl.halos.halomod_Pk2D(cosmo, hmc, pE, prof2=pE, lk_arr=lk_arr, a_arr=a_arr)
pk_em = ccl.halos.halomod_Pk2D(cosmo, hmc, pE, prof2=pM, lk_arr=lk_arr, a_arr=a_arr)

# Non-linear power spectrum
pk = cosmo.get_nonlin_power()

#%%

ls = np.unique(np.geomspace(1, 500, 256).astype(int)).astype(float)

# DM-DM auto-correlation
cls_frb_00 = ccl.angular_cl(cosmo, t_frb, t_frb, ls, p_of_k_a=pk_ee)
cls_frb_aa = ccl.angular_cl(cosmo, t_frb_a, t_frb_a, ls, p_of_k_a=pk)
cls_frb_0a = ccl.angular_cl(cosmo, t_frb, t_frb, ls, p_of_k_a=pk)
cls_frb_a0 = ccl.angular_cl(cosmo, t_frb_a, t_frb_a, ls, p_of_k_a=pk_ee)

#%%

# Plot angular power spectra without noise
plt.figure(figsize=(8, 6))
plt.plot(ls, cls_frb_a0, color='crimson', linewidth=2, label=r'$n_{\rm Alonso}(z)$, $P_{\rm ee}(k)$')
plt.plot(ls, cls_frb_aa, color='crimson', linestyle='dashed', label=r'$n_{\rm Alonso}(z)$, $P_{\rm nonlin}(k)$')
plt.plot(ls, cls_frb_00, color='mediumblue', linewidth=2, label=r'$n_{\rm Reischke}(z)$, $P_{\rm ee}(k)$')
plt.plot(ls, cls_frb_0a, color='mediumblue', linestyle='dashed', label=r'$n_{\rm Reischke}(z)$, $P_{\rm nonlin}(k)$')
plt.xlabel(r'$\ell$', fontsize=24)
plt.ylabel(r'$C_{\ell}^{\mathcal{DD}} \;\; [\rm pc^2 \, cm^{-6}]$', fontsize=24)
plt.xlim(1, 500)
plt.loglog()
plt.tick_params(which='both', top=True, right=True, direction='in', width=1, length=5)
plt.legend(fontsize=20, frameon=False, ncol=1, loc="lower left")
#plt.savefig('dm_effect_of_nz.pdf', format="pdf", bbox_inches="tight")
plt.show()

#%%

# FRB shot noise
sigma_host_0 = 90 # cm^{-3} pc
integrand = nz * sigma_host_0**2 / (1+zz)**2
sigma_host_sq = np.trapz(integrand, zz)
n_per_deg_sq = 0.5 # number density per degree squared
n_per_sr = n_per_deg_sq * (180/np.pi)**2
nl_frb = np.ones(len(ls)) * sigma_host_sq / n_per_sr

# Plot angular power spectra with noise
plt.figure(figsize=(8, 6))
plt.plot(ls, cls_frb_a0+nl_frb, color='crimson', linewidth=2, label=r'$n_{\rm Alonso}(z)$, $P_{\rm ee}(k)$')
plt.plot(ls, cls_frb_aa+nl_frb, color='crimson', linestyle='dashed', label=r'$n_{\rm Alonso}(z)$, $P_{\rm nonlin}(k)$')
plt.plot(ls, cls_frb_00+nl_frb, color='mediumblue', linewidth=2, label=r'$n_{\rm Reischke}(z)$, $P_{\rm ee}(k)$')
plt.plot(ls, cls_frb_0a+nl_frb, color='mediumblue', linestyle='dashed', label=r'$n_{\rm Reischke}(z)$, $P_{\rm nonlin}(k)$')
plt.xlabel(r'$\ell$', fontsize=24)
plt.ylabel(r'$C_{\ell}^{\mathcal{DD}} + N_{\ell}^{\mathcal{DD}} \;\; [\rm pc^2 \, cm^{-6}]$', fontsize=24)
plt.xlim(1, 500)
plt.loglog()
plt.tick_params(which='both', top=True, right=True, direction='in', width=1, length=5)
plt.legend(fontsize=20, frameon=False, ncol=1, loc="upper right")
plt.show()

#%%
# Cross-correlations with WL

#%%

# Lensing redshift distributions
d = np.load('data/redshift_distributions_lsst.npz')
z = d['z']
dndz = d['dndz']
ndens = d['ndens_arcmin']

# WL tracers
t_wl0 = ccl.WeakLensingTracer(cosmo, dndz=(z, dndz[0,:]))
t_wl1 = ccl.WeakLensingTracer(cosmo, dndz=(z, dndz[1,:]))
t_wl2 = ccl.WeakLensingTracer(cosmo, dndz=(z, dndz[2,:]))
t_wl3 = ccl.WeakLensingTracer(cosmo, dndz=(z, dndz[3,:]))
t_wl4 = ccl.WeakLensingTracer(cosmo, dndz=(z, dndz[4,:]))
t_wls = [t_wl0, t_wl1, t_wl2, t_wl3, t_wl4]

# DM-WL cross-correlations
pk_em = pk
cls_x0 = ccl.angular_cl(cosmo, t_wl0, t_frb, ls, p_of_k_a=pk_em)
cls_x1 = ccl.angular_cl(cosmo, t_wl1, t_frb, ls, p_of_k_a=pk_em)
cls_x2 = ccl.angular_cl(cosmo, t_wl2, t_frb, ls, p_of_k_a=pk_em)
cls_x3 = ccl.angular_cl(cosmo, t_wl3, t_frb, ls, p_of_k_a=pk_em)
cls_x4 = ccl.angular_cl(cosmo, t_wl4, t_frb, ls, p_of_k_a=pk_em)
cls_x = [cls_x0, cls_x1, cls_x2, cls_x3, cls_x4]

#%%

# Plot DM-DM auto-correlation
plt.figure(figsize=(8, 6))
plt.plot(ls, cls_frb_0a, color='black', linewidth=2, label=r'$\mathcal{D}-\mathcal{D}$')
plt.xlabel(r'$\ell$', fontsize=24)
plt.ylabel(r'$C_{\ell}^{\mathcal{DD}} \;\; [\rm pc^2 \, cm^{-6}]$', fontsize=24)
plt.xlim(2, 500)
plt.loglog()
plt.tick_params(which='both', top=True, right=True, direction='in', width=1, length=5)
#plt.savefig('dm_power_spectrum_test_plot.pdf', format="pdf", bbox_inches="tight")
plt.show()

#%%

# Plot DM-WL cross-correlations
plt.figure(figsize=(8, 6))
plt.plot(ls, cls_x0, color='mediumblue', linewidth=2, label=r'$\gamma^{(0)}-\mathcal{D}$')
plt.plot(ls, cls_x1, color='deepskyblue', linewidth=2, label=r'$\gamma^{(1)}-\mathcal{D}$')
plt.plot(ls, cls_x2, color='blueviolet', linewidth=2, label=r'$\gamma^{(2)}-\mathcal{D}$')
plt.plot(ls, cls_x3, color='hotpink', linewidth=2, label=r'$\gamma^{(3)}-\mathcal{D}$')
plt.plot(ls, cls_x4, color='crimson', linewidth=2, label=r'$\gamma^{(4)}-\mathcal{D}$')
plt.xlabel(r'$\ell$', fontsize=24)
plt.ylabel(r'$C_{\ell}^{\gamma\mathcal{D}} \;\; [\rm pc \, cm^{-3}]$', fontsize=24)
plt.xlim(2, 500)
plt.ylim(6e-7, 2e-3)
plt.loglog()
plt.tick_params(which='both', top=True, right=True, direction='in', width=1, length=5)
plt.legend(fontsize=20, frameon=False, ncol=2, loc="lower center")
#plt.savefig('dm_wl_power_spectrum_test_plot.pdf', format="pdf", bbox_inches="tight")
plt.show()
