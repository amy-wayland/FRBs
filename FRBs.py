import numpy as np
import pyccl as ccl
import HaloProfiles as hp
import matplotlib.pyplot as plt

#%%

plt.rcParams.update({
    "mathtext.fontset": "stix",
    "font.family": "serif",
    "font.size": 16})

#%%
# Cosmology

COSMO_P18 = {"Omega_c": 0.26066676,
             "Omega_b": 0.048974682,
             "h": 0.6766,
             "n_s": 0.9665,
             "sigma8": 0.8102,
             "transfer_function": "eisenstein_hu"}

cosmo = ccl.Cosmology(**COSMO_P18)
cosmo.compute_growth()

#%%
# Redshift Distribution

A = 3.5
zz = np.linspace(0.1, 5, 32)
nz = zz**2 * np.exp(-A*zz)

#%%
# FRB Kernel

mp = 1.0 # 1.67e-27 # proton mass in kg
C = (3 * cosmo['H0']**2 * cosmo['Omega_b'] * ccl.comoving_angular_distance(cosmo, 1/(1+0.1))) \
    / (8 * np.pi * ccl.physical_constants.GNEWT * mp) # prefactor

C = 1.0
a_arr = 1/(1+zz[::-1])
dz_dchi = np.gradient(zz / ccl.comoving_angular_distance(cosmo, a_arr))
E_of_z = (ccl.h_over_h0(cosmo, a_arr) / ccl.physical_constants.CLIGHT_HMPC)**2
F_of_z = 0.1

#%%
# FRB Tracer

chis = ccl.comoving_radial_distance(cosmo, 1/(1+zz))

W_chi = C * F_of_z * (1+zz) * dz_dchi / E_of_z
W_frb = W_chi * np.trapz(nz, zz) / dz_dchi

t_frb = ccl.Tracer()
t_frb.add_tracer(cosmo, kernel=(chis, W_frb))

#%%
# Plot

plt.figure(figsize=(8, 6))
plt.plot(chis, W_frb, color='mediumblue', linewidth=2)
plt.xlabel(r'$\chi$', fontsize=26)
plt.ylabel(r'$W_{\mathcal{D}}(\chi)$', fontsize=26)
plt.loglog()
plt.xlim(chis[0], chis[-1])
plt.tick_params(which='both', top=True, right=True, direction='in', width=1, length=5)
plt.show()

#%%
# 3D Power Spectrum - HE Model

k_arr = np.logspace(-3, 1, 128)
lk_arr = np.log(k_arr)

hmd_200c = ccl.halos.MassDef200c
cM = ccl.halos.ConcentrationDuffy08(mass_def=hmd_200c)
nM = ccl.halos.MassFuncTinker08(mass_def=hmd_200c)
bM = ccl.halos.HaloBiasTinker10(mass_def=hmd_200c)
pM = ccl.halos.HaloProfileNFW(mass_def=hmd_200c, concentration=cM)
pE = hp.HaloProfileDensityHE(mass_def=hmd_200c, concentration=cM, lMc=14.0, beta=0.6, A_star=0.03, eta_b=0.5)
hmc = ccl.halos.HMCalculator(mass_function=nM, halo_bias=bM, mass_def=hmd_200c, log10M_max=15.0, log10M_min=10.0, nM=32)
pk_mm = ccl.halos.halomod_Pk2D(cosmo, hmc, pM, lk_arr=lk_arr, a_arr=a_arr)
pk_ee = ccl.halos.halomod_Pk2D(cosmo, hmc, pE, prof2=pE, lk_arr=lk_arr, a_arr=a_arr)
pk_em = ccl.halos.halomod_Pk2D(cosmo, hmc, pM, prof2=pE, lk_arr=lk_arr, a_arr=a_arr)

#%%
# 3D Power Spectrum - BaccoEmu Model

emu_pknl = ccl.emulators.BaccoemuNonlinear()
emu_bar = ccl.baryons.BaryonsBaccoemu()

pk_dmo = emu_pknl.get_pk2d(cosmo)

emu_bar.update_parameters(log10_M_c=14.0, log10_eta=-0.3,
                              log10_beta=-0.22, log10_M1_z0_cen=10.674,
                              log10_theta_out=0.25,
                              log10_theta_inn=-0.86,
                              log10_M_inn=13.574)

pk_bar = emu_bar.include_baryonic_effects(cosmo, pk_dmo)

#%%
# FRB-FRB Angular Power Spectrum

ls = np.unique(np.geomspace(2, 2000, 256).astype(int)).astype(float)
pk_ee = cosmo.get_nonlin_power()
cls_frb = ccl.angular_cl(cosmo, t_frb, t_frb, ls, p_of_k_a=pk_em)

n = 2e4 # number of FRBs in sample 

sigma_host0_pc_per_cm3 = 50
sigma_host0 = sigma_host0_pc_per_cm3

integrand = nz * sigma_host0**2 / (1+zz)**2
sigma_host = np.trapz(integrand, zz)

cls_frb = cls_frb + sigma_host**2 / n

#%%
# Plot

plt.figure(figsize=(8, 6))
plt.plot(ls, cls_frb, color='mediumblue', linewidth=2)
plt.xlabel(r'$\ell$', fontsize=26)
plt.ylabel(r'$C_{\ell}^{\mathcal{DD}}$', fontsize=26)
plt.xlim(2, 2000)
plt.loglog()
plt.tick_params(which='both', top=True, right=True, direction='in', width=1, length=5)
plt.show()

#%%
# Cosmic shear

d = np.load('data/redshift_distributions_lsst.npz')

z = d['z']
dndz = d['dndz']

t_shear = ccl.WeakLensingTracer(cosmo, dndz=(z, dndz[2,:]))
cls_x = ccl.angular_cl(cosmo, t_shear, t_frb, ls, p_of_k_a=pk_mm)

#%%
# Plot

plt.figure(figsize=(8, 6))
#plt.plot(ls, cls_frb, color='mediumblue', linewidth=2, label=r'$\mathcal{D} - \mathcal{D}$')
plt.plot(ls, cls_x, color='crimson', linewidth=2, label=r'$\gamma - \mathcal{D}$')
plt.xlabel(r'$\ell$', fontsize=26)
plt.ylabel(r'$C_{\ell}^{\gamma \mathcal{D}}$', fontsize=26)
plt.xlim(2, 2000)
plt.loglog()
plt.tick_params(which='both', top=True, right=True, direction='in', width=1, length=5)
plt.show()
