import numpy as np
import pyccl as ccl
import matplotlib.pyplot as plt
import sacc

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
alpha = 2.5
zz = np.linspace(0.1, 5, 32)
nz = zz**2 * np.exp(-alpha*zz)

# Lensing redshift distributions
d = np.load('data/redshift_distributions_lsst.npz')
z = d['z']
dndz = d['dndz']
ndens = d['ndens_arcmin']

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

# Want [\chi_H] = [pc]
chi_H_Mpc = ccl.comoving_angular_distance(cosmo, a=1/(1+0.55))
chi_H_pc =  1e6 * chi_H_Mpc
chi_H = chi_H_pc

# Prefactor in units of [A] = [pc cm^{-3}]
A = (3 * H0**2 * cosmo['Omega_b'] * chi_H) / (8 * np.pi * G * mp)

#%%

aa = 1/(1+zz)
chis = ccl.comoving_radial_distance(cosmo, aa)
dchi_dz = np.gradient(chis, zz)
dz_dchi = 1 / dchi_dz

F_of_z = 0.9
W_chi = A * F_of_z * (1+zz) * nz * dz_dchi

t_frb = ccl.Tracer()
t_frb.add_tracer(cosmo, kernel=(chis, W_chi))
t_wl0 = ccl.WeakLensingTracer(cosmo, dndz=(z, dndz[0,:]))
t_wl1 = ccl.WeakLensingTracer(cosmo, dndz=(z, dndz[1,:]))
t_wl2 = ccl.WeakLensingTracer(cosmo, dndz=(z, dndz[2,:]))
t_wl3 = ccl.WeakLensingTracer(cosmo, dndz=(z, dndz[3,:]))

#%%

emu_pknl = ccl.emulators.BaccoemuNonlinear()
emu_bar = ccl.baryons.BaryonsBaccoemu()
pk_dmo = emu_pknl.get_pk2d(cosmo)

# Add baryonic effects
emu_bar.update_parameters(log10_M_c=14.0, log10_eta=-0.3,
                              log10_beta=-0.22, log10_M1_z0_cen=10.674,
                              log10_theta_out=0.25,
                              log10_theta_inn=-0.86,
                              log10_M_inn=13.574)

pk = emu_bar.include_baryonic_effects(cosmo, pk_dmo)

#%%

ls = np.unique(np.geomspace(2, 2000, 256).astype(int)).astype(float)

# DM-DM auto-correlation
cls_frb = ccl.angular_cl(cosmo, t_frb, t_frb, ls, p_of_k_a=pk)

# WL-DM cross-correlations
cls_x0 = ccl.angular_cl(cosmo, t_wl0, t_frb, ls, p_of_k_a=pk)
cls_x1 = ccl.angular_cl(cosmo, t_wl1, t_frb, ls, p_of_k_a=pk)
cls_x2 = ccl.angular_cl(cosmo, t_wl2, t_frb, ls, p_of_k_a=pk)
cls_x3 = ccl.angular_cl(cosmo, t_wl3, t_frb, ls, p_of_k_a=pk)

#%%

plt.rcParams.update({
    "mathtext.fontset": "stix",
    "font.family": "serif",
    "font.size": 16})

plt.figure(figsize=(8, 6))
plt.plot(ls, cls_frb, color='crimson', linewidth=2, label=r'$\mathcal{D}-\mathcal{D}$')
plt.plot(ls, cls_x0, color='mediumblue', linewidth=2, label=r'$\gamma^{(0)}-\mathcal{D}$')
plt.plot(ls, cls_x1, color='deepskyblue', linewidth=2, label=r'$\gamma^{(1)}-\mathcal{D}$')
plt.plot(ls, cls_x2, color='blueviolet', linewidth=2, label=r'$\gamma^{(2)}-\mathcal{D}$')
plt.plot(ls, cls_x3, color='hotpink', linewidth=2, label=r'$\gamma^{(3)}-\mathcal{D}$')
plt.xlabel(r'$\ell$', fontsize=24)
plt.ylabel(r'$C_{\ell}$', fontsize=24)
plt.xlim(2, 2000)
plt.loglog()
plt.tick_params(which='both', top=True, right=True, direction='in', width=1, length=5)
plt.legend(fontsize=20, frameon=False, ncol=1, loc="lower left")
plt.show()

#%%

n_l = len(ls)
n_bins = len(ndens)

# FRB - WL cross-correlations
Cls = np.zeros((n_bins, n_l))
for i in range(n_bins):
    t_wli = ccl.WeakLensingTracer(cosmo, dndz=(z, dndz[i,:]))
    Cls[i, :] = ccl.angular_cl(cosmo, t_wli, t_frb, ls, p_of_k_a=pk)
    
#%%

f_sky = 0.5
dl = np.gradient(ls)

# Construct covariance matrix using the Knox formula
covar = np.zeros((n_bins, n_l, n_bins, n_l))
for i in range(n_bins):
    for j in range(n_bins):
        C_iD = Cls[i, :]
        C_jD = Cls[j, :]
        cov = (C_iD * C_jD) / ((2 * ls + 1) * dl * f_sky)
        covar [i, :, j, :] = np.diag(cov)
covar = covar.reshape(n_bins * n_l, n_bins * n_l)

#%%

s = sacc.Sacc()

# Add FRB dispersion measure tracer
s.add_tracer('NZ', 'dm_tracer',
             quantity='galaxy_density',
             spin=0,
             z=zz,
             nz=nz,
             extra_columns={'error': 0.1 * nz})

# Add weak lensing tracers
for i in range(n_bins):
    s.add_tracer('NZ', 'wl_{x}'.format(x=i),
                 quantity='galaxy_shear',
                 spin=2,
                 z=z,
                 nz=dndz[i,:],
                 extra_columns={'error': 0.1*dndz[i,:]},
                 sigma_g=0.28)

# Add the angular power spectra
for i in range(n_bins):
    tracer_wl = 'wl_{}'.format(i)
    tracer_dm = 'dm_tracer'
    cl = Cls[i, :]
    s.add_ell_cl('cl_eb', tracer_wl, tracer_dm, ls, cl)

# Add covariance
s.add_covariance(covar)

# Write to SACC file
s.save_fits("wl-frb.fits", overwrite=True)
