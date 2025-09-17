import numpy as np
import pyccl as ccl
import matplotlib.pyplot as plt
import sacc
import HaloProfiles as hp

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
zz = np.linspace(0.1, 5, 183)
nz = zz**2 * np.exp(-alpha*zz)
nz = nz/np.trapz(nz, zz)
#zz = np.linspace(0, 2, 1024)
#nz = np.exp(-(0.5*(zz-1.0)/0.15)**2)/np.sqrt(2*np.pi*0.15**2)

# Lensing redshift distributions
d = np.load('data/redshift_distributions_lsst.npz')
z = d['z']
dndz = d['dndz']
ndens = d['ndens_arcmin']

#%%

h = cosmo['H0'] / 100

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

# Want [\chi_H] = [pc/h]
chi_H_Mpc = ccl.comoving_angular_distance(cosmo, a=1/(1+0.55)) / h
chi_H_pc = 1e6 * chi_H_Mpc
chi_H = chi_H_pc

# Prefactor in units of [A] = [pc cm^{-3} h^{-2}]
A = (3 * H0**2 * cosmo['Omega_b'] * chi_H) / (8 * np.pi * G * mp)

#%%

aa = 1/(1+zz)
chis = ccl.comoving_radial_distance(cosmo, aa)
dchi_dz = np.gradient(chis, zz)
dz_dchi = 1/dchi_dz

#%%

from scipy.integrate import cumulative_trapezoid

# Compute cumulative integral of nz from high z to low z
nz_integrated = cumulative_trapezoid(nz[::-1], zz[::-1], initial=0)[::-1]

#%%

F_of_z = 0.9
W_chi = A * F_of_z * (1+zz) * nz_integrated * dz_dchi

t_frb = ccl.Tracer()
t_frb.add_tracer(cosmo, kernel=(chis, W_chi))

t_wl0 = ccl.WeakLensingTracer(cosmo, dndz=(z, dndz[0,:]))
t_wl1 = ccl.WeakLensingTracer(cosmo, dndz=(z, dndz[1,:]))
t_wl2 = ccl.WeakLensingTracer(cosmo, dndz=(z, dndz[2,:]))
t_wl3 = ccl.WeakLensingTracer(cosmo, dndz=(z, dndz[3,:]))
t_wl4 = ccl.WeakLensingTracer(cosmo, dndz=(z, dndz[4,:]))
t_wls = [t_wl0, t_wl1, t_wl2, t_wl3, t_wl4]

#%%

k_arr = np.logspace(-3, 1, 24)
lk_arr = np.log(k_arr)
a_arr = np.linspace(0.1, 1, 32)

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

#%%

#pk = cosmo.get_nonlin_power()
#pk_ee = pk_em = pk_mm = pk

ls = np.unique(np.geomspace(2, 2000, 256).astype(int)).astype(float)

# DM-DM auto-correlation
cls_frb = ccl.angular_cl(cosmo, t_frb, t_frb, ls, p_of_k_a=pk_ee)

# FRB shot noise
sigma_host_0 = 90 # cm^{-3} pc
integrand = nz * sigma_host_0**2 / (1+zz)**2
sigma_host_sq = np.trapz(integrand, zz)
n_per_deg_sq = 0.5 # number density per degree squared
n_per_sr = n_per_deg_sq * (180/np.pi)**2
nl_frb = np.ones(len(ls)) * sigma_host_sq / n_per_sr
cls_frb += nl_frb

# WL-DM cross-correlations
cls_x0 = ccl.angular_cl(cosmo, t_wl0, t_frb, ls, p_of_k_a=pk_em)
cls_x1 = ccl.angular_cl(cosmo, t_wl1, t_frb, ls, p_of_k_a=pk_em)
cls_x2 = ccl.angular_cl(cosmo, t_wl2, t_frb, ls, p_of_k_a=pk_em)
cls_x3 = ccl.angular_cl(cosmo, t_wl3, t_frb, ls, p_of_k_a=pk_em)
cls_x4 = ccl.angular_cl(cosmo, t_wl4, t_frb, ls, p_of_k_a=pk_em)
cls_x = [cls_x0, cls_x1, cls_x2, cls_x3, cls_x4]

# WL-WL auto-correlations
n_ell = len(ls)
n_bins = len(ndens)
cls_wl = np.zeros([n_bins, n_bins, n_ell])

for i in range(n_bins):
    n_i = np.ones(len(ls)) * 0.28**2 / (ndens[i] * (60 * 180 / np.pi)**2)
    wl_i = t_wls[i]
    for j in range(n_bins):
        wl_j = t_wls[j]
        cls_ij = ccl.angular_cl(cosmo, wl_i, wl_j, ls, p_of_k_a=pk_mm)
        if i==j: 
            cls_wl[i,j,:] = cls_ij + n_i
        else:
            cls_wl[i,j,:] = cls_ij
            
#%%

plt.rcParams.update({
    "mathtext.fontset": "stix",
    "font.family": "serif",
    "font.size": 16})

plt.figure(figsize=(8, 6))
plt.plot(ls, cls_frb, color='black', linewidth=2, label=r'$\mathcal{D}-\mathcal{D}$')
plt.xlabel(r'$\ell$', fontsize=24)
plt.ylabel(r'$C_{\ell}^{\mathcal{DD}} \;\; [\rm pc^2 \, cm^{-6}]$', fontsize=24)
plt.xlim(2, 2000)
plt.loglog()
plt.tick_params(which='both', top=True, right=True, direction='in', width=1, length=5)
#plt.savefig('dm_power_spectrum_test_plot.pdf', format="pdf", bbox_inches="tight")
plt.show()

#%%

plt.figure(figsize=(8, 6))
plt.plot(ls, -cls_x0, color='mediumblue', linewidth=2, label=r'$\gamma^{(0)}-\mathcal{D}$')
plt.plot(ls, -cls_x1, color='deepskyblue', linewidth=2, label=r'$\gamma^{(1)}-\mathcal{D}$')
plt.plot(ls, -cls_x2, color='blueviolet', linewidth=2, label=r'$\gamma^{(2)}-\mathcal{D}$')
plt.plot(ls, -cls_x3, color='hotpink', linewidth=2, label=r'$\gamma^{(3)}-\mathcal{D}$')
plt.plot(ls, -cls_x4, color='crimson', linewidth=2, label=r'$\gamma^{(4)}-\mathcal{D}$')
plt.xlabel(r'$\ell$', fontsize=24)
plt.ylabel(r'$C_{\ell}^{\gamma\mathcal{D}} \;\; [\rm pc \, cm^{-3}]$', fontsize=24)
plt.xlim(2, 2000)
plt.ylim(5e-9, 1e-3)
plt.loglog()
plt.tick_params(which='both', top=True, right=True, direction='in', width=1, length=5)
plt.legend(fontsize=20, frameon=False, ncol=2, loc="lower center")
#plt.savefig('dm_wl_power_spectrum_test_plot.pdf', format="pdf", bbox_inches="tight")
plt.show()

#%%

# Construct matrix of angular power spectra
n_fields = 1 + n_bins # 1 for FRB DM, n_bins for WL
cls_matrix = np.zeros((n_fields, n_fields, n_ell))

cls_matrix[0, 0, :] = cls_frb # (0,0): DM-DM

for i in range(n_bins):
    cls_matrix[0, i+1, :] = cls_x[i] # (0,i): DM-WL
    cls_matrix[i+1, 0, :] = cls_x[i] # (i,0): WL-DM
    
for i in range(n_bins):
    for j in range(n_bins):
        cls_matrix[i+1, j+1, :] = cls_wl[i, j, :] # (i,j): WL-WL 

#%%

tracers = [t_frb, t_wl0, t_wl1, t_wl2, t_wl3, t_wl4]
labels = ['dm_tracer', 'wl_0', 'wl_1', 'wl_2', 'wl_3', 'wl_4']
n_tracers = len(tracers)

# List of spectra indices
spec_indices = []
for i in range(n_tracers):
    for j in range(i, n_tracers):
        spec_indices.append((i, j))

n_specs = len(spec_indices)
dl = np.gradient(ls)
f_sky = 0.1 # note that the FRB sample has f_sky = 0.7 while our WL sample has f_sky = 0.1

# Construct the covariance matrix
cov = np.zeros((n_specs, n_ell, n_specs, n_ell))

for a, (i, j) in enumerate(spec_indices):
    for b, (k, l) in enumerate(spec_indices):
        C_ik = cls_matrix[i, k]
        C_jl = cls_matrix[j, l]
        C_il = cls_matrix[i, l]
        C_jk = cls_matrix[j, k]
        covar = (C_ik * C_jl + C_il * C_jk) / ((2 * ls + 1) * dl * f_sky)
        cov[a, :, b, :] = np.diag(covar)

covar = cov.reshape(n_specs * n_ell, n_specs * n_ell)

#%%

# Create SACC object
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
for idx, (i, j) in enumerate(spec_indices):
    tracer_i = labels[i]
    tracer_j = labels[j]
    cl = cls_matrix[i, j, :]
    
    if tracer_i == tracer_j:
        if tracer_i == 'dm_tracer':
            cl -= nl_frb
        elif 'wl_' in tracer_i:
            wl_index = int(tracer_i.split('_')[1])
            n_i = np.ones(len(ls)) * 0.28**2 / (ndens[wl_index] * (60 * 180 / np.pi)**2)
            cl -= n_i
        
    s.add_ell_cl('cl_eb', tracer_i, tracer_j, ls, cl)

# Add covariance
s.add_covariance(covar)

# Write to SACC file
s.save_fits("wl-frb.fits", overwrite=True)

#%%

# Calculate S/N for DM-DM
cl_frb = cls_matrix[0, 0, :]
var_00 = np.diagonal(cov[0, :, 0, :])
sn_frb = np.sqrt(np.sum(cl_frb**2 / var_00))
print(sn_frb)

# Calculate S/N for WL-DM
var_x0 = np.diagonal(cov[1, :, 1, :])
sn_x0 = np.sqrt(np.sum(cls_x0**2 / var_x0))
print(sn_x0)
