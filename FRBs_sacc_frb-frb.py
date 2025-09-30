import numpy as np
import pyccl as ccl
import sacc
import HaloProfiles as hp
import matplotlib.pyplot as plt

#%%

class HaloProfileBaryons(ccl.halos.HaloProfile):
    def __init__(self, profile1, profile2, weight1, weight2, cosmo, hmc):
        super().__init__(mass_def=profile1.mass_def)
        self.profile1 = profile1
        self.profile2 = profile2
        self.w1 = weight1
        self.w2 = weight2

        a = 1/(1+0.55)
        norm_p1 = profile1.get_normalization(cosmo, a, hmc=hmc)
        norm_p2 = profile2.get_normalization(cosmo, a, hmc=hmc)
        self.norm = 1.0 / (self.w1 * norm_p1 + self.w2 * norm_p2)
 
    def _real(self, cosmo, a, M, r):
        p1 = self.profile1.real(cosmo, a, M, r)
        p2 = self.profile2.real(cosmo, a, M, r)
        return self.norm * (self.w1 * p1 + self.w2 * p2)

    def _fourier(self, cosmo, a, M, k):
        p1 = self.profile1.fourier(cosmo, a, M, k)
        p2 = self.profile2.fourier(cosmo, a, M, k)
        return self.norm * (self.w1 * p1 + self.w2 * p2)

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

# [W_{\chi}] = [A] = [cm^{-3}]
# Factor of 1e6 so that Cl is in units of [pc cm^{-3}]
h = cosmo['H0'] / 100
chis = ccl.comoving_radial_distance(cosmo, aa[::-1])[::-1]
W_chi = A * (1+zz) * nz_integrated * 1e6

t_frb = ccl.Tracer()
t_frb.add_tracer(cosmo, kernel=(chis, W_chi))

#%%

k_arr = np.logspace(-3, 3, 48)
lk_arr = np.log(k_arr)

# Halo model implementation
hmd_200c = ccl.halos.MassDef200c
cM = ccl.halos.ConcentrationDuffy08(mass_def=hmd_200c)
nM = ccl.halos.MassFuncTinker08(mass_def=hmd_200c)
bM = ccl.halos.HaloBiasTinker10(mass_def=hmd_200c)
pM = ccl.halos.HaloProfileNFW(mass_def=hmd_200c, concentration=cM)
pE = hp.HaloProfileDensityHE(mass_def=hmd_200c, concentration=cM, lMc=14.0, beta=0.6, A_star=0.03, eta_b=0.5)
hmc = ccl.halos.HMCalculator(mass_function=nM, halo_bias=bM, mass_def=hmd_200c, log10M_max=15.0, log10M_min=10.0, nM=32)

#%%

# Add baryonic effects into the matter power spectrum
f_bar = cosmo['Omega_b'] / cosmo['Omega_m']
f_cdm = cosmo['Omega_c'] / cosmo['Omega_m']
f_star = 0.03
pM_bar = HaloProfileBaryons(pM, pE, f_cdm, f_bar-f_star, cosmo=cosmo, hmc=hmc)

#%%

# m-m, e-e, and e-m power spectra
pk_mm_nfw = ccl.halos.halomod_Pk2D(cosmo, hmc, pM, lk_arr=lk_arr, a_arr=aa[::-1])
pk_mm = ccl.halos.halomod_Pk2D(cosmo, hmc, pM_bar, lk_arr=lk_arr, a_arr=aa[::-1])
pk_ee = ccl.halos.halomod_Pk2D(cosmo, hmc, pE, prof2=pE, lk_arr=lk_arr, a_arr=aa[::-1])
pk_em = ccl.halos.halomod_Pk2D(cosmo, hmc, pE, prof2=pM_bar, lk_arr=lk_arr, a_arr=aa[::-1])

#%%

a = 1/(1+0.55)

plt.rcParams.update({
    "mathtext.fontset": "stix",
    "font.family": "serif",
    "font.size": 16})

plt.figure(figsize=(8, 6))
plt.plot(k_arr, pk_mm_nfw(k_arr, a), label=r'$P_{\rm NFW}(k)$', color='crimson')
plt.plot(k_arr, pk_mm(k_arr, a), label=r'$P_{\rm bar}(k)$', color='mediumblue')
plt.plot(k_arr, pk_ee(k_arr, a), label=r'$P_{\rm ee}(k)$', color='deepskyblue')
plt.plot(k_arr, pk_em(k_arr, a), label=r'$P_{\rm em}(k)$', color='blueviolet')
plt.loglog()
plt.xlabel(r'$k$', fontsize=28)
plt.ylabel(r'$P(k)$', fontsize=28)
plt.xlim(1e-3, 1e3)
plt.tick_params(which='both', top=True, right=True, direction='in', width=1, length=5)
plt.legend(fontsize=20, frameon=False, ncol=2, loc="lower center")
plt.show()

#%%

ls = np.unique(np.geomspace(1, 500, 256).astype(int)).astype(float)

# DM-DM auto-correlation
cls_frb = ccl.angular_cl(cosmo, t_frb, t_frb, ls, p_of_k_a=pk_ee)

# FRB shot noise
sigma_host_0 = 90 # pc cm^{-3}
integrand = nz * sigma_host_0**2 / (1+zz)**2
sigma_host_sq = np.trapz(integrand, zz)
n_per_deg_sq = 0.5 # number density per degree squared
n_per_sr = n_per_deg_sq * (180/np.pi)**2
nl_frb = np.ones(len(ls)) * sigma_host_sq / n_per_sr

#%%

dl = np.gradient(ls)
f_sky = 0.7

# Construct the covariance matrix
cov = (2*(cls_frb+nl_frb)**2)/((2*ls+1)*dl*f_sky)

# Calculate S/N
sn_frb = np.sqrt(np.sum(cls_frb**2/cov))
print(f"S/N (FRB x FRB): {sn_frb:.1f}")

#%%

# Create SACC object
s = sacc.Sacc()

# Add FRB dispersion measure tracer
s.add_tracer('NZ', 'frb',
             quantity='dm_correlation',
             spin=0,
             z=zz,
             nz=nz)

# Add angular power spectrum
s.add_ell_cl('cl_00', 'frb', 'frb', ls, cls_frb)

# Add covariance
s.add_covariance(cov)

# Write to SACC file
s.save_fits("frb-frb.fits", overwrite=True)
