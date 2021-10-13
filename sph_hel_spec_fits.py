import shtns 
import numpy as np
from astropy.io import fits
import argparse
import h5py
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description='magnetic helicity and energy spectra')
parser.add_argument('--onlyenergy',type =bool, default=False, help='set to True for computing only energy spectra')
parser.add_argument('--savespectra',type =bool, default=False, help='set to True for saving the spectra in a hdf5 file')
args = parser.parse_args()
onlyenergy = args.onlyenergy
savespectra = args.savespectra

filename = ''
mag_data = fits.open(filename)[0].data.copy()		#read in the r,theta, and phi components of the magnetic field over a theta and phi surface
													#this assumes the data is stored in the PrimaryHDU, please change if not

bbr = mag_data[0].copy()    
bbt = mag_data[1].copy()	 
bbp = mag_data[2].copy()

time,nlat,nphi = bbr.shape


lmax = 127											#lmax decides mmax and nphi>2*mmax, where nphi is the 
                                                    #number of points in the phi/azimuthal direction. Therefore choose accordingly.

en_c = np.zeros((time,lmax))
hel_c = np.zeros((time,lmax),dtype=complex)

for it in range(time):
	sh = shtns.sht(lmax=lmax,norm=shtns.sht_orthonormal|shtns.SHT_NO_CS_PHASE) #creating the sht object, orthonormalised, w/o the (-1)^m factor
	nlat,nphi = sh.set_grid(nlat=nlat,nphi=nphi) 							   
	qlm,slm,tlm = sh.analys(bbr[it],bbt[it],bbp[it]) 					   	   #vec-harmonics expansion
	re_qlm = np.zeros((sh.lmax+1,sh.mmax+1),dtype=complex)					   
	re_slm = np.zeros((sh.lmax+1,sh.mmax+1),dtype=complex)
	re_tlm = np.zeros((sh.lmax+1,sh.mmax+1),dtype=complex) 

	for l in range(sh.lmax+1): 												   #rearranging the arrays in a more intuitive manner
		for m in range(l+1): 
			re_qlm[l,m] = qlm[sh.idx(l,m)]
			re_tlm[l,m] = tlm[sh.idx(l,m)] 
			re_slm[l,m] = slm[sh.idx(l,m)] 

	if not(onlyenergy):														   #only relevant for computing magnetic helicity
		del sh
		sh = shtns.sht(lmax=lmax,norm=shtns.sht_orthonormal|shtns.SHT_NO_CS_PHASE) 
		nlat,nphi = sh.set_grid(nlat=nlat,nphi=nphi) 

		bbz = np.zeros(bbr[it].shape)
		qlm_pr,slm_pr,tlm_pr = sh.analys(bbp[it],bbz,bbr[it]) 


		re_qlm_pr = np.zeros((sh.lmax+1,sh.mmax+1),dtype=complex)
		re_slm_pr = np.zeros((sh.lmax+1,sh.mmax+1),dtype=complex)
		re_tlm_pr= np.zeros((sh.lmax+1,sh.mmax+1),dtype=complex) 

		for l in range(sh.lmax+1): 
			for m in range(l+1): 
				re_qlm_pr[l,m] = qlm_pr[sh.idx(l,m)]
				re_tlm_pr[l,m] = tlm_pr[sh.idx(l,m)] 
				re_slm_pr[l,m] = slm_pr[sh.idx(l,m)] 

		del sh, bbz
		sh = shtns.sht(lmax=lmax,norm=shtns.sht_orthonormal|shtns.SHT_NO_CS_PHASE) 
		nlat,nphi = sh.set_grid(nlat=nlat,nphi=nphi) 

		bbz = np.zeros(bbr[it].shape)
		qlm_tr,slm_tr,tlm_tr = sh.analys(bbt[it],bbr[it],bbz) 


		re_qlm_tr = np.zeros((sh.lmax+1,sh.mmax+1),dtype=complex)
		re_slm_tr = np.zeros((sh.lmax+1,sh.mmax+1),dtype=complex)
		re_tlm_tr= np.zeros((sh.lmax+1,sh.mmax+1),dtype=complex) 

		for l in range(sh.lmax+1): 
			for m in range(l+1): 
				re_qlm_tr[l,m] = qlm_tr[sh.idx(l,m)]
				re_tlm_tr[l,m] = tlm_tr[sh.idx(l,m)] 
				re_slm_tr[l,m] = slm_tr[sh.idx(l,m)] 


	en_cc = np.zeros(sh.lmax)
	hel_cc = np.zeros(sh.lmax,dtype=complex)


	for l in range(1,sh.lmax): 
		for m in range(l+1):
			if m==0:
				cm = 1
			else:
				cm = 2
			tmp1 = cm*(np.abs(re_qlm[l,m])**2 + l*(l+1)*(np.abs(re_tlm[l,m])**2 + np.abs(re_slm[l,m])**2))
			if not(onlyenergy):
				tmp2 = cm*((re_qlm[l,m])*np.conj(re_tlm[l+1,m])-(re_qlm_pr[l,m])*np.conj(re_tlm_pr[l+1,m])-(re_qlm_tr[l,m])*np.conj(re_tlm_tr[l+1,m]))
			en_cc[l-1] += tmp1
			if not(onlyenergy):
				hel_cc[l-1] += tmp2
	en_c[it] = en_cc.copy()
	hel_c[it] = hel_cc.copy()


degree = np.arange(1,lmax+1)
t_begin = 10															#time index from which to average the spectra															
een_c = np.mean(en_c[t_begin::],axis=0)									#preparing the energy and helicity spectra to plot it on
if not(onlyenergy):													    #a log-log scale
	hhel_c = np.mean(np.real(hel_c[t_begin::,:]),axis=0)
	red_h = hhel_c.copy() 
	blue_h = hhel_c.copy()
	blue_h[blue_h>0] = 0 
	red_h[red_h<0] = 0  

plt.ion()
plt.figure()
plt.loglog(degree[0:-1],een_c[0:-1],'g')
scl = np.arange(1,lmax+1,dtype=float)
scl +=0.5

if not(onlyenergy):
	plt.loglog(degree[0:-1],-1*blue_h[0:-1]*scl[0:-1],'ob') 
	plt.loglog(degree[0:-1],red_h[0:-1]*scl[0:-1],'xr') 
	plt.loglog(degree[0:-1],np.abs(hhel_c.real[0:-1])*scl[0:-1],'--k')
plt.show()

if savespectra:
	file = h5py.File('spectra.hdf5','w')
	gp1 = file.create_group('spectra')
	gp1.create_dataset('magenergy',data = en_c)	
	gp1.create_dataset('degree', data = degree)
	if not(onlyenergy):
		gp1 = file.create_dataset('maghelicity',data = hel_c)
	file.close()


