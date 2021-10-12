import shtns 
import numpy as np
import pencil as pc
import argparse
import h5py

parser = argparse.ArgumentParser(description='magnetic helicity and energy spectra')
parser.add_argument('--onlyenergy',type =bool, default=False, help='set to True for computing only energy spectra')
args = parser.parse_args()
onlyenergy = args.onlyenergy


# bbr = pc.read.slices(extension='yz',field='bb1').yz.bb1	#read in the r,theta, and phi components of the magnetic field 
# bbt = pc.read.slices(extension='yz',field='bb2').yz.bb2	#over a theta and phi surface
# bbp = pc.read.slices(extension='yz',field='bb3').yz.bb3
var = h5py.File('ameya1_w_sc_prof.hdf5','r')

bbr = np.array(var['mag']['rad'])
bbt = np.array(var['mag']['the'])
bbp = np.array(var['mag']['phi'])
time,nlat,nphi = bbr.shape


lmax = 127												#lmax decides mmax and nphi>2*mmax, where nphi is the 
                                                        #number of points in the phi/azimuthal direction. Therefore choose accordingly.

en_c = np.zeros((time,lmax))
hel_c = np.zeros((time,lmax),dtype=complex)

for it in range(time):
	sh = shtns.sht(lmax=lmax,norm=shtns.sht_orthonormal|shtns.SHT_NO_CS_PHASE) #creating the sht object, orthonormalised, w/o the (-1)^m factor
	nlat,nphi = sh.set_grid(nlat=nlat,nphi=nphi) 							   #IMPORTANT!!
	qlm,slm,tlm = sh.analys(bbr[it].T,bbt[it].T,bbp[it].T) 					   #vec-harmonics expansion, transposing is only necessary to have the [theta,phi] 
	re_qlm = np.zeros((sh.lmax+1,sh.mmax+1),dtype=complex)					   #shape arrays for expansion coefficients, indexed with [l,m]
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
		qlm_pr,slm_pr,tlm_pr = sh.analys(bbp[it].T,bbz.T,bbr[it].T) 


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
		qlm_tr,slm_tr,tlm_tr = sh.analys(bbt[it].T,bbr[it].T,bbz.T) 


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
			hel_cc[l-1] += tmp2
	en_c[it] = en_cc.copy()
	hel_c[it] = hel_cc.copy()

degree = np.arange(1,lmax+1)
t_begin = 10														#time index from which to average the spectra																			
een_c = np.mean(en_c[t_begin::],axis=0)									#preparing the energy and helicity spectra to plot it on
if not(onlyenergy):													#a log-log scale
	hhel_c = np.mean(np.real(hel_c[t_begin::,:]),axis=0)
	red_h = hhel_c.copy() 
	blue_h = hhel_c.copy()
	blue_h[blue_h>0] = 0 
	red_h[red_h<0] = 0  

plt.ion()
plt.figure()
plt.loglog(degree,een_c,'g')
scl = np.arange(1,lmax+1,dtype=float)
scl +=0.5

if not(onlyenergy):
	plt.loglog(degree,-1*blue_h*scl,'ob') 
	plt.loglog(degree,red_h*scl,'xr') 
	plt.loglog(degree,np.abs(hhel_c.real)*scl,'--k')
plt.show()


