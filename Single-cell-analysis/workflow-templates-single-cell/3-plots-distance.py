"""

@author: david boehringer 

"""

import numpy as np
import matplotlib.pyplot as plt
import glob as glob
from scipy import stats

### read in distance shell data for orientation and intensity in matrix
# DMSO
shells_mid_dmso =  np.concatenate([[np.loadtxt(i) for i in glob.glob(r"eval//*B02_G*\\shells-mid_px.txt")],
                             [np.loadtxt(i) for i in glob.glob(r"eval//*B03_G*\\shells-mid_px.txt")]])
ang_ind_dmso =   np.concatenate([[np.loadtxt(i) for i in glob.glob(r"eval//*B02_G*\\dist_angle_individ.txt")],
                             [np.loadtxt(i) for i in glob.glob(r"eval//*B03_G*\\dist_angle_individ.txt")]])
int_ind_dmso =  np.concatenate([[np.loadtxt(i) for i in glob.glob(r"eval//*B02_G*\\dist_int_individ_norm.txt")],
                             [np.loadtxt(i) for i in glob.glob(r"eval//*B03_G*\\dist_int_individ_norm.txt")]])

# find max
max_shells_dmso = np.max([shells_mid_dmso[i].shape[0] for i in range(shells_mid_dmso.shape[0])])
#make matrix for mean value 
# find max
max_shells_dmso = np.max([shells_mid_dmso[i].shape[0] for i in range(shells_mid_dmso.shape[0])])
# empty matrix
matrix_ang_ind_dmso = np.zeros([max_shells_dmso,shells_mid_dmso.shape[0]])
matrix_int_ind_dmso = np.zeros([max_shells_dmso,shells_mid_dmso.shape[0]])
# fill matrix
for cell in range(shells_mid_dmso.shape[0]):
    matrix_ang_ind_dmso[:len(ang_ind_dmso[cell]), cell] = ang_ind_dmso[cell]
 
    matrix_int_ind_dmso[:len(ang_ind_dmso[cell]), cell] = int_ind_dmso[cell]
# set no data (0 value) to nan
matrix_ang_ind_dmso[matrix_ang_ind_dmso==0] = np.nan
matrix_int_ind_dmso[matrix_int_ind_dmso==0] = np.nan



# ROCK Inhibitr data
shells_mid_rock =  np.concatenate([[np.loadtxt(i) for i in glob.glob(r"eval//*C02_G*\\shells-mid_px.txt")],
                             [np.loadtxt(i) for i in glob.glob(r"eval//*C03_G*\\shells-mid_px.txt")]])
ang_ind_rock =   np.concatenate([[np.loadtxt(i) for i in glob.glob(r"eval//*C02_G*\\dist_angle_individ.txt")],
                             [np.loadtxt(i) for i in glob.glob(r"eval//*C03_G*\\dist_angle_individ.txt")]])
int_ind_rock =  np.concatenate([[np.loadtxt(i) for i in glob.glob(r"eval//*C02_G*\\dist_int_individ_norm.txt")],
                             [np.loadtxt(i) for i in glob.glob(r"eval//*C03_G*\\dist_int_individ_norm.txt")]])
# find max
max_shells_rock = np.max([shells_mid_rock[i].shape[0] for i in range(shells_mid_rock.shape[0])])
#make matrix for mean value 
# find max
max_shells_rock = np.max([shells_mid_rock[i].shape[0] for i in range(shells_mid_rock.shape[0])])
# emptz matrix
matrix_ang_ind_rock = np.zeros([max_shells_rock,shells_mid_rock.shape[0]])
matrix_int_ind_rock = np.zeros([max_shells_rock,shells_mid_rock.shape[0]])
# fill matrix
for cell in range(shells_mid_rock.shape[0]):
    matrix_ang_ind_rock[:len(ang_ind_rock[cell]), cell] = ang_ind_rock[cell]
    matrix_int_ind_rock[:len(ang_ind_rock[cell]), cell] = int_ind_rock[cell]
# set no data (0 value) to nan
matrix_ang_ind_rock[matrix_ang_ind_rock==0] = np.nan
matrix_int_ind_rock[matrix_int_ind_rock==0] = np.nan




"""
plots
"""


plt.style.use("seaborn-colorblind")
plt.figure(figsize=(6,4))
# plot means
plt.plot(np.arange(0,max_shells_dmso)*5, np.nanmean(matrix_ang_ind_dmso,axis=1), "o-", color="C0", label="DMSO (n={})".format(str(ang_ind_dmso.shape[0])))
plt.plot(np.arange(0,max_shells_rock)*5, np.nanmean(matrix_ang_ind_rock,axis=1), "o-", color="C1", label="ROCK (n={})".format(str(ang_ind_rock.shape[0])))
# plot +- std error
plt.fill_between(np.arange(0,max_shells_dmso)*5, np.nanmean(matrix_ang_ind_dmso,axis=1)-stats.sem(matrix_ang_ind_dmso, nan_policy = "omit",axis=1),
                                                 np.nanmean(matrix_ang_ind_dmso,axis=1)+stats.sem(matrix_ang_ind_dmso,nan_policy = "omit",axis=1),"o-",color="C0", alpha=0.1)
plt.fill_between(np.arange(0,max_shells_rock)*5, np.nanmean(matrix_ang_ind_rock,axis=1)-stats.sem(matrix_ang_ind_rock, nan_policy = "omit",axis=1),
                                                 np.nanmean(matrix_ang_ind_rock,axis=1)+stats.sem(matrix_ang_ind_rock,nan_policy = "omit",axis=1),"o-",color="C1", alpha=0.1)
plt.legend()

plt.grid()
plt.xlabel("Distance (µm)", fontsize=12)
plt.ylabel("Angle (‎°)", fontsize=12)
plt.tight_layout()
plt.savefig(r"angle_individ_rock_dmso.png", dpi=500)



plt.figure(figsize=(6,4))
# plot orientation over distance
orienation_dmso = np.cos(2*np.nanmean(matrix_ang_ind_dmso,axis=1)*np.pi/180)    
orienation_rock = np.cos(2*np.nanmean(matrix_ang_ind_dmso,axis=1)*np.pi/180) 
plt.plot(np.arange(0,max_shells_dmso)*5,orienation_dmso, "o-",color="C0", label="DMSO (n={})".format(str(ang_ind_dmso.shape[0])))
plt.plot(np.arange(0,max_shells_rock)*5,orienation_rock, "o-",color="C1", label="ROCK (n={})".format(str(ang_ind_rock.shape[0])))
plt.grid()
plt.legend()
plt.xlabel("Distance (µm)", fontsize=12)
plt.ylabel("Orientation (‎a.u)", fontsize=12)
plt.tight_layout()
plt.savefig(r"orientation.png", dpi=500)



plt.figure(figsize=(6,4))
plt.plot(np.arange(0,max_shells_dmso)*5, np.nanmean(matrix_int_ind_dmso,axis=1), "o-", color="C0",label="DMSO (n={})".format(str(ang_ind_dmso.shape[0])))
plt.plot(np.arange(0,max_shells_rock)*5, np.nanmean(matrix_int_ind_rock,axis=1), "o-", color="C1",label="ROCK (n={})".format(str(ang_ind_rock.shape[0])))
# plot +- std error
plt.fill_between(np.arange(0,max_shells_dmso)*5, np.nanmean(matrix_int_ind_dmso,axis=1)-stats.sem(matrix_int_ind_dmso, nan_policy = "omit",axis=1),
                                                 np.nanmean(matrix_int_ind_dmso,axis=1)+stats.sem(matrix_int_ind_dmso,nan_policy = "omit",axis=1),"o-",color="C0", alpha=0.1)
plt.fill_between(np.arange(0,max_shells_rock)*5, np.nanmean(matrix_int_ind_rock,axis=1)-stats.sem(matrix_int_ind_rock, nan_policy = "omit",axis=1),
                                                 np.nanmean(matrix_int_ind_rock,axis=1)+stats.sem(matrix_int_ind_rock,nan_policy = "omit",axis=1),"o-",color="C1", alpha=0.1)


plt.xlim(0,125)
# plt.ylim(0,1)
plt.show()
plt.grid()
plt.legend()
plt.xlabel("Distance (µm)", fontsize=12)
plt.ylabel("Intensity (a.u)", fontsize=12)
plt.tight_layout()
plt.savefig(r"Intensity.png", dpi=500)

