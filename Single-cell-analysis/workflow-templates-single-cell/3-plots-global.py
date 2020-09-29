
import matplotlib.pyplot as plt
import numpy as np
import glob as glob



"""
Analyze data of all cells 
"""


# read in the data from different wells
b02 = [np.loadtxt(i) for i in glob.glob(r"eval//*B02_G*\cos2a_mean_we_coh.txt")]
b03 = [np.loadtxt(i) for i in glob.glob(r"eval//*B03_G*\cos2a_mean_we_coh.txt")]
c02 = [np.loadtxt(i) for i in glob.glob(r"eval//*C02_G*\cos2a_mean_we_coh.txt")]
c03 = [np.loadtxt(i) for i in glob.glob(r"eval//*C03_G*\cos2a_mean_we_coh.txt")]

# combine data from wells according to layout
dmso = np.concatenate([b02,b03])
rock = np.concatenate([c02,c03])

# get mean data
lengths = [len(dmso),len(rock)]
conditions = ["DMSO (n={})".format(lengths[0]), "Rock. Inh. (10 uM, n={})".format(lengths[1])]   
            
means = [np.nanmean(dmso),np.nanmean(rock)]
std = [np.nanstd(dmso),np.nanstd(rock)]
err = np.array(std)/np.sqrt(lengths)


# create plots and save results
# mean angle deviation
plt.figure()
plt.bar([1,2], means, yerr=err,
        color="lightgreen") 
# horizontal line indicating random
plt.xticks([1,2],conditions, fontsize = 11)
plt.yticks(fontsize = 14)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.ylabel("Orientation", fontsize = 12)
plt.legend()
plt.tight_layout()
plt.savefig(r"Global_orientation_rock_vs_dmso",dpi=500)
plt.show()



# mean intensities in first 2 shells
# read in the data
shell = 2  # equals 10 um 

b02 = np.array([np.nanmean(np.loadtxt(i)[:shell]) for i in glob.glob(r"eval//*B02_G*\\dist_int_individ_norm.txt")]) 
b03 =  np.array([np.nanmean(np.loadtxt(i)[:shell]) for i in glob.glob(r"eval//*B03_G*\\dist_int_individ_norm.txt")]) 
c02 =  np.array([np.nanmean(np.loadtxt(i)[:shell]) for i in glob.glob(r"eval//*C02_G*\\dist_int_individ_norm.txt")]) 
c03 =  np.array([np.nanmean(np.loadtxt(i)[:shell]) for i in glob.glob(r"eval//*C03_G*\\dist_int_individ_norm.txt")]) 

dmso = np.concatenate([b02,b03])
rock = np.concatenate([c02,c03])
# get mean data
lengths = [len(dmso),len(rock)]
conditions = ["DMSO (n={})".format(lengths[0]), "Rock. Inh. (10 uM, n={})".format(lengths[1])]          
means = [np.nanmean(dmso),np.nanmean(rock)]
std = [np.nanstd(dmso),np.nanstd(rock)]
err = np.array(std)/np.sqrt(lengths)

# create plots and save results
# mean angle deviation
plt.figure()
plt.bar([1,2], means, yerr=err,
        color="lightgreen") #, label="day 4 (some hours after treatment)") 
# horizontal line indicating random
plt.xticks([1,2],conditions, fontsize = 11)
plt.ylim(0.7,1.3)
plt.yticks(fontsize = 14)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.ylabel("Intensity", fontsize = 12)
plt.legend()
plt.tight_layout()
plt.savefig(r"Intensity_first10um",dpi=500)
plt.show()

