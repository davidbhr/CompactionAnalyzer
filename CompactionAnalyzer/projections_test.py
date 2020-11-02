# import numpy as np
# import matplotlib.pyplot as plt
# import imageio 
import glob as glob
from skimage import io
import os
# from PIL import Image
# import matplotlib.image as mpimg
from tqdm import tqdm
from natsort import natsorted
from CompactionAnalyzer.CompactionFunctions import normalize,max_projection
from CompactionAnalyzer.utilities import load_stack




# # read image stack
# img_list = natsorted(glob.glob(r"..\\TestData\\*.tif"))[:1]   # total 80 um step 2.5 um   now 30um stack
# stack = load_stack(img_list)
# #maxproj 
# maxproj = max_projection(stack)
# #create outfolder HERE AS SUBFOLDER OF CURRENT SCRIPT
# outfolder = "Max_all//"+f.split(os.sep)[1]+"//"+f.split(os.sep)[2] #+ "//"+ f.split(os.sep)[2]   # cut-7to7
# if not os.path.exists(outfolder):
#     os.makedirs(outfolder)
# maxproj.save(outfolder+ "//" + channel + ".tif" )








