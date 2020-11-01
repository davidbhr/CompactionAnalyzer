import numpy as np
import matplotlib.pyplot as plt
import imageio 
import glob as glob
from skimage import io
import os
from PIL import Image
import matplotlib.image as mpimg
from tqdm import tqdm
from natsort import natsorted

def load_stack(stack_list):
    z = len(stack_list)
    img_example = io.imread(stack_list[0], as_gray=True)
    h,w = np.shape(img_example)
    tiffarray=np.zeros((h,w,z))
    for n,img in enumerate(stack_list):
        tiffarray[:,:,n]=  (io.imread(stack_list[n], as_gray=True))  #normalize
    return tiffarray    
        
        

def normalize(image, lb=0.1, ub=99.9):
    '''
    normalizes image to  a range from 0 and 1 and cuts of extreme values
    e.g. lower then 0.1 percentile and higher then 99.9m percentile

    :param image:
    :param lb: percentile of lower bound for filter
    :param ub: percentile of upper bound for filter
    :return:
    '''

    image = image - np.percentile(image, lb)  # 1 Percentile
    image = image / np.percentile(image, ub)  # norm to 99 Percentile
    image[image < 0] = 0.0
    image[image > 1] = 1.0
    return image



# list of folders for individual cells from which we create the max intensity projections
folders = glob.glob(r"..//data//Day*//*frames")  

#loop through these folders
for f in tqdm(folders):
            # read in the existing channels individually 
            for channel in  ["C003","C004","C001"]:       
                    # read image stack
                    img_list = natsorted(glob.glob(f+"\\*"+channel+"*"))   # total 80 um step 2.5 um   now 30um stack
                    stack = load_stack(img_list)
                    #maxproj and normalizing   convert to uint8
                    maxproj = ((normalize(np.max(stack,axis=2), lb=0, ub=100)*255)).astype("uint8")  #normalize()   normalize
                    # convert array to image
                    im_max= Image.fromarray(maxproj, mode= "L")  
                
                    #create outfolder HERE AS SUBFOLDER OF CURRENT SCRIPT
                    outfolder = "Max_all//"+f.split(os.sep)[1]+"//"+f.split(os.sep)[2] #+ "//"+ f.split(os.sep)[2]   # cut-7to7
                    if not os.path.exists(outfolder):
                        os.makedirs(outfolder)
                    im_max.save(outfolder+ "//" + channel + ".tif" )








