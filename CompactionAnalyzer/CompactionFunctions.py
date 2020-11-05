import numpy as np
from skimage import io
from PIL import Image
from skimage.filters import gaussian, threshold_otsu, threshold_yen
import scipy.ndimage.morphology as scipy_morph
import scipy.ndimage.measurements as scipy_meas
from skimage.morphology import remove_small_objects
import imageio
from skimage import color
from scipy.ndimage.morphology import distance_transform_edt
import copy
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import glob as glob
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

def max_projection(stack):
    max_array = (normalize(np.max(stack,axis=2), lb=0, ub=100)*255).astype("uint8")
    im =Image.fromarray(max_array, mode= "L")   
    return im


def custom_mask(img,show_segmentation=True):
    """
    Image segmentation function to create a custom polygon mask, and evalute radius and position of the masked object.
    Need to use %matplotlib qt in jupyter notebook
    Args:
        img(array): Grayscale image as a Numpy array
    Returns:
        dict: Dictionary with keys: mask, radius, centroid (x/y)
    """
    from roipoly import RoiPoly
    # need to install RoiPoly package via pip to use this function  
    height = img.shape[0]
    width  = img.shape[1]
    # click polygon mask interactive
    plt.ion()
    plt.imshow(img, extent=[0, width, height, 0])
    plt.text(0.5, 1.05,'Click Polygon Mask with left click, finish with right click',  fontsize=12,
         horizontalalignment='center',
         verticalalignment='center',c='darkred', transform= plt.gca().transAxes)#     transform = ax.transAxes)
    my_roi = RoiPoly(color='r')
    # Extract mask and segementation details
    #mask = np.flipud(my_roi.get_mask(img))   # flip mask due to imshow 
    mask = (my_roi.get_mask(img))
    # determine radius of spheroid
    radius = np.sqrt(np.sum(mask) / np.pi)
    # determine center of mass
    cy, cx = scipy_meas.center_of_mass(mask)
    # hide pop-up windows   
    plt.ioff()
    # return dictionary containing mask information
    # # show segmentation
    # if show_segmentation:
    #     plt.figure()
    #     plt.subplot(121), plt.imshow(img)
    #     plt.subplot(122), plt.imshow(mask)    
    return {'mask': mask, 'radius': radius, 'centroid': (cx, cy)} 




def segment_cell(img, thres=1, gaus1 = 8, gaus2=80, iterartions=1,show_segmentation = False, segmention="otsu"):   
    """
    Image segmentation function to create  mask, radius, and position of a spheroid in a grayscale image.
    Args:
        img(array): Grayscale image as a Numpy array
        thres(float): To adjust the segmentation, keep 1
        segmention: use "otsu" or "yen"  as segmentation method
        iterations: iterations of closing steps , might increase to get more 
        robust segmentation but less precise segmentation, by default 1
    Returns:
        dict: Dictionary with keys: mask, radius, centroid (x/y)
    """
    height = img.shape[0]
    width = img.shape[1]
    # local gaussian   
    img = np.abs(gaussian(img, sigma=gaus1) - gaussian(img, sigma=gaus2))
    # segment cell
    if segmention == "yen":
        mask = img > threshold_yen(img) * thres
    if segmention == "otsu":
        mask = img > threshold_otsu(img) * thres  
    
    # remove other objects
    
    mask = scipy_morph.binary_closing(mask, iterations=iterartions)
    mask = remove_small_objects(mask, min_size=1000)
    mask = scipy_morph.binary_dilation(mask, iterations=iterartions)
    mask = scipy_morph.binary_fill_holes(mask)
    

    # identify spheroid as the most centered object
    labeled_mask, max_lbl = scipy_meas.label(mask)
    center_of_mass = np.array(scipy_meas.center_of_mass(mask, labeled_mask, range(1, max_lbl + 1)))
    distance_to_center = np.sqrt(np.sum((center_of_mass - np.array([height / 2, width / 2])) ** 2, axis=1))

    mask = (labeled_mask == distance_to_center.argmin() + 1)

    # show segmentation
    if show_segmentation:
        plt.figure()
        plt.subplot(121), plt.imshow(img)
        plt.subplot(122), plt.imshow(mask)
    
    # determine radius of spheroid
    radius = np.sqrt(np.sum(mask) / np.pi)

    # determine center position of spheroid
    cy, cx = center_of_mass[distance_to_center.argmin()]

    # return dictionary containing spheroid information
    return {'mask': mask, 'radius': radius, 'centroid': (cx, cy)}




# def StuctureAnalysisMain():
    
#     return 







