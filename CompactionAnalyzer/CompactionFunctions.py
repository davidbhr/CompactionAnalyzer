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
import matplotlib.pyplot as plt
import os
import pandas as pd
import glob as glob
from tqdm import tqdm
from natsort import natsorted
from CompactionAnalyzer.CompactionFunctions import *
from CompactionAnalyzer.utilities import *
from CompactionAnalyzer.StructureTensor import *
from CompactionAnalyzer.plotting import *
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage import img_as_float
from skimage.morphology import reconstruction
import warnings



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

def regional_maxima(img):
    # following https://scikit-image.org/docs/stable/auto_examples/color_exposure/plot_regional_maxima.html
    image = img_as_float(img)
    mask = image
    seed = np.copy(image)
    seed[1:-1, 1:-1] = image.min()
    dilated = reconstruction(seed, mask, method='dilation')
    image=image-dilated
    return image

def segment_cell(img, thres=1, seg_gaus1 = 0.5, seg_gaus2=100, seg_iter=1,show_segmentation = False, 
                 segmention_method="otsu", seg_invert=False, regional_max_correction=True, segmention_min_area=1000):   
    """
    Image segmentation function to create  mask, radius, and position of a spheroid in a grayscale image.
    Args:
        img(array): Grayscale image as a Numpy array
        thres(float): To adjust the segmentation, keep 1
        segmention_method: use "otsu", "yen" or "entropy" as segmentation method
        seg_iter: iterations of closing steps , might increase to get more 
        robust segmentation but less precise segmentation, by default 1
        seg_invert(bool): Default is false ans segements bright objects, if True
        the segementatio is inverted and detects dark objects
    Returns:
        dict: Dictionary with keys: mask, radius, centroid (x/y)
    """
    height = img.shape[0]
    width = img.shape[1]
    
    
    # local gaussian (or single gaussian if gaus2 == None)  
    if seg_gaus2 is not None:
        img = np.abs(gaussian(img, sigma=seg_gaus1) - gaussian(img, sigma=seg_gaus2))
    else:
        img = gaussian(img, sigma=seg_gaus1) 
        
    # regional maxima to get rid of background noise
    if regional_max_correction:
        img = regional_maxima(img)
      
    # segment cell
    if segmention_method == "yen":
        if seg_invert==False:
            mask = img > threshold_yen(img) * thres

        if seg_invert==True:
            mask = img < threshold_yen(img) * thres 

    if segmention_method == "otsu":
         if seg_invert==False:
            mask = img > threshold_otsu(img) * thres

         if seg_invert==True:
            mask = img < threshold_otsu(img) * thres
            
    if segmention_method == "entropy":    
         if seg_invert==False:
            img2 = entropy(img.astype(np.uint8), disk(10))
            mask = img2 > threshold_otsu(img2) * thres
         if seg_invert==True:
            img2 = entropy(img.astype(np.uint8), disk(10))
            mask = img2 < threshold_otsu(img2) * thres
                     

    # remove other objects
    if seg_iter is not None:
        mask = scipy_morph.binary_closing(mask, iterations=seg_iter)
        mask = remove_small_objects(mask, min_size=segmention_min_area)
        mask = scipy_morph.binary_dilation(mask, iterations=seg_iter)
        mask = scipy_morph.binary_fill_holes(mask)
    

    # identify spheroid as the most centered object
    labeled_mask, max_lbl = scipy_meas.label(mask)
    center_of_mass = np.array(scipy_meas.center_of_mass(mask, labeled_mask, range(1, max_lbl + 1)))
    distance_to_center = np.sqrt(np.sum((center_of_mass - np.array([height / 2, width / 2])) ** 2, axis=1))

    mask = (labeled_mask == distance_to_center.argmin() + 1)

    # show segmentation
    if show_segmentation:
        plt.ion()
        plt.figure()
        plt.subplot(121), plt.imshow(img); plt.title("Gauss blurred cell image")
        plt.subplot(122), plt.imshow(mask); plt.title("Segmention")
        plt.tight_layout(); plt.show()
        print ("Display segmention and stop evaluation for testing")
        stop  # just display segmention and stop evaluation for testing 
        
    
    # determine radius of spheroid
    radius = np.sqrt(np.sum(mask) / np.pi)

    # determine center position of spheroid
    cy, cx = center_of_mass[distance_to_center.argmin()]

    # return dictionary containing spheroid information
    return {'mask': mask, 'radius': radius, 'centroid': (cx, cy)}




def StuctureAnalysisMain(fiber_list,
                         cell_list, 
                         out_list,
                         scale=1,                       # imagescale as um per pixel
                         sigma_tensor = None ,          # sigma of applied gauss filter / window for structure tensor analysis in px
                                                        # should be in the order of the objects to analyze !! 
                                                        # 7 um for collagen 
                         edge = 40   ,                  # Cutt of pixels at the edge since values at the border cannot be trusted
                         segmention_thres = 1.0 ,       # for cell segemetntion, thres 1 equals normal otsu threshold , user also can specify gaus1 + gaus2 in segmentation if needed
                         seg_gaus1=0.5, seg_gaus2 = 100 , # 2 gaus filters used as bandpassfilter for local contrast enhancement; For seg_gaus2 = None a single gauss filter is applied
                         max_dist = None,               # optional: specify the maximal distance around cell center for analysis (in px)
                         regional_max_correction = True,# background correction using regional maxima approach
                         show_segmentation = False ,    # display the segmentation output
                         sigma_first_blur  = 0.5  ,     # slight first bluring of whole image before appplying the structure tensor
                         angle_sections = 5    ,        # size of angle sections in degree 
                         shell_width =  None        ,   # pixel width of distance shells (px-value=um-value/scale)
                         manual_segmention = False    , # segmentation of mask by manual clicking the cell outline
                         plotting = True     ,          # creates and saves individual figures additionally to the excel files 
                         dpi = 200      ,               # resolution of figures 
                         SaveNumpy = False       ,      # saves numpy arrays for later analysis - can create large data files
                         norm1=1,norm2 = 99  ,          # contrast spreading for input images between norm1- and norm2-percentile; values below norm1-percentile are set to zero and
                                                        # values above norm2-percentile are set to 1
                         seg_invert=False,              # if segmentation is inverted dark objects are detected inseated of bright objects
                         seg_iter = 1,                  # number of repetitions of  binary closing, dilation and filling holes steps
                         segmention_method="otsu",      #  use "otsu", "entropy" or "yen"  as segmentation method
                         segmention_min_area = 1000,    # small bjects below this px-area are removed during cell segmentation
                         load_segmentation = False,     # if True enter the path of the segementation.npy - file in path_seg
                         path_seg = None):              # to load a mask
    """
    Main analysis
    
    Loops through the list of all fiber and cell images and computes the structure analysis. 
    Results are stored in the folders specified in the out_list analog to the 2 other list 
    (all three lists can be genereated automatically by generate_lists)
    
    Setting are described above

    Returns
    -------
    None.

    """
    plt.ioff()
   # plt.ion()
    
    # if not specified use 7um for tensor analysis and 5um shells
    if not sigma_tensor:
        sigma_tensor = 7/scale
    if not shell_width:
        shell_width = 5/scale
        
   
        
    # loop thorugh cells
    for n,i in tqdm(enumerate(fiber_list)):

        #create output folder if not existing
        if not os.path.exists(out_list[n]):
            os.makedirs(out_list[n])
           
        #### save a parameters file   
        import yaml
        dict_file = {
                  'Parameters': {'scale': [scale], 'sigma_tensor': [sigma_tensor], 'edge': [edge],
                                 'seg_gaus1': [seg_gaus1], 'seg_gaus2': [seg_gaus2], 'show_segmentation': [show_segmentation],
                                 'regional_max_correction': [regional_max_correction], 'sigma_first_blur': [sigma_first_blur], 
                                 'angle_sections': [angle_sections], 'shell_width': [shell_width],
                                 'manual_segmention': [manual_segmention], 'plotting': [plotting], 'dpi': [dpi],
                                 'SaveNumpy': [SaveNumpy], 'norm1': [norm1], 'norm2': [norm2],
                                 'seg_invert': [seg_invert], 'seg_iter': [seg_iter], 'segmention_method': [segmention_method],
                                 'load_segmentation': [load_segmentation], 'path_seg': [path_seg],
                                 'max_dist': [max_dist]
                                 
                                 },
                  'Data' :   {'fiber_list': [fiber_list], 'cell_list': [cell_list], 'out_list': [out_list]}
                    }  
        with open(os.path.join(out_list[n],"parameters.yml"), 'w') as yaml_file:
            yaml.dump(dict_file, yaml_file, default_flow_style=False)
  
            
        # load images
        im_cell  = imageio.imread(cell_list[n])  #color.rgb2gray(..)
        im_fiber = imageio.imread(fiber_list[n])   #color.rgb2gray()
        
        ## if 3 channels convert to grey  
        if len(im_cell.shape) == 3 :
            im_cell = color.rgb2gray(im_cell)
        if len(im_fiber.shape) == 3 :
            im_fiber = color.rgb2gray(im_fiber)    
        
        # # applying normalizing/ contrast spreading
        im_cell_n = normalize(im_cell, norm1, norm2)
        im_fiber_n = normalize(im_fiber, norm1, norm2)  
        im_fiber_g = gaussian(im_fiber_n, sigma=sigma_first_blur)     # blur fiber image slightly (test with local gauss - similar)
        

        # segment cell (either manual or automatically)
        if manual_segmention==True:
            segmention = custom_mask(im_cell_n)
    
        if (manual_segmention==False) and (load_segmentation == False):
            segmention = segment_cell(im_cell_n, thres= segmention_thres, seg_gaus1 = seg_gaus1, seg_gaus2=seg_gaus2,
                                      show_segmentation = show_segmentation,seg_invert=seg_invert,seg_iter=seg_iter,
                                      segmention_method=segmention_method, regional_max_correction=regional_max_correction, segmention_min_area=segmention_min_area)   
              
        if load_segmentation:
              segmention = np.load(path_seg,allow_pickle=True).item()             
        
        # center of the new cropped image (to avoid edge effects)
        center_small = (segmention["centroid"][0]-edge,segmention["centroid"][1]-edge)
        
        # set segmention mask to nan to avoid effects within cell  (maybe not needed if signal below cell makes sense)
        im_fiber_g_forstructure = im_fiber_g.copy()
        im_fiber_g_forstructure[segmention["mask"]] = np.nan
     
        """
        Structure tensor
        """
        # Structure Tensor Orientation
        # get structure tensor
        ori, max_evec, min_evec, max_eval, min_eval = analyze_local(im_fiber_g_forstructure, sigma=sigma_tensor, size=0, filter_type="gaussian")
        # cut off edges as specified
        if edge != 0:
            ori, max_evec, min_evec, max_eval, min_eval = ori[edge:-edge,edge:-edge], max_evec[edge:-edge,edge:-edge], min_evec[edge:-edge,edge:-edge], \
                                                          max_eval[edge:-edge,edge:-edge], min_eval[edge:-edge,edge:-edge]
        """
        coordinates
        """
        # Calculate Angle + Distances 
        y,x = np.indices(ori.shape)
        dx = x - center_small[0]
        dy = y - center_small[1]
        distance = np.sqrt(dx ** 2 + dy ** 2)  # dist to center
        angle = np.arctan2(dy, dx) *360/(2*np.pi)
        dx_norm = (dx/distance)
        dy_norm = (dy/distance)
        dist_surface = distance_transform_edt(~segmention["mask"])[edge:-edge,edge:-edge] # dist to surface
    
          
        """
        total image analysis
        """
        # Angular deviation from orietation to center vector - 
        angle_dev = np.arccos(np.abs(dx_norm * min_evec[:,:,0] + dy_norm*min_evec[:,:,1])) * 360/(2*np.pi) 
        # calculate oreination from angle_dev since they are normal distributed and np.abs((dx_norm * min_evec[:,:,0] + dy_norm*min_evec[:,:,1])) is not
        orientation_dev_01 = angle_dev/90  
        # orientation_dev_01 = np.abs((dx_norm * min_evec[:,:,0] + dy_norm*min_evec[:,:,1]))
        orientation_dev = -(2*orientation_dev_01-1)  # norm from -1 to 1 ; changed sign
        
        ### set values to nan if max distance is specified
        ## for following analysis then only orientation within a certtain distance to the cell are used
        if max_dist:
            angle_dev[distance>=max_dist] = np.nan
            ori[distance>=max_dist] = np.nan
            angle[distance>=max_dist] = np.nan # also for angular evaluation
  
        # weighting by coherence
        angle_dev_weighted = (angle_dev * ori) / np.nanmean(ori)     # no angle values anymore (stretched due weights) but the mean later is again an angle
        orientation_dev_weighted_01 =  ((orientation_dev_01 * ori) / np.nanmean(ori)) 
        orientation_dev_weighted = -(2  *orientation_dev_weighted_01 - 1)
        

        # weighting by coherence and image intensity
        im_fiber_g = im_fiber_g[edge:-edge,edge:-edge]
        # could also use non filtered image
        # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4497681/ these guys
        # use a threshold in the intensity image// only coherence and orientation vectors
        # corresponding to pixels in the intesity iamage above a threshold are considered.
        weight_image = gaussian(im_fiber_g,sigma=15)
        angle_dev_weighted2 = (angle_dev_weighted *weight_image) / np.nanmean(weight_image)
        orientation_dev_weighted2_01 = (orientation_dev_01 *weight_image*ori) / np.nanmean(weight_image*ori)
        orientation_dev_weighted2 =  -(2 * orientation_dev_weighted2_01 - 1)
        
        # also weighting the coherency like this
        ori_weight2 = (ori * weight_image) / np.nanmean(weight_image)
        
     
        # GRADIENT towards center   
        grad_y = np.gradient(im_fiber_g, axis=0)
        grad_x = np.gradient(im_fiber_g, axis=1)
        dx_norm_a = -dy_norm.copy()
        dy_norm_a = dx_norm.copy()
        s_to_center = ((grad_x * dx_norm) + (grad_y * dy_norm))**2
        s_around_center = ((grad_x * dx_norm_a) + (grad_y * dy_norm_a))**2
        # Ignore RuntimeWarnings here 
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            s_norm1 = (s_around_center - s_to_center)/(grad_x**2 + grad_y**2)
            #s_norm2 = (s_around_center - s_to_center)/(s_around_center + s_to_center)
            
        
        # save values for total image analysis
        # Total Value for complete image (without the mask)
        # averages
        alpha_dev_total1 = np.nanmean(angle_dev[(~segmention["mask"][edge:-edge,edge:-edge])])
        alpha_dev_total2 = np.nanmean(angle_dev_weighted[(~segmention["mask"][edge:-edge, edge:-edge])])
        alpha_dev_total3 = np.nanmean(angle_dev_weighted2[(~segmention["mask"][edge:-edge, edge:-edge])])
        # cos_dev_total1 = np.nanmean(np.cos(2*angle_dev[(~segmention["mask"][edge:-edge, edge:-edge])]*np.pi/180))
        # cos_dev_total2 = np.nanmean(np.cos(2*angle_dev_weighted[(~segmention["mask"][edge:-edge, edge:-edge])]*np.pi/180))
        # cos_dev_total3 = np.nanmean(np.cos(2*angle_dev_weighted2[(~segmention["mask"][edge:-edge, edge:-edge])]*np.pi/180))   
        cos_dev_total1 = np.nanmean(orientation_dev[(~segmention["mask"][edge:-edge,edge:-edge])   ])
        cos_dev_total2 = np.nanmean(orientation_dev_weighted[(~segmention["mask"][edge:-edge,edge:-edge])     ])
        cos_dev_total3 = np.nanmean(orientation_dev_weighted2[(~segmention["mask"][edge:-edge,edge:-edge])     ])
        coh_total = np.nanmean(ori[(~segmention["mask"][edge:-edge,edge:-edge]) ])
        coh_total2 = np.nanmean(ori_weight2[(~segmention["mask"][edge:-edge, edge:-edge])])
             
        
        # create excel sheet with results for total image   
        # initialize result dictionary
        results_total = {'Mean Coherency': [], 'Mean Coherency (weighted by intensity)': [], 'Mean Angle': [],
                   'Mean Angle (weighted by coherency)': [], 'Mean Angle (weighted by intensity and coherency)': [], 'Orientation': [],
                   'Orientation (weighted by coherency)': [], 'Orientation (weighted by intensity and coherency)': [], }       
       
        results_total['Mean Coherency'].append(coh_total)
        results_total['Mean Coherency (weighted by intensity)'].append(coh_total2)
        results_total['Mean Angle'].append(alpha_dev_total1)
        results_total['Mean Angle (weighted by coherency)'].append(alpha_dev_total2)
        results_total['Mean Angle (weighted by intensity and coherency)'].append(alpha_dev_total3)
        results_total['Orientation'].append(cos_dev_total1)
        results_total['Orientation (weighted by coherency)'].append(cos_dev_total2)
        results_total['Orientation (weighted by intensity and coherency)'].append(cos_dev_total3)
        
        excel_total = pd.DataFrame.from_dict(results_total)
        excel_total.to_excel(os.path.join(out_list[n],"results_total.xlsx"))
    
    
        """
        Angular sections
        """
        
        
        # initialize result dictionary
        results_angle = {'Angles': [], 'Angles Plotting': [], 'Angle Deviation': [], 'Angle Deviation (weighted by coherency)': [], 
                         'Angle Deviation (weighted by intensity and coherency)': [],
                         'Orientation': [], 'Orientation (weighted by coherency)': [], 
                         'Orientation (weighted by intensity and coherency)': [], 
                         'Coherency (weighted by intensity)': [],'Coherency': [], 
                         'Gradient': [],'Mean Intensity': []                  
                         }      
    
        # make the angle analysis in sections
        for alpha in range(-180, 180, angle_sections):
                # mask the angle section
                mask_angle = (angle > (alpha-angle_sections/2)) & (angle <= (alpha+angle_sections/2)) & (~segmention["mask"][edge:-edge,edge:-edge])
                
                # in case we have a distance limit only draw angles upn that distance ### not necessary anymore now use "max_dist" for both
                # if angle_sections_distance:
                #     mask_angle = (angle > (alpha-angle_sections/2)) & (angle <= (alpha+angle_sections/2)) & (~segmention["mask"][edge:-edge,edge:-edge]) & (distance<angle_sections_distance)
                     
                # special case at borders   
                if alpha == -180:
                      mask_angle = ((angle > (180-angle_sections/2)) | (angle <= (alpha+angle_sections/2))) & (~segmention["mask"][edge:-edge,edge:-edge])
                            
                if alpha == 180:
                      mask_angle = ((angle > (alpha-angle_sections/2)) | (angle <= (-180 +angle_sections/2))) & (~segmention["mask"][edge:-edge,edge:-edge])     
                     
               
                # save angle data
                angle_current = alpha * np.pi / 180
                results_angle['Angles'].append(angle_current)  
                if angle_current>0:
                    results_angle['Angles Plotting'].append(np.abs(angle_current-(2*np.pi)))  
                else:
                    results_angle['Angles Plotting'].append(np.abs(angle_current))  
                results_angle['Coherency'].append(np.nanmean(ori[mask_angle]))
                results_angle['Coherency (weighted by intensity)'].append(np.nanmean(ori_weight2[mask_angle]))
                results_angle['Mean Intensity'].append(np.nanmean(im_fiber_g[mask_angle]))
                results_angle['Gradient'].append(np.nanmean(s_norm1[mask_angle]))
                results_angle['Angle Deviation'].append(np.nanmean(angle_dev[mask_angle]))
                # weight per slice
                results_angle['Angle Deviation (weighted by coherency)'].append(np.nanmean(angle_dev[mask_angle]*ori[mask_angle]/np.nanmean(ori[mask_angle])))
                results_angle['Angle Deviation (weighted by intensity and coherency)'].append(np.nanmean(angle_dev[mask_angle]*ori[mask_angle]*weight_image[mask_angle]/np.nanmean(ori[mask_angle]*weight_image[mask_angle])))
                results_angle['Orientation'].append(np.nanmean(orientation_dev[mask_angle]))
                results_angle['Orientation (weighted by coherency)'].append(np.nanmean(orientation_dev[mask_angle]*ori[mask_angle]/np.nanmean(ori[mask_angle])))
                results_angle['Orientation (weighted by intensity and coherency)'].append(np.nanmean(orientation_dev[mask_angle]*ori[mask_angle]*weight_image[mask_angle]/np.nanmean(ori[mask_angle]*weight_image[mask_angle])))

        # create excel sheet with results for angle analysis       
        excel_angles= pd.DataFrame.from_dict(results_angle)
        excel_angles.to_excel(os.path.join(out_list[n],"results_angles.xlsx"))
    
    
        """
        Distance Evaluation
        """
        
         # initialize result dictionary
        results_distance = {'Shell_mid (px)': [], 'Shell_mid (µm)': [], 'Intensity (accumulated)': [], 
                            'Intensity (individual)': [], 'Intensity Norm (individual)': [],
                            'Intensity Norm (accumulated)': [],  'Angle (accumulated)': [],  'Angle (individual)': [],  'Angle (individual-weightInt)': [], 
                            'Orientation (accumulated)': [], 'Orientation (individual)': [], 'Orientation (individual-weightInt)': [],
                            'Angle (individual)': [], 
                            'Intensity disttocenter (accumulated)': [], 
                            'Intensity disttocenter (individual)': [], 'Intensity Norm disttocenter (individual)': [], 
                            'Intensity Norm disttocenter (accumulated)': [], 
                            'Angle disttocenter (accumulated)': [],  'Angle disttocenter (individual)': [],  
                            'Orientation disttocenter (accumulated)': [], 'Orientation disttocenter (individual)': [], 
                            'Angle disttocenter (individual)': []            
                         }      
    
        mask_shells = {'Mask_shell': [], 'Mask_shell_center': [] } 
        # shell distances
        shells = np.arange(0, dist_surface.max(), shell_width)
        midofshells = (shells + shell_width/2)[:-1]
        results_distance['Shell_mid (px)'].extend(list(midofshells) )
        results_distance['Shell_mid (µm)'].extend([i*scale for i in midofshells])
    
        
        # make the distance shell analysis
        for i in range(len(shells)-1):
            # Ignore RuntimeWarnings here 
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                
                # distance shells parallel to surface
                # mask of individual shells and accumulation of all points closer to the correponding cell
                mask_shell = (dist_surface > (shells[i])) & (dist_surface <= (shells[i+1])) & (~segmention["mask"][edge:-edge,edge:-edge])  
                mask_shell_lower=  (dist_surface <= (shells[i+1])) & (~segmention["mask"][edge:-edge,edge:-edge])
                mask_shells['Mask_shell'].append(mask_shell)
                # calculate mintensity and angle deviation within the growing shells (always within start shell to highest shell)
                # weight wuth shell values :
                    
                # Save Angle and Orientation - Weight by coherency or coherency+intensity shell-wise now
                results_distance['Angle (accumulated)'].append(np.nanmean(angle_dev[mask_shell_lower]*ori[mask_shell_lower]/np.nanmean(ori[mask_shell_lower])))
                results_distance['Angle (individual)'].append(np.nanmean(angle_dev[mask_shell]*ori[mask_shell]/np.nanmean(ori[mask_shell]) ))    # exclusively in certain shell
                results_distance['Angle (individual-weightInt)'].append(np.nanmean(angle_dev[mask_shell]*ori[mask_shell]*weight_image[mask_shell]/np.nanmean(ori[mask_shell]*weight_image[mask_shell]) ))    # exclusively in certain shell
                results_distance['Orientation (accumulated)'].append(np.nanmean(orientation_dev[mask_shell_lower]*ori[mask_shell_lower]/np.nanmean(ori[mask_shell_lower])))
                results_distance['Orientation (individual)'].append(np.nanmean(orientation_dev[mask_shell]*ori[mask_shell]/np.nanmean(ori[mask_shell]) ))    # exclusively in certain shell
                results_distance['Orientation (individual-weightInt)'].append(np.nanmean(orientation_dev[mask_shell]*ori[mask_shell]*weight_image[mask_shell]/np.nanmean(ori[mask_shell]*weight_image[mask_shell]) ))    
               
        
               # mean intensity
                results_distance['Intensity (accumulated)'].append(np.nanmean(im_fiber_g[mask_shell_lower]))          # accumulation of lower shells
                results_distance['Intensity (individual)'].append(np.nanmean(im_fiber_g[mask_shell])  )    # exclusively in certain shell
      
                # distance shells as circles around center
                mask_shell_center = (distance > (shells[i])) & (distance <= (shells[i+1])) & (~segmention["mask"][edge:-edge,edge:-edge])  
                mask_shell_lower_center=  (distance <= (shells[i+1])) & (~segmention["mask"][edge:-edge,edge:-edge])
                mask_shells['Mask_shell_center'].append(mask_shell_center)
              
                
                results_distance['Angle disttocenter (accumulated)'].append(np.nanmean(angle_dev[mask_shell_lower_center]*ori[mask_shell_lower_center]/np.nanmean(ori[mask_shell_lower_center])))
                results_distance['Angle disttocenter (individual)'].append(np.nanmean(angle_dev[mask_shell_center]*ori[mask_shell_center]/np.nanmean(ori[mask_shell_center]) ))    # exclusively in certain shell
                results_distance['Orientation disttocenter (accumulated)'].append(np.nanmean(orientation_dev_weighted[mask_shell_lower_center]*ori[mask_shell_lower_center]/np.nanmean(ori[mask_shell_lower_center])))
                results_distance['Orientation disttocenter (individual)'].append(np.nanmean(orientation_dev_weighted[mask_shell_center]*ori[mask_shell_center]/np.nanmean(ori[mask_shell_center]) ))    # exclusively in certain shell
                # mean intensity
                results_distance['Intensity disttocenter (accumulated)'].append(np.nanmean(im_fiber_g[mask_shell_lower_center]))          # accumulation of lower shells
                results_distance['Intensity disttocenter (individual)'].append(np.nanmean(im_fiber_g[mask_shell_center])  )    # exclusively in certain shell
                
    
        # create excel sheet with results for angle analysis       
        # norm intensities   (Baseline: mean intensity of the 2 outmost shells)
        norm_individ_int = np.array(results_distance['Intensity (individual)']/np.nanmean(np.array(results_distance['Intensity (individual)'][-2:])))
        results_distance['Intensity Norm (individual)'].extend( list(norm_individ_int) )  
        norm_accum_int = np.array(results_distance['Intensity (accumulated)'])/ np.nanmean(np.array(results_distance['Intensity (accumulated)'][-2:]))
        results_distance['Intensity Norm (accumulated)'].extend(list(norm_accum_int) )  
        
        # norm intensities   (Baseline: mean intensity of the 2 outmost shells)
        # now distance shells are calculatedt spherical around center instead of surface shells
        norm_individ_int_c = np.array(results_distance['Intensity disttocenter (individual)']/np.nanmean(np.array(results_distance['Intensity disttocenter (individual)'][-2:])))
        results_distance['Intensity Norm disttocenter (individual)'].extend( list(norm_individ_int_c) )  
        norm_accum_int_c = np.array(results_distance['Intensity disttocenter (accumulated)'])/ np.nanmean(np.array(results_distance['Intensity disttocenter (accumulated)'][-2:]))
        results_distance['Intensity Norm disttocenter (accumulated)'].extend(list(norm_accum_int_c) )    
        # create excel sheet with results for angle analysis       
        excel_distance =  pd.DataFrame.from_dict(results_distance)
        excel_distance.to_excel(os.path.join(out_list[n],"results_distance.xlsx"))
        
      
        # Halflife values - Leave for now..
        # # Calculate value where ntensity drops 25%
        # distintdrop = np.abs(np.array(results_distance['Intensity Norm (individual)'])-0.75)
        # # distance where int  drops   to 75% 
        # halflife_int =  midofshells[np.where(distintdrop  == np.nanmin(distintdrop))[1]]
        # # if decrease is not within range (minimum equals last value) then set to nan
        # if halflife_int == midofshells[-1]:
        #     halflife_int = np.nan
        # # # Calculate value where orientation drops to 75% within maxorientation(min) to 45° (random) range 
        # # difference to 45 degree instead of min-max range    
        # # calculate halflife of maximal orientation over distance
        # # difference to 45 degree for all
        # diffdist = np.array(dist_angle_individ)-45
        # # maximal orientation
        # diffmax = np.nanmin(diffdist)
        # diffmax_pos = np.where(diffmax==diffdist)[0][0]
        # # difference  angle drops to 75% 
        # diff2 = np.abs(diffdist-(0.75*diffmax))
        # diff2[:diffmax_pos] = np.nan    # only look at distances on the right side /further out 
        # halflife_ori =  midofshells[np.where(diff2 == np.nanmin(diff2))]    
        # # if decrease is not within range (minimum equals last value) then set to nan
        # if halflife_ori == midofshells[-1]:
        #     halflife_ori = np.nan
            # try:
        #     np.savetxt(os.path.join(out_list[n],"meanangle_within10shells.txt"), [dist_angle_accum[9]]) 
        # except:
        #     pass
    
        if SaveNumpy:
            #create output folder if not existing
            numpy_out = os.path.join(out_list[n], r"NumpyArrays" )
            if not os.path.exists(numpy_out):
                os.makedirs(numpy_out)
                
            np.save(os.path.join(numpy_out, "segmention.npy" ),segmention)
            np.save(os.path.join(numpy_out, "AngleMap.npy" ),angle_dev )    
            np.save(os.path.join(numpy_out, "AngleMap(weight_int).npy" ),angle_dev_weighted)    
            np.save(os.path.join(numpy_out, "AngleMap(weight_int_coh).npy"),angle_dev_weighted2 )    
            np.save(os.path.join(numpy_out, "OrientationMap.npy" ),orientation_dev )    
            np.save(os.path.join(numpy_out, "OrientationMap_weight_int.npy" ),orientation_dev_weighted)    
            np.save(os.path.join(numpy_out, "OrientationMap_weight_intcoh.npy" ),orientation_dev_weighted2 )    
            np.save(os.path.join(numpy_out, "FiberImageCrop.npy" ),normalize(im_fiber_n[edge:-edge,edge:-edge]) )      
            np.save(os.path.join(numpy_out, "CoherencyMap.npy" ),ori )  
            np.save(os.path.join(numpy_out, "CoherencyMap(weighted_int).npy" ),ori_weight2)  
            np.save(os.path.join(numpy_out, "Vector_min_ax0.npy"),min_evec[:,:,0])  
            np.save(os.path.join(numpy_out, "Vector_min_ax1.npy"),min_evec[:,:,1])   
            np.save(os.path.join(numpy_out, "mask_surface_shells.npy"),mask_shells['Mask_shell'])                                                                                   
            np.save(os.path.join(numpy_out, "mask_spherical_shells.npy"),mask_shells['Mask_shell_center'])   
                                                           
        
    
    
        """
        Plott results
        """
        
        if plotting:
            #create output folder if not existing
            figures = os.path.join(out_list[n],"Figures")
            if not os.path.exists(figures):
                os.makedirs(figures)
    
            # plot orientation and angle deviation maps together with orientation structure 
            plot_angle_dev(angle_map = angle_dev, 
                           vec0=min_evec[:,:,0] ,vec1=min_evec[:,:,1] ,coherency_map=ori,
                           path_png= os.path.join(figures,"Angle_deviation.png"),label="Angle Deviation",dpi=dpi,cmap="viridis_r")
            plot_angle_dev(angle_map = angle_dev_weighted2, 
                           vec0=min_evec[:,:,0] ,vec1=min_evec[:,:,1] ,coherency_map=ori,
                           path_png= os.path.join(figures,"Angle_deviation_weighted.png"),label="Angle Deviation",dpi=dpi,cmap="viridis_r")
            plot_angle_dev(angle_map = orientation_dev_weighted2 ,  
                           vec0=min_evec[:,:,0] ,vec1=min_evec[:,:,1] ,coherency_map=ori,
                           path_png= os.path.join(figures,"Orientation_weighted.png"),label="Orientation",dpi=dpi,cmap="viridis")
            plot_angle_dev(angle_map = orientation_dev,  
                           vec0=min_evec[:,:,0] ,vec1=min_evec[:,:,1] ,coherency_map=ori,
                           path_png= os.path.join(figures,"Orientation.png"),label="Orientation",dpi=dpi,cmap="viridis")
            plt.close("all") 
            # pure coherency and pure orientation
            plot_coherency(ori,path_png= os.path.join(figures,"coherency_noquiver.png"))
            plot_coherency(orientation_dev_weighted2,
                           path_png= os.path.join(figures,"Orientation_weighted_noquiver.png"),
                           label="Orientation",dpi=dpi) 
            plt.close("all") 
            # Polar plots        
            plot_polar(results_angle['Angles Plotting'], results_angle['Coherency (weighted by intensity)'],
                       path_png= os.path.join(figures,"polar_coherency_weighted.png"), label = "Coherency (weighted)",dpi=dpi)
            
                    
            plot_polar(results_angle['Angles Plotting'], results_angle['Coherency (weighted by intensity)'],
                       path_png= os.path.join(figures,"polar_coherency_double.png"), label = "Coherency (weighted)",
                       something2 = results_angle['Coherency'], label2 = "Coherency",dpi=dpi)
            
            plot_polar(results_angle['Angles Plotting'], results_angle['Mean Intensity'],
                       path_png= os.path.join(figures,"polar_intensity.png"), label = "Mean Intensity",dpi=dpi)
            
            plot_polar(results_angle['Angles Plotting'], results_angle['Orientation (weighted by intensity and coherency)'],
                               path_png= os.path.join(figures,"Orientation_weighted_polar.png"), label = "Orientation",dpi=dpi)
            
            plot_polar(results_angle['Angles Plotting'], results_angle['Orientation'],
                               path_png= os.path.join(figures,"Orientation_polar.png"), label = "Orientation",dpi=dpi)
            plt.close("all") 
            # summarizing triple plot
            plot_triple(results_angle,results_total , path_png= os.path.join(figures,"Triple_plot.png") ,dpi=dpi)
       
            # quiver plots with center overlay   
            quiv_coherency_center(vec0=min_evec[:,:,0] ,vec1=min_evec[:,:,1] ,coherency_map=ori,
                             center0=center_small[0],center1 = center_small[1],
                           path_png= os.path.join(figures,"quiv_coherency_center.png"),dpi=dpi )
            plt.close("all") 
            # image fiber + segmention
            plot_fiber_seg(fiber_image=normalize(im_fiber_n[edge:-edge,edge:-edge]) ,
                           c0=center_small[0],c1=center_small[1],
                           segmention=segmention["mask"][edge:-edge,edge:-edge], 
                           path_png=os.path.join(figures,"fiber_segemention.png"),dpi=dpi,scale=scale )
        
            # plot overlaywith fiber image 
            plot_overlay(fiber_image=normalize(im_fiber_n[edge:-edge,edge:-edge]) ,
                           c0=center_small[0],c1=center_small[1], vec0=min_evec[:,:,0],
                           vec1=min_evec[:,:,1], coherency_map=ori, show_n=15,
                           segmention=segmention["mask"][edge:-edge,edge:-edge], 
                           path_png=os.path.join(figures,"overlay.png"),dpi=dpi ,scale=scale)
            
            plot_overlay(fiber_image=normalize(im_fiber_n[edge:-edge,edge:-edge]) ,
                           c0=center_small[0],c1=center_small[1], vec0=min_evec[:,:,0],
                           vec1=min_evec[:,:,1], coherency_map=ori,show_n=10,
                           segmention=segmention["mask"][edge:-edge,edge:-edge], 
                           path_png=os.path.join(figures,"overlay2.png"),dpi=dpi ,scale=scale)
            plt.close("all") 
            ### DISTANCE PLOTS
            # plot shells - deactived as it consumes a lot of time
            # plot_shells(mask_shells['Mask_shell'],path_png=os.path.join(figures,"shells.png"),dpi=dpi )
        
            # plot intensity and orientation (idividual) over distance 
            plot_distance(results_distance,string_plot = "Orientation (individual)",
                      path_png=os.path.join(figures,"Orientation_distance.png"),dpi=dpi)
        
            plot_distance(results_distance,string_plot = "Intensity Norm (individual)",
              path_png=os.path.join(figures,"Intensity_distance.png"),dpi=dpi)
            plt.close("all") 
            # save raw cell image   
            plot_cell(im_cell_n, path_png=os.path.join(figures,"cell-raw.png"),  scale=scale, dpi=dpi)

            plt.close("all")    
            

    return






def SummarizeResultsTotal(data, output_folder= None):
    if not output_folder:
        output_folder=os.path.join(data,"CombinedFiles")
     #create output folder if not existing
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)    
    
    list_total = glob.glob(data+"\\**\\results_total.xlsx", recursive=True)

    
    
    # initialize a combining result total dictionary for all cells
    results_total_combined = { 'Path':[], 'Mean Angle (weighted by coherency)': [], 'Mean Angle (weighted by intensity and coherency)': [], 
               'Orientation (weighted by coherency)': [], 'Orientation (weighted by intensity and coherency)': [], 
               'Overall weighted Oriantation (mean all cells)': [],
               'Overall weighted Oriantation (std all cells)': []} 
    
    for i, path in enumerate(list_total):
       results_total_combined['Orientation (weighted by coherency)'].append(float(pd.read_excel(list_total[i])['Orientation (weighted by coherency)']))
       results_total_combined['Orientation (weighted by intensity and coherency)'].append(float(pd.read_excel(list_total[i])['Orientation (weighted by intensity and coherency)']))
       results_total_combined['Mean Angle (weighted by coherency)'].append(float(pd.read_excel(list_total[i])['Mean Angle (weighted by coherency)']))
       results_total_combined['Mean Angle (weighted by intensity and coherency)'].append(float(pd.read_excel(list_total[i])['Mean Angle (weighted by intensity and coherency)']))
       results_total_combined['Path'].append(str(list_total[i]))
       
    ovreall_orientation = np.nanmean(results_total_combined['Orientation (weighted by intensity and coherency)'])  
    ovreall_orientation_std = np.nanstd(results_total_combined['Orientation (weighted by intensity and coherency)'])  
    results_total_combined['Overall weighted Oriantation (mean all cells)'].extend([ovreall_orientation]*len(list_total))
    results_total_combined['Overall weighted Oriantation (std all cells)'].extend([ovreall_orientation_std]*len(list_total))
    
        
     # create excel sheet with results for angle analysis       
    excel_total_combined =  pd.DataFrame.from_dict(results_total_combined)
    excel_total_combined.to_excel(os.path.join(output_folder,"results_total_combined.xlsx"))
    
    
    return results_total_combined



def SummarizeResultsDistance(data, output_folder= None, dpi=200):
    plt.ioff()
    if not output_folder:
        output_folder=os.path.join(data,"CombinedFiles")
     #create output folder if not existing
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)   
    
    list_distance = glob.glob(data+"\\**\\results_distance.xlsx", recursive=True)    
     # initialize a combining distance result dictionary for all cells
    results_distance_combined = { 'Path':[], 'Shell_mid (µm)': [], 'Intensity': [],
                              'Intensity Norm': [], 'Orientation': [],
                              'Angle': []} 

    # find maximum length (different distances in images depending on cell positon - but all should have same scaling)       
    find_max =  [np.array(pd.read_excel(list_distance[i])['Shell_mid (µm)']) for i, path in enumerate(list_distance)]
    max_length = np.max([len(i) for i in find_max])
    
    # make matrix according to longest cell and fill rest with np.nan 
    distances = np.empty([len(list_distance),max_length])
    distances[:] = np.nan
    # fill the matrix with data
    for n,i in enumerate(list_distance):
        length = len(pd.read_excel(list_distance[n])['Shell_mid (µm)'])
        distances[n,:length] = pd.read_excel(list_distance[n])['Shell_mid (µm)']  
    results_distance_combined['Shell_mid (µm)'].extend(np.nanmean(distances, axis=0)  )
    
    
    # make matrix according to longest cell and fill rest with np.nan 
    intensity = np.empty([len(list_distance),max_length])
    intensity[:] = np.nan
    # fill the matrix with data
    for n,i in enumerate(list_distance):
        length = len(pd.read_excel(list_distance[n])['Intensity (individual)'])
        intensity[n,:length] = pd.read_excel(list_distance[n])['Intensity (individual)']  
    results_distance_combined['Intensity'].extend(np.nanmean(intensity, axis=0)  )
    
    # make matrix according to longest cell and fill rest with np.nan 
    intensity_norm = np.empty([len(list_distance),max_length])
    intensity_norm[:] = np.nan
    # fill the matrix with data
    for n,i in enumerate(list_distance):
        length = len(pd.read_excel(list_distance[n])['Intensity Norm (individual)'])
        intensity_norm[n,:length] = pd.read_excel(list_distance[n])['Intensity Norm (individual)']  
    results_distance_combined['Intensity Norm'].extend(np.nanmean(intensity_norm, axis=0)  )
    
    # make matrix according to longest cell and fill rest with np.nan 
    angle = np.empty([len(list_distance),max_length])
    angle[:] = np.nan
    # fill the matrix with data
    for n,i in enumerate(list_distance):
        length = len(pd.read_excel(list_distance[n])['Angle (individual)'])
        angle[n,:length] = pd.read_excel(list_distance[n])['Angle (individual)']  
    results_distance_combined['Angle'].extend(np.nanmean(angle, axis=0)  )

    # make matrix according to longest cell and fill rest with np.nan 
    Orientation = np.empty([len(list_distance),max_length])
    Orientation[:] = np.nan
    # fill the matrix with data
    for n,i in enumerate(list_distance):
        length = len(pd.read_excel(list_distance[n])['Orientation (individual)'])
        Orientation[n,:length] = pd.read_excel(list_distance[n])['Orientation (individual)']  
    results_distance_combined['Orientation'].extend(np.nanmean(Orientation, axis=0)  )
    
    # append path names
    results_distance_combined['Path'].extend([str(data)]*max_length)
    
    
    # plot intensity and orientation averaged over all cells over distance 
    plot_distance(results_distance_combined,string_plot = "Intensity",
              path_png=os.path.join(output_folder,"Dist_Intensity_allcells.png"),dpi=dpi)
    
   
    # plot intensity and orientation averaged over all cells over distance 
    plot_distance(results_distance_combined,string_plot = "Intensity Norm",
              path_png=os.path.join(output_folder,"Dist_Intensity_Norm_allcells.png"),dpi=dpi)
    
    # plot intensity and orientation averaged over all cells over distance 
    plot_distance(results_distance_combined,string_plot = "Orientation",
              path_png=os.path.join(output_folder,"Dist_Orientation_allcells.png"),dpi=dpi)
    
        # plot intensity and orientation averaged over all cells over distance 
    plot_distance(results_distance_combined,string_plot = "Angle",
              path_png=os.path.join(output_folder,"Angle_allcells.png"),dpi=dpi)
    
    # create excel sheet with results for angle analysis       
    excel_distance_combined =  pd.DataFrame.from_dict(results_distance_combined)
    excel_distance_combined.to_excel(os.path.join(output_folder,"results_distance_combined.xlsx"))

    plt.close("all")


