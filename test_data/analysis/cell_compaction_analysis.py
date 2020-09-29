"""
Created on Wed Jul  1 15:45:34 2020

@author: david boehringer + andreas bauer

Evaluates the fiber orientation around cells that compact collagen tissue.
Evaluates the orientation of collagen fibers towards the segmented cell center 
by using the structure tensor (globally / in angle sections / within distance shells).

Needs an Image of the cell and one image of the fibers (e.g. maximum projection of small stack)

"""

import matplotlib.pyplot as plt
from skimage.filters import gaussian
from scipy.ndimage.filters import uniform_filter
import os
import copy
import numpy as np
from skimage.draw import circle
from scipy.signal import convolve2d
from fibre_orientation_structure_tensor import *
import glob as glob
from tqdm import tqdm
import matplotlib
import scipy.ndimage.morphology as scipy_morph
import scipy.ndimage.measurements as scipy_meas
from skimage.filters import gaussian, threshold_otsu, threshold_yen
from skimage.morphology import remove_small_objects
from skimage.exposure import adjust_gamma   
from PIL import Image
from copy import copy
import imageio
from skimage import color
from scipy.ndimage.morphology import distance_transform_edt
from natsort import natsorted

def set_vmin_vmax(x, vmin, vmax):
    if not isinstance(vmin, (float, int)):
        vmin = np.nanmin(x)
    if not isinstance(vmax, (float, int)):
        vmax = np.nanmax(x)
    if isinstance(vmax, (float, int)) and not isinstance(vmin, (float, int)):
        vmin = vmax - 1 if vmin > vmax else None
    return vmin, vmax


def filter_values(ar1, ar2, abs_filter=0, f_dist=3):
    '''
    function to filter out values from an array for better display
    :param ar1:
    :param ar2:
    :param ar:
    :param f_dist: distance betweeen filtered values
    :return:
    '''
    # ar1_show=np.zeros(ar1.shape)+np.nan
    # ar2_show=np.zeros(ar2.shape)+np.nan
    pixx = np.arange(np.shape(ar1)[0])
    pixy = np.arange(np.shape(ar1)[1])
    xv, yv = np.meshgrid(pixy, pixx)

    def_abs = np.sqrt((ar1 ** 2 + ar2 ** 2))
    select_x = ((xv - 1) % f_dist) == 0
    select_y = ((yv - 1) % f_dist) == 0
    select_size = def_abs > abs_filter
    select = select_x * select_y * select_size
    s1 = ar1[select]
    s2 = ar2[select]
    return s1, s2, xv[select], yv[select]


def show_quiver(fx, fy, filter=[0, 1], scale_ratio=0.4, headwidth=None, headlength=None, headaxislength=None,
                width=None, cmap="rainbow",
                figsize=None, cbar_str="", ax=None, fig=None
                , vmin=None, vmax=None, cbar_axes_fraction=0.2, cbar_tick_label_size=15
                , cbar_width="2%", cbar_height="50%", cbar_borderpad=0.1,
                cbar_style="not-clickpoints", plot_style="not-clickpoints", cbar_title_pad=1, plot_cbar=True, alpha=1,
                ax_origin="upper", **kwargs):
    # list of all necessary quiver parameters
    quiver_parameters = {"headwidth": headwidth, "headlength": headlength, "headaxislength": headaxislength,
                         "width": width, "scale_units": "xy", "angles": "xy", "scale": None}
    quiver_parameters = {key: value for key, value in quiver_parameters.items() if not value is None}

    fx = fx.astype("float64")
    fy = fy.astype("float64")
    dims = fx.shape  # needed for scaling
    if not isinstance(ax, matplotlib.axes.Axes):
        fig = plt.figure(figsize=figsize)
        ax = plt.axes()
    map_values = np.sqrt(fx ** 2 + fy ** 2)
    vmin, vmax = set_vmin_vmax(map_values, vmin, vmax)
    im = plt.imshow(map_values, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha, origin=ax_origin)  # imshowing
    if plot_style == "clickpoints":
        ax.set_position([0, 0, 1, 1])
    ax.set_axis_off()
    # plotting arrows
    fx, fy, xs, ys = filter_values(fx, fy, abs_filter=filter[0],
                                   f_dist=filter[1])  # filtering every n-th value and every value smaller then x
    if scale_ratio:  # optional custom scaling with the image axis lenght
        fx, fy = scale_for_quiver(fx, fy, dims=dims, scale_ratio=scale_ratio)
        quiver_parameters["scale"] = 1  # disabeling the auto scaling behavior of quiver
    plt.quiver(xs, ys, fx, fy, **quiver_parameters)  # plotting the arrows
    if plot_cbar:
        add_colorbar(vmin, vmax, cmap, ax=ax, cbar_style=cbar_style, cbar_width=cbar_width, cbar_height=cbar_height,
                     cbar_borderpad=cbar_borderpad, v=cbar_tick_label_size, cbar_str=cbar_str,
                     cbar_axes_fraction=cbar_axes_fraction, cbar_title_pad=cbar_title_pad)
    return fig, ax

def scale_for_quiver(ar1, ar2, dims, scale_ratio=0.2, return_scale=False):
    scale = scale_ratio * np.max(dims) / np.nanmax(np.sqrt((ar1) ** 2 + (ar2) ** 2))
    if return_scale:
        return scale
    return ar1 * scale, ar2 * scale


def add_colorbar(vmin, vmax, cmap="rainbow", ax=None, cbar_style="not-clickpoints", cbar_width="2%",
                 cbar_height="50%", cbar_borderpad=0.1, cbar_tick_label_size=15, cbar_str="",
                 cbar_axes_fraction=0.2, shrink=0.8, aspect=20, cbar_title_pad=1, **kwargs):
    from contextlib import suppress
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=matplotlib.cm.get_cmap(cmap), norm=norm)
    sm.set_array([])  # bug fix for lower matplotlib version
    if cbar_style == "clickpoints":  # colorbar inside of the plot
        cbaxes = inset_axes(ax, width=cbar_width, height=cbar_height, loc=5, borderpad=cbar_borderpad * 30)
        cb0 = plt.colorbar(sm, cax=cbaxes)
        with suppress(TypeError, AttributeError):
            cbaxes.set_title(cbar_str, color="white", pad=cbar_title_pad)
        cbaxes.tick_params(colors="white", labelsize=cbar_tick_label_size)
    else:  # colorbar outide of the plot
        cb0 = plt.colorbar(sm, aspect=aspect, shrink=shrink, fraction=cbar_axes_fraction,
                           pad=cbar_borderpad)  # just exploiting the axis generation by a plt.colorbar
        cb0.outline.set_visible(False)
        cb0.ax.tick_params(labelsize=cbar_tick_label_size)
        with suppress(TypeError, AttributeError):
            cb0.ax.set_title(cbar_str, color="black", pad=cbar_title_pad)
    return cb0


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




def segment_cell(img, thres=1, gaus1 = 8, gaus2=80, iterartions=1,show_segmentation = True, segmention="otsu"):   
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





# maxprojection Data
# read in list of cells and list of fibers to evaluate (must be in same order)
# use glob.glob or individual list of paths []    

fiber_list = natsorted(glob.glob(r"..\fiber.tif"))   #fiber_random_c24
cell_list = natsorted(glob.glob(r"..\cell.tif"))   #check that order is same to fiber

scale =  0.318 # um per pixel


# Set Parameters 
sigma_tensor = 7/scale  # sigma of applied gauss filter / window for structure tensor analysis in px
                    # should be in the order of the objects to analyze !! 
                    # 7 um for collagen 
edge = 40   # Cutt of pixels at the edge since values at the border cannot be trusted
segmention_thres = 1  # for cell segemetntion, thres 1 equals normal otsu threshold , user also can specify gaus1 + gaus2 in segmentation if needed
sigma_first_blur  = 0.5 # slight first bluring of whole image before using structure tensor
angle_sections = 5   # size of angle sections in degree 
shell_width =  5/scale   # pixel width of distance shells
manual_segmention = False

# create output folder accordingly
out_list = [os.path.join("test_data_output", os.path.basename(cell_list[i])[:-4]) for i in range(len(cell_list))]





# loop thorugh cells
for n,i in tqdm(enumerate(fiber_list)):
    #create output folder if not existing
    if not os.path.exists(out_list[n]):
        os.makedirs(out_list[n])
    # load images
    im_cell  = color.rgb2gray(imageio.imread(cell_list[n]))
    im_fiber = color.rgb2gray(imageio.imread(fiber_list[n]))
    #"contrast spreading" by setting all values below norm1-percentile to zero and
    # # all values above norm2-percentile to 1
    norm1 = 5
    norm2 = 95
    # # applying normalizing/ contrast spreading
    im_cell_n = normalize(im_cell, norm1, norm2)
    im_fiber_n = normalize(im_fiber, norm1, norm2)  
    im_fiber_g = gaussian(im_fiber_n, sigma=sigma_first_blur)     # blur fiber image slightly (test with local gauss - similar)
    
    # segment cell (either manual or automatically)
    if manual_segmention:
        segmention = custom_mask(im_cell_n)
    else:
        segmention = segment_cell(im_cell_n, thres= segmention_thres)    # thres 1 equals normal otsu threshold
    
    # center incropped image (to avoid edge effects)
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
    if edge is not 0:
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

    # Angular deviation from orietation to center vector
    angle_dev = np.arccos(np.abs(dx_norm * min_evec[:,:,0] + dy_norm*min_evec[:,:,1])) * 360/(2*np.pi)
    # weighting by coherence
    angle_dev_weighted = (angle_dev * ori) / np.nanmean(ori)     # no angle values anymore but the mean later is again an angle
    # weighting by coherence and image intensity
    im_fiber_g = im_fiber_g[edge:-edge,edge:-edge]
    # could also use non filtered image
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4497681/ these guys
    # use a threshold in the intensity image// only coherence and orientation vectors
    # corresponding to pixels in the intesity iamage above a threshold are considered.
    weight_image = gaussian(im_fiber_g,sigma=15)
    angle_dev_weighted2 = (angle_dev_weighted *weight_image) / np.nanmean(weight_image)
    # also weighting the coherency like this
    ori_weight2 = (ori * weight_image) / np.nanmean(weight_image)
    
 
    # GRADIENT towards center   
    grad_y = np.gradient(im_fiber_g, axis=0)
    grad_x = np.gradient(im_fiber_g, axis=1)
    dx_norm_a = -dy_norm.copy()
    dy_norm_a = dx_norm.copy()
    s_to_center = ((grad_x * dx_norm) + (grad_y * dy_norm))**2
    s_around_center = ((grad_x * dx_norm_a) + (grad_y * dy_norm_a))**2
    s_norm1 = (s_around_center - s_to_center)/(grad_x**2 + grad_y**2)
    #s_norm2 = (s_around_center - s_to_center)/(s_around_center + s_to_center)
    
    
    # save values for total image analysis
    # Total Value for complete image (without the mask)
    # averages
    alpha_dev_total1 = np.nanmean(angle_dev[(~segmention["mask"][edge:-edge,edge:-edge])])
    alpha_dev_total2 = np.nanmean(angle_dev_weighted[(~segmention["mask"][edge:-edge, edge:-edge])])
    alpha_dev_total3 = np.nanmean(angle_dev_weighted2[(~segmention["mask"][edge:-edge, edge:-edge])])
    cos_dev_total1 = np.nanmean(np.cos(2*angle_dev[(~segmention["mask"][edge:-edge, edge:-edge])]*np.pi/180))
    cos_dev_total2 = np.nanmean(np.cos(2*angle_dev_weighted[(~segmention["mask"][edge:-edge, edge:-edge])]*np.pi/180))
    cos_dev_total3 = np.nanmean(np.cos(2*angle_dev_weighted2[(~segmention["mask"][edge:-edge, edge:-edge])]*np.pi/180))
    coh_total = np.nanmean(ori[(~segmention["mask"][edge:-edge,edge:-edge]) ])
    coh_total2 = np.nanmean(ori_weight2[(~segmention["mask"][edge:-edge, edge:-edge])])
    # save to txt file
    values = [coh_total ,coh_total2, alpha_dev_total1, alpha_dev_total2, alpha_dev_total3, 
              2*np.cos(alpha_dev_total1*np.pi/180)-1, 2*np.cos(alpha_dev_total2*np.pi/180)-1, 2*np.cos(alpha_dev_total3*np.pi/180)-1]
    strings = ["mean_coh.txt", "mean_coh_w_int.txt", "mean_angle.txt",
      "mean_angle_we_coh.txt", "mean_angle_we_coh_int.txt", 
      "cos2a_mean.txt","cos2a_mean_we_coh.txt", "cos2a_we_coh_int.txt"]
    for i,v in enumerate(values):
        np.savetxt(os.path.join(out_list[n],strings[i]), [v] ) 
        

   
    """
    Angular sections
    """
    ori_angle = []
    ori_mean = []
    ori_mean_weight = []
    int_mean = []
    proj_angle = []
    grad_slice = []
    alpha_dev_slice = []
    alpha_dev_slice_weight = []
    alpha_dev_slice_weight2 = []
    
    # make the angle analysis in sections
    ang_sec = angle_sections
    for alpha in range(-180, 180, ang_sec):
            mask_angle = (angle > (alpha-ang_sec/2)) & (angle <= (alpha+ang_sec/2)) & (~segmention["mask"][edge:-edge,edge:-edge])
            if alpha == -180:
                  mask_angle = ((angle > (180-ang_sec/2)) | (angle <= (alpha+ang_sec/2))) & (~segmention["mask"][edge:-edge,edge:-edge])
            if alpha == 180:
                  mask_angle = ((angle > (alpha-ang_sec/2)) | (angle <= (-180 +ang_sec/2))) & (~segmention["mask"][edge:-edge,edge:-edge])            
            ori_angle.append(alpha)
            ori_mean.append(np.nanmean(ori[mask_angle]))
            ori_mean_weight.append(np.nanmean(ori_weight2[mask_angle]))
            int_mean.append(np.nanmean(im_fiber_g[mask_angle]))
            grad_slice.append(np.nanmean(s_norm1[mask_angle]))
            alpha_dev_slice.append(np.nanmean(angle_dev[mask_angle]))
            alpha_dev_slice_weight.append(np.nanmean(angle_dev_weighted[mask_angle]))
            alpha_dev_slice_weight2.append(np.nanmean(angle_dev_weighted2[mask_angle]))


    # translating the angles to coordinates in polar plot
    angle_plotting1 = (np.array(ori_angle) * np.pi / 180)
    angle_plotting = angle_plotting1.copy()
    angle_plotting[angle_plotting1 < 0] =  np.abs(angle_plotting[angle_plotting1 < 0])
    angle_plotting[angle_plotting1 > 0] =  np.abs(angle_plotting[angle_plotting1>0] - 2* np.pi)
        
    # save results 
    np.savetxt(os.path.join(out_list[n],"angle_plotting.txt"), angle_plotting)
    np.savetxt(os.path.join(out_list[n],"alpha_dev_slice.txt"), alpha_dev_slice)
    np.savetxt(os.path.join(out_list[n],"alpha_dev_slice_weight.txt"), alpha_dev_slice_weight)
    np.savetxt(os.path.join(out_list[n],"alpha_dev_slice_weight2.txt"), alpha_dev_slice_weight2)
    np.savetxt(os.path.join(out_list[n],"alpha_ori_mean_weight.txt"), ori_mean_weight)
    np.savetxt(os.path.join(out_list[n],"alpha_grad_slice.txt"), grad_slice)


    """
    Distance Evaluation
    """

    shells = np.arange(0, dist_surface.max(), shell_width)
    midofshells = (shells + shell_width/2)[:-1]
    allshellmasks = []
    # intensity and angle within individual shell and accumulation of all inner shells
    # distance shells parallel to surface
    dist_int_accum = []
    dist_angle_accum = []
    dist_int_individ = []
    dist_angle_individ = []
    # distance shells as circles around center
    dist_int_accum_center = []
    dist_angle_accum_center = []
    dist_int_individ_center = []
    dist_angle_individ_center = []

    
    # make the distance shell analysis
    for i in range(len(shells)-1):
        # distance shells parallel to surface
        # mask of individual shells and accumulation of all points closer to the correponding cell
        mask_shell = (dist_surface > (shells[i])) & (dist_surface <= (shells[i+1])) & (~segmention["mask"][edge:-edge,edge:-edge])  
        mask_shell_lower=  (dist_surface <= (shells[i+1])) & (~segmention["mask"][edge:-edge,edge:-edge])
        allshellmasks.append(mask_shell)
        # calculate mintensity and angle deviation within the growing shells (always within start shell to highest shell)
        # THINK ABOUT ANGLE DEV W 2 (with int here ?)
        dist_angle_accum.append(np.nanmean(angle_dev_weighted[mask_shell_lower])  ) # accumulation of lower shells
        dist_angle_individ.append(np.nanmean(angle_dev_weighted[mask_shell])  )    # exclusively in certain shell
        # MAYBE TODO:  weight by coherency+Intensity   within shell instead of the pure angle ?
        # mean intensity
        dist_int_accum.append(np.nanmean(im_fiber_g[mask_shell_lower]))          # accumulation of lower shells
        dist_int_individ.append(np.nanmean(im_fiber_g[mask_shell])  )    # exclusively in certain shell
        
        # distance shells as circles around center
        mask_shell_center = (distance > (shells[i])) & (distance <= (shells[i+1])) & (~segmention["mask"][edge:-edge,edge:-edge])  
        mask_shell_lower_center=  (distance <= (shells[i+1])) & (~segmention["mask"][edge:-edge,edge:-edge])
        # calculate mintensity and angle deviation within the growing shells (always within start shell to highest shell)
        # THINK ABOUT ANGLE DEV W 2 (with int here ?)
        dist_angle_accum_center.append(np.nanmean(angle_dev_weighted[mask_shell_lower_center])  ) # accumulation of lower shells
        dist_angle_individ_center.append(np.nanmean(angle_dev_weighted[mask_shell_center])  )    # exclusively in certain shell
        # MAYBE TODO:  weight by coherency+Intensity   within shell instead of the pure angle ?
        # mean intensity
        dist_int_accum_center.append(np.nanmean(im_fiber_g[mask_shell_lower_center]))          # accumulation of lower shells
        dist_int_individ_center.append(np.nanmean(im_fiber_g[mask_shell_center])  )    # exclusively in certain shell
        

        
    # norm intensities   (Baseline: mean intensity of the 2 outmost shells)
    dist_int_individ_norm = np.array(dist_int_individ)/ np.nanmean(np.array(dist_int_individ[-2:]))    
    dist_int_accum_norm = np.array(dist_int_accum)/ np.nanmean(np.array(dist_int_accum[-2:]))    
    dist_int_individ_center_norm = np.array(dist_int_individ_center)/ np.nanmean(np.array(dist_int_individ_center[-2:])) 
    dist_int_accum_center_norm = np.array(dist_int_accum_center)/ np.nanmean(np.array(dist_int_accum_center[-2:])) 
    # Calculate value where ntensity drops 25%
    distintdrop = np.abs(dist_int_individ_norm-0.75)
    # distance where int  drops   to 75% 
    halflife_int =  midofshells[np.where(distintdrop == np.nanmin(distintdrop.min()))] [0] 
    # if decrease is not within range (minimum equals last value) then set to nan
    if halflife_int == midofshells[-1]:
        halflife_int = np.nan

    # # Calculate value where orientation drops to 75% within maxorientation(min) to 45° (random) range 
    # difference to 45 degree instead of min-max range    
    # calculate halflife of maximal orientation over distance
    # difference to 45 degree for all
    diffdist = np.array(dist_angle_individ)-45
    # maximal orientation
    diffmax = np.nanmin(diffdist)
    diffmax_pos = np.where(diffmax==diffdist)[0][0]
    # difference  angle drops to 75% 
    diff2 = np.abs(diffdist-(0.75*diffmax))
    diff2[:diffmax_pos] = np.nan    # only look at distances on the right side /further out 
    halflife_ori =  midofshells[np.where(diff2 == np.nanmin(diff2))]    
    # if decrease is not within range (minimum equals last value) then set to nan
    if halflife_ori == midofshells[-1]:
        halflife_ori = np.nan
        
    
    # save distane arrays
    np.savetxt(os.path.join(out_list[n],"shells-mid_px.txt"), midofshells)
    np.savetxt(os.path.join(out_list[n],"dist_int_individ.txt"), dist_int_individ)
    np.savetxt(os.path.join(out_list[n],"dist_angle_individ.txt"), dist_angle_individ)
    np.savetxt(os.path.join(out_list[n],"dist_int_accum.txt"), dist_int_accum)
    np.savetxt(os.path.join(out_list[n],"dist_angle_accum.txt"), dist_angle_accum)
    np.savetxt(os.path.join(out_list[n],"distdrop25_ori_px.txt"), [halflife_ori])
    np.savetxt(os.path.join(out_list[n],"distdrop25_int_px.txt"), [halflife_int])
    np.savetxt(os.path.join(out_list[n],"dist_int_individ_center.txt"), dist_int_individ_center)
    np.savetxt(os.path.join(out_list[n],"dist_angle_individ_center.txt"), dist_angle_individ_center)
    np.savetxt(os.path.join(out_list[n],"dist_int_accum_center.txt"), dist_int_accum_center)
    np.savetxt(os.path.join(out_list[n],"dist_angle_accum_center.txt"), dist_angle_accum_center)
    np.savetxt(os.path.join(out_list[n],"dist_int_individ_norm.txt"), dist_int_individ_norm)
    np.savetxt(os.path.join(out_list[n],"dist_int_accum_norm.txt"), dist_int_accum_norm)
    np.savetxt(os.path.join(out_list[n],"dist_int_individ_norm_center.txt"), dist_int_individ_center_norm)
    np.savetxt(os.path.join(out_list[n],"dist_int_accum_norm_center.txt"), dist_int_accum_center_norm)
    
    
    try:
        np.savetxt(os.path.join(out_list[n],"meanangle_within10shells.txt"), [dist_angle_accum[9]]) 
    except:
        pass

    
    """
    save plots here
    """
    
    # angle deviation no weights
    plt.figure();plt.imshow(angle_dev); plt.colorbar()
    mx = min_evec[:,:,0] * ori
    my = min_evec[:,:,1] * ori
    mx, my, x, y = filter_values(mx, my, abs_filter=0,
                                   f_dist=15)  #
    plt.quiver(x, y, mx*300, my*300,scale=1,scale_units="xy", angles="xy")
    plt.savefig(os.path.join(out_list[n],"angle_dev_quiv.png"), dpi=200)
    
     # angle deviation weights intensity + coherency
    plt.figure();plt.imshow(angle_dev_weighted2); plt.colorbar()
    mx = min_evec[:,:,0] * ori
    my = min_evec[:,:,1] * ori
    mx, my, x, y = filter_values(mx, my, abs_filter=0,
                                   f_dist=15)  #
    plt.quiver(x, y, mx*300, my*300,scale=1,scale_units="xy", angles="xy")
    plt.savefig(os.path.join(out_list[n],"angle_dev_quiv_weight_i_c.png"), dpi=200)
    
    # pure coherency
    plt.figure();plt.imshow(ori); plt.colorbar();  
    plt.savefig(os.path.join(out_list[n],"coherency.png"), dpi=200)
    
     # Angular deviation from orietation to center vector
    #angle_dev= np.arccos(np.abs(dx_norm*min_evec[:,:,1][edge:-edge,edge:-edge] + dy_norm*min_evec[:,:,0][edge:-edge,edge:-edge] ))  *360/(2*np.pi)
    plt.figure();plt.imshow(angle_dev, origin="upper", cmap="viridis");plt.colorbar(); plt.savefig(os.path.join(out_list[n],"angle_dev.png"), dpi=200)

    # test to appreciate how the angles work in plot
    # a = np.linspace(np.pi, np.pi * 2, len(angle_plotting))
    # b = np.linspace(0, 1, len(angle_plotting))
    # plt.figure()
    # ax = plt.subplot(111, projection="polar")
    # ax.plot(angle_plotting, b, label="angle_plotting")
    # ax.plot((np.array(ori_angle) * np.pi / 180), b, label="ori_angle directly")
    # plt.legend()
    # plt.title("illustration of how angles in the polar plot work")
    
    
    


    plt.figure(figsize=(5,5))
    axs1 = plt.subplot(111, projection="polar")
    axs1.plot(angle_plotting, ori_mean_weight, label="Allignment Collagen" , linewidth=2, c = "C0")
    plt.savefig(os.path.join(out_list[n],"orientation_w.png"), dpi=200)
 
    # Triple plot
    plt.figure(figsize=(20,6))
    axs1 = plt.subplot(131, projection="polar")
    axs1.plot(angle_plotting, ori_mean, label="Allignment Collagen" )
    axs1.plot(angle_plotting, ori_mean_weight, label="Allignment Collagen weighted with intesity" )
    #axs1.plot(angle_plotting,  int_mean, c = "C1" , label="Intensity Collagen")
    plt.legend(fontsize=12)
    ax2 = plt.subplot(132, projection="polar")
    ax2.plot(angle_plotting, alpha_dev_slice , label="no weights")
    ax2.plot(angle_plotting, alpha_dev_slice_weight , label="coherence weighting")
    ax2.plot(angle_plotting, alpha_dev_slice_weight2, label="coherence and intensity weighting")
    #ax.set_rlim(bottom=90, top=0)  # put allignet values further out for visualization
    plt.legend(fontsize=12)
    strings = ["mean coherency", "mean_coherency\nweighted by intensity", "mean angle",
               "mean angle weighted\nby coherency", "mean angle weighted\nby coherency and intensity"]
    values = [coh_total ,coh_total2, alpha_dev_total1, alpha_dev_total2, alpha_dev_total3]
    values = [[str(np.round(x,4))] for x in values]
    table_text = [[strings[i], values[i]] for i in range(len(values))]
    ax3 = plt.subplot(133)
    ax3.axis('tight')
    ax3.axis('off')
    ax3.table(cellText = values, rowLabels=strings,bbox=[0.6,0.2,0.7,0.9])
    #plt.tight_layout()
    plt.savefig(os.path.join(out_list[n],"orientation_triple.png"), dpi=200)
    
    
    f = np.nanpercentile(ori,0.75)
    fig5, ax5 = show_quiver (min_evec[:,:,0] * ori, min_evec[:,:,1] * ori, filter=[f, 15], scale_ratio=0.1,width=0.003, cbar_str="coherency", cmap="viridis")
    ax5.plot(center_small[0],center_small[1],"o")
    plt.savefig(os.path.join(out_list[n],"coh_quiver.png"), dpi=200)
    
    
    # plot max +seg
    fig7= plt.figure() 
    my_norm = matplotlib.colors.Normalize(vmin=0.99, vmax=1, clip=False)  
    cmap =  copy(plt.get_cmap('Greys'))
    # everything under vmin gets transparent (all zeros in mask)
    cmap.set_under('k', alpha=0)
    #everything else visible
    cmap.set_over('k', alpha=1)
    # plot mask and center
    plt.imshow(normalize(im_fiber_n[edge:-edge,edge:-edge]), origin="upper")
    plt.imshow(segmention["mask"][edge:-edge,edge:-edge], cmap=cmap, norm = my_norm, origin="upper")
    center_small = (segmention["centroid"][0]-edge,segmention["centroid"][1]-edge)
    plt.scatter(center_small[0],center_small[1], c= "w")
    plt.tight_layout()
    plt.savefig(os.path.join(out_list[n],"seg-max.png"), dpi=200)
    
    # plot overlay
    my_norm = matplotlib.colors.Normalize(vmin=0.99, vmax=1, clip=False)  
    cmap =  copy(plt.get_cmap('Greys'))
    # everything under vmin gets transparent (all zeros in mask)
    cmap.set_under('k', alpha=0)
    #everything else visible
    cmap.set_over('k', alpha=1)
    # plot mask and center
    show_quiver (min_evec[:,:,0] * ori, min_evec[:,:,1] * ori, filter=[f, 15],alpha=0 , scale_ratio=0.1,width=0.002, plot_cbar=False, cbar_str="coherency", cmap="viridis")
    plt.imshow(normalize(im_fiber_n[edge:-edge,edge:-edge]), origin="upper")
    plt.imshow(segmention["mask"][edge:-edge,edge:-edge], cmap=cmap, norm = my_norm, origin="upper")
    center_small = (segmention["centroid"][0]-edge,segmention["centroid"][1]-edge)
    plt.scatter(center_small[0],center_small[1], c= "w")
    plt.tight_layout()
    plt.savefig(os.path.join(out_list[n],"struc-tens-o.png"), dpi=200)
    
    
    # plot shells
    plt.figure()
    cmap_list = ["Greens","Greys","Reds","Oranges","Blues","PuBu","GnBu"]
    for s in  range(len(allshellmasks)):
        my_norm = matplotlib.colors.Normalize(vmin=0.99, vmax=1, clip=False)  
        cmap =  copy(plt.get_cmap(cmap_list[s%len(cmap_list)]))
        # everything under vmin gets transparent (all zeros in mask)
        cmap.set_under('k', alpha=1)
        #everything else visible
        cmap.set_over('k', alpha=0)
        # plot mask and center
        plt.imshow(allshellmasks[-s], cmap = cmap ,  origin="upper", alpha= 0.2 ) #- 0.2* s/len(allshellmasks) )
    plt.savefig(os.path.join(out_list[n],"shells.png"), dpi=200)   
        
     # plot distance shell analysis    
    plt.figure(figsize=(7,3))
    plt.subplot(121)    
    plt.plot(midofshells,dist_angle_individ,"o-", c="lightgreen",label="orientation")
    plt.plot([halflife_ori,halflife_ori],[np.min(dist_angle_individ),np.max(dist_angle_individ)], c="orange", linestyle="--")
    plt.grid()
    plt.tight_layout()
    plt.xlabel("distance (px)")
    plt.ylabel("orientation")
    plt.subplot(122)    
    plt.plot(midofshells,dist_int_individ_norm,"o-", c="plum", label="intensity")
    plt.grid()
    plt.plot([halflife_int,halflife_int],[np.min(dist_int_individ_norm),np.max(dist_int_individ_norm)], c="orange", linestyle="--")
    plt.xlabel("distance (px)")
    plt.ylabel("intensity")
    plt.tight_layout()
    plt.savefig(os.path.join(out_list[n],"distance-shell-individ.png"), dpi=200)
    
    # plot distance shell analysis    
    plt.figure(figsize=(7,3))
    plt.subplot(121)    
    plt.plot(midofshells,dist_angle_individ_center,"o-", c="lightgreen",label="orientation")
    plt.plot([halflife_ori,halflife_ori],[np.min(dist_angle_individ_center),np.max(dist_angle_individ_center)], c="orange", linestyle="--")
    plt.grid()
    plt.tight_layout()
    plt.xlabel("distance (px)")
    plt.ylabel("orientation")
    plt.subplot(122)    
    plt.plot(midofshells,dist_int_individ_center_norm,"o-", c="plum", label="intensity")
    plt.grid()
    plt.plot([halflife_int,halflife_int],[np.min(dist_int_individ_center_norm),np.max(dist_int_individ_center_norm)], c="orange", linestyle="--")
    plt.xlabel("distance (px)")
    plt.ylabel("intensity")
    plt.tight_layout()
    plt.savefig(os.path.join(out_list[n],"distance-shell-individ_center.png"), dpi=200)
        
    # plot distance shell analysis    
    plt.figure(figsize=(7,3))
    plt.subplot(121)    
    plt.plot(midofshells,dist_angle_accum,"o-", c="lightgreen",label="orientation")
    #plt.plot([halflife_ori,halflife_ori],[np.min(dist_angle),np.max(dist_angle)], c="orange", linestyle="--")
    plt.grid()
    plt.tight_layout()
    plt.xlabel("distance (px)")
    plt.ylabel("orientation")
    plt.subplot(122)    
    plt.plot(midofshells,dist_int_accum_norm,"o-", c="plum", label="intensity")
    plt.grid()
    #plt.plot([halflife_int,halflife_int],[np.min(dist_int),np.max(dist_int)], c="orange", linestyle="--")
    plt.xlabel("distance (px)")
    plt.ylabel("intensity")
    plt.tight_layout()
    plt.savefig(os.path.join(out_list[n],"distance-shell-accum.png"), dpi=200)
   
    
     # plot overlay version 2
    my_norm = matplotlib.colors.Normalize(vmin=0.99, vmax=1, clip=False)  
    cmap =  copy(plt.get_cmap('Greys'))
    # everything under vmin gets transparent (all zeros in mask)
    cmap.set_under('k', alpha=0)
    #everything else visible
    cmap.set_over('k', alpha=1)
    # plot mask and center
    show_quiver (min_evec[:,:,0] * ori, min_evec[:,:,1] * ori, filter=[0, 8],alpha=0 , scale_ratio=0.1,width=0.0012, plot_cbar=False, cbar_str="coherency", cmap="viridis")
    plt.imshow(normalize(im_fiber_n[edge:-edge,edge:-edge]), origin="upper")
    plt.imshow(segmention["mask"][edge:-edge,edge:-edge], cmap=cmap, norm = my_norm, origin="upper")
    center_small = (segmention["centroid"][0]-edge,segmention["centroid"][1]-edge)
    plt.scatter(center_small[0],center_small[1], c= "w")
    plt.tight_layout()
    plt.savefig(os.path.join(out_list[n],"struc-tens-o2.png"), dpi=200)
    
    plt.figure()
    plt.imshow(im_cell_n)
    plt.savefig(os.path.join(out_list[n],"cell-raw.png"), dpi=200)
    
    
    # fig7= plt.figure() 
    # my_norm = matplotlib.colors.Normalize(vmin=0.99, vmax=1, clip=False)  
    # cmap =  copy(plt.get_cmap('Greys'))
    # # everything under vmin gets transparent (all zeros in mask)
    # cmap.set_under('k', alpha=0)
    # #everything else visible
    # cmap.set_over('k', alpha=1)
    # # plot mask and center
    # plt.subplot(121)
    # plt.imshow(dist_surface, cmap=cmap, norm = my_norm, origin="upper")
    # plt.subplot(122)
    # plt.imshow(normalize(im_fiber_n[edge:-edge,edge:-edge]), origin="upper")
    
    
    
    plt.close("all")