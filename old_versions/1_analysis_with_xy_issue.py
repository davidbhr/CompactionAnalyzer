"""
Created on Wed Jul  1 15:45:34 2020

@author: david boehringer

Evaluates the fiber orientation around cells that compact collagen tissue.
Applying the Structure tensor (Script Anreas Bauer) and evaluates the orientation within 
angle sections around the segmented cell center.

Needs an Image of cell and one image of the fibers

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
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import remove_small_objects
from skimage.exposure import adjust_gamma   
from PIL import Image
from copy import copy
import imageio
from skimage import color

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


def segment_cell(img, thres=1):
    """
    Image segmentation function to create  mask, radius, and position of a spheroid in a grayscale image.
    Args:
        img(array): Grayscale image as a Numpy array
        thres(float): To adjust the segmentation
    Returns:
        dict: Dictionary with keys: mask, radius, centroid (x/y)
    """
    height = img.shape[0]
    width = img.shape[1]
    # local gaussian   
    img = gaussian(img, sigma=12) - gaussian(img, sigma=40)
    # segment cell
    mask = img > threshold_otsu(img) * thres    #[::-1]
    # remove other objects
    
    mask = scipy_morph.binary_closing(mask, iterations=3)
    mask = remove_small_objects(mask, min_size=2000)
    mask = scipy_morph.binary_dilation(mask, iterations=3)
    mask = scipy_morph.binary_fill_holes(mask)
    

    # identify spheroid as the most centered object
    labeled_mask, max_lbl = scipy_meas.label(mask)
    center_of_mass = np.array(scipy_meas.center_of_mass(mask, labeled_mask, range(1, max_lbl + 1)))
    distance_to_center = np.sqrt(np.sum((center_of_mass - np.array([height / 2, width / 2])) ** 2, axis=1))

    mask = (labeled_mask == distance_to_center.argmin() + 1)

    # determine radius of spheroid
    radius = np.sqrt(np.sum(mask) / np.pi)

    # determine center position of spheroid
    cy, cx = center_of_mass[distance_to_center.argmin()]

    # return dictionary containing spheroid information
    return {'mask': mask, 'radius': radius, 'centroid': (cx, cy)}



# read in cells to evaluate
# maxproj
fiber_list = glob.glob(r"test_data\fiber.tif")   #fiber_random_c24
cell_list = glob.glob(r"test_data\cell.tif")   #check that order is same to fiber

# create output folder accordingly
#out_list = [os.path.join("angle_eval","2-angle",cell_list[i].split(os.sep)[0], os.path.basename(cell_list[i])[:-4]) for i in range(len(cell_list))]
out_list = [os.path.join("analysis",cell_list[i].split(os.sep)[0], os.path.basename(cell_list[i])[:-4]) for i in range(len(cell_list))]

# loop thorugh cells
for n,i in tqdm(enumerate(fiber_list)):
    
    #for testing take certain image only
    #n=60   #31
    
    
    # create output folder if it does not exist, print warning otherwise
    if not os.path.exists(out_list[n]):
        os.makedirs(out_list[n])
    else:
        print('WARNING: Output folder already exists! ({})'.format(out_list[n]))
    
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

    # segment cell
    segmention = segment_cell(im_cell_n, thres=1) # thres 1 equals normal otsu threshold

    
    # save individual plots 
    fig1, ax1 = plt.figure(),plt.imshow(im_cell_n,origin="lower")
    plt.tight_layout()
    fig1.savefig(os.path.join(out_list[n],"cell.png"), dpi=200)
    fig2, ax2 = plt.figure(),plt.imshow(im_fiber_n,origin="lower")
    plt.tight_layout()
    fig2.savefig(os.path.join(out_list[n],"fiber.png"), dpi=200)
    fig3, ax3 = plt.figure(), (plt.imshow(segmention["mask"], alpha= 1, cmap="Reds_r",origin="lower"), plt.scatter(segmention["centroid"][0],segmention["centroid"][1]))
    plt.tight_layout()
    fig3.savefig(os.path.join(out_list[n],"mask.png"), dpi=200)

    
    # create overlay plot
    fig4, ax4 = plt.figure(), plt.imshow(im_fiber_n,origin="lower")
    # specify color range vmin shortlz above zero
    my_norm = matplotlib.colors.Normalize(vmin=0.99, vmax=1, clip=False)  
    cmap =  copy(plt.get_cmap('Greys'))
    # everzthing under vmin gets transparent (all zeros in mask)
    cmap.set_under('k', alpha=0)
    #everzthing else visible
    cmap.set_over('k', alpha=1)
    # plot mask and center
    plt.imshow(segmention["mask"], cmap=cmap, norm = my_norm,origin="lower")
    plt.scatter(segmention["centroid"][0],segmention["centroid"][1], c= "w")
    plt.tight_layout()       
    fig4.savefig(os.path.join(out_list[n],"overlay-seg-fiber.png"), dpi=200)   #ntpath.abspath(
    
    """
    Structure tensor
    """
    # Structure Tensor Orientation
    # blur fiber image slightly (trz local gauss - similar)
    im_fiber_g = gaussian(im_fiber_n, sigma=0.5) 
    # get structure tensor
    ori, max_evec, min_evec, max_eval, min_eval = analyze_local(im_fiber_g, sigma=15, size=0, filter_type="gaussian")
    f = np.percentile(ori,0.75)
    # cutt off the edges    
    edge = 40
    fig5, ax5 =  show_quiver (min_evec[:,:,0][edge:-edge,edge:-edge] * ori[edge:-edge,edge:-edge], min_evec[:,:,1][edge:-edge,edge:-edge] * ori[edge:-edge,edge:-edge],filter=[f, 15], scale_ratio=0.1,width=0.003, cbar_str="coherency", cmap="viridis")
    plt.tight_layout()
    fig5.savefig(os.path.join(out_list[n],"struc-tens.png"), dpi=200)


    fig6 = plt.figure() 
    show_quiver (min_evec[:,:,0][edge:-edge,edge:-edge] * ori[edge:-edge,edge:-edge], min_evec[:,:,1][edge:-edge,edge:-edge] * ori[edge:-edge,edge:-edge], filter=[f, 15],alpha=0 , scale_ratio=0.1,width=0.002, cbar_str="coherency", cmap="viridis")
    plt.imshow(im_fiber_n[edge:-edge,edge:-edge],origin="lower")
    plt.tight_layout()
    plt.savefig(os.path.join(out_list[n],"struc-tens-oy.png"), dpi=200)
    
    
    fig7= plt.figure() 
    my_norm = matplotlib.colors.Normalize(vmin=0.99, vmax=1, clip=False)  
    cmap =  copy(plt.get_cmap('Greys'))
    # everything under vmin gets transparent (all zeros in mask)
    cmap.set_under('k', alpha=0)
    #everything else visible
    cmap.set_over('k', alpha=1)
    # plot mask and center
    show_quiver (min_evec[:,:,0][edge:-edge,edge:-edge] * ori[edge:-edge,edge:-edge], min_evec[:,:,1][edge:-edge,edge:-edge] * ori[edge:-edge,edge:-edge], filter=[f, 15],alpha=0 , scale_ratio=0.1,width=0.002, cbar_str="coherency", cmap="viridis")
    plt.imshow(im_fiber_n[edge:-edge,edge:-edge], origin="lower")
    plt.imshow(segmention["mask"][edge:-edge,edge:-edge], cmap=cmap, norm = my_norm, origin="lower")
    
    center_small = (segmention["centroid"][0]-edge,segmention["centroid"][1]-edge)
    plt.scatter(center_small[0],center_small[1], c= "w")
    plt.tight_layout()
    plt.savefig(os.path.join(out_list[n],"struc-tens-o2.png"), dpi=200)



    

    """
    ANGLE ANALYSIS + gradient 
    """
    
    # methode 1 Angle deviation to center
    # winkel zwischen min evec und dr vektor (dx,dy) (normiert !)
    # gewichtet mit coherencecy in winkelsegmenten mitteeln
    
    # methode 2 : Gradient to center
    # wie stark ist gradient zum zentrum ausgerichtet

    # Calculate Angle + Distances
    y,x = np.indices(ori[edge:-edge,edge:-edge].shape)
    dx = x - center_small[0]
    dy = y - center_small[1]
    distance = np.sqrt(dx ** 2 + dy ** 2)
    angle = np.arctan2(dy, dx) *360/(2*np.pi)
    #plt.imshow(distance)
    #plt.imshow(angle)
    dx_norm = dx/distance
    dy_norm = dy/distance
    
 
    # Angular deviation from orietation to center vector
    angle_dev= np.arccos(np.abs(dx_norm*min_evec[:,:,1][edge:-edge,edge:-edge] + dy_norm*min_evec[:,:,0][edge:-edge,edge:-edge] ))  *360/(2*np.pi)
    plt.figure();plt.imshow(angle_dev, origin="lower");plt.colorbar(); plt.savefig(os.path.join(out_list[n],"angle_dev.png"), dpi=200)

    
    # GRADIENT    
    grad_y = np.gradient(im_fiber_g, axis=0)  [edge:-edge,edge:-edge]
    grad_x = np.gradient(im_fiber_g, axis=1)[edge:-edge,edge:-edge] # between -1 and 1   
    s = (grad_x* dx_norm) + (grad_y * dy_norm)
    s= s**2  # since + -  should not matter here   between 0 and 1


    ori_angle = []
    ori_mean = []
    int_mean = []
    proj_angle = []
    grad_slice = []
    alpha_dev_slice = []
    alpha_dev_slice_weight = []
    
    angle_dev_normed = (angle_dev*ori[edge:-edge,edge:-edge]) / np.mean(ori[edge:-edge,edge:-edge]) 
    s_norm = (s) / (grad_x**2 + grad_y**2)
    
    ang_sec = 5
    for alpha in range(-180, 180, ang_sec):  

            # mask all vectors in angle section and discard data within the cell
            mask_angle = (angle > (alpha-ang_sec/2)) & (angle <= (alpha+ang_sec/2)) & (~segmention["mask"][edge:-edge,edge:-edge])  
            
            # special care at the border (take opposite angle sections here)
            if alpha == -180:
                  mask_angle = ( (angle > (180-ang_sec/2)) | (angle <= (alpha+ang_sec/2)))  & (~segmention["mask"][edge:-edge,edge:-edge]) 
            if alpha == 180:
                  mask_angle = ( (angle > (alpha-ang_sec/2)) | (angle <= (-180 +ang_sec/2)))  & (~segmention["mask"][edge:-edge,edge:-edge]) 
                    
            ori_angle.append(alpha)
            ori_mean.append(np.mean(ori[edge:-edge,edge:-edge][mask_angle]))
            int_mean.append(np.mean(im_fiber_n[edge:-edge,edge:-edge][mask_angle]))

            # gradient to center
            grad_slice.append(np.nanmean(s_norm[mask_angle])) 
            
            # angle deviation to center    weighted and not weighted wir coheency
            alpha_dev_slice.append(np.nanmean(angle_dev[mask_angle])) 
            
            alpha_dev_slice_weight.append( np.mean( angle_dev_normed[mask_angle]) )
            
            #alpha_dev_slice_weight.append(np.average(angle_dev[mask_angle], weights= ori[edge:-edge,edge:-edge][mask_angle] ))

    
    # Total Value for complete image (without the mask)
    # average angle deviation
    alpha_dev_total = np.nanmean(angle_dev_normed[(~segmention["mask"][edge:-edge,edge:-edge])])
    # total gradient value for whole image
    grad_total = np.nanmean(s_norm[(~segmention["mask"][edge:-edge,edge:-edge])])
    # total coherency 
    coh_total = np.nanmean(ori[edge:-edge,edge:-edge][(~segmention["mask"][edge:-edge,edge:-edge]) ])
    
    print(alpha_dev_total)
    np.savetxt(os.path.join(out_list[n],"alpha_dev_total.txt"), [alpha_dev_total])
    np.savetxt(os.path.join(out_list[n],"grad_total.txt"), [grad_total])
    np.savetxt(os.path.join(out_list[n],"coh_total.txt"), [coh_total])
    
    
    # dipol character
    dip_ang = (np.max(alpha_dev_slice_weight)-np.min(alpha_dev_slice_weight))/ (np.max(alpha_dev_slice_weight)+np.min(alpha_dev_slice_weight))
    dip_ori = (np.max(ori_mean / np.max(ori_mean))-np.min(ori_mean / np.max(ori_mean)))/ (np.max(ori_mean / np.max(ori_mean))+np.min(ori_mean / np.max(ori_mean)))
    np.savetxt(os.path.join(out_list[n],"dip_ori.txt"), [dip_ori])
    np.savetxt(os.path.join(out_list[n],"dip_ang.txt"), [dip_ang])
   
    
   # to do coherence over distance
    # figsdf = plt.figure()
    # s = np.array(sorted(zip(  distance.flatten(), ori[edge:-edge,edge:-edge].flatten()  )) )
    # d = s[:,1]  #gaussian(s[:,0])
    # o = gaussian(s[:,0])
    # plt.scatter(d,o)
    
    """
    polar plots
    """
    
    fig13 = plt.figure()
    ax = plt.subplot(111, projection="polar")
    grad_slice = np.array(grad_slice)
    ax.plot( np.array(ori_angle)*np.pi/180,  (grad_slice-np.min(grad_slice))/(np.max(grad_slice)-np.min(grad_slice))   )  #/ np.max(grad_slice)
    plt.tight_layout()
    ax.set_rlim(bottom=1, top=0)
    plt.savefig(os.path.join(out_list[n],"gradient_norm.png"), dpi=200)       
   

    fig11 = plt.figure()
    ax = plt.subplot(111, projection="polar")
    ax.plot( np.array(ori_angle)*np.pi/180,  ori_mean / np.max(ori_mean)  )
    plt.tight_layout()
    plt.savefig(os.path.join(out_list[n],"polar-coher.png"), dpi=200)
    
    fig12 = plt.figure()
    ax = plt.subplot(111, projection="polar")
    ax.plot( np.array(ori_angle)*np.pi/180,  ori_mean / np.max(ori_mean),    c = "C0" , label="Allignment Collagen" )
    ax.plot( np.array(ori_angle)*np.pi/180,  int_mean / np.array([np.max(int_mean)]),  c = "C1" , label="Intensity Collagen")
    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(out_list[n],"polar-coher-2.png"), dpi=200)
    
        
    fig14 = plt.figure()
    ax = plt.subplot(111, projection="polar")
    ax.plot( np.array(ori_angle)*np.pi/180, alpha_dev_slice , label="no weights")
    ax.plot( np.array(ori_angle)*np.pi/180, alpha_dev_slice_weight , label="weights")
    ax.set_rlim(bottom=90, top=0)  # put allignet values further out for visualization
    

    
    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(out_list[n],"alpha_dev.png"), dpi=200)      
    

    
    plt.close('all')
    
    


    
    
    
    