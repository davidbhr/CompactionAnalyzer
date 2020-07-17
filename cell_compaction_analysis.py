"""
Created on Wed Jul  1 15:45:34 2020

@author: david + andi

Evaluates the fiber orientation around cells that compact collagen tissue.
Applying the Structure tensor and evaluates the orientation within 
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


edge = 40
# create output folder accordingly
#out_list = [os.path.join("angle_eval","2-angle",cell_list[i].split(os.sep)[0], os.path.basename(cell_list[i])[:-4]) for i in range(len(cell_list))]
out_list = [os.path.join("analysis",cell_list[i].split(os.sep)[0], os.path.basename(cell_list[i])[:-4]) for i in range(len(cell_list))]


# loop thorugh cells
for n,i in tqdm(enumerate(fiber_list)):
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
    """
    Structure tensor
    """
    # Structure Tensor Orientation
    # blur fiber image slightly (trz local gauss - similar)
    im_fiber_g = gaussian(im_fiber_n, sigma=0.5)
    # get structure tensor
    ori, max_evec, min_evec, max_eval, min_eval = analyze_local(im_fiber_g, sigma=15, size=0, filter_type="gaussian")
    ori, max_evec, min_evec, max_eval, min_eval = ori[edge:-edge,edge:-edge], max_evec[edge:-edge,edge:-edge], min_evec[edge:-edge,edge:-edge], \
                                                  max_eval[edge:-edge,edge:-edge], min_eval[edge:-edge,edge:-edge]

    f = np.percentile(ori,0.75)

    # cutt off the edges    
    edge = 40
    fig5, ax5 = show_quiver (min_evec[:,:,0] * ori, min_evec[:,:,1] * ori, filter=[f, 15], scale_ratio=0.1,width=0.003, cbar_str="coherency", cmap="viridis")
    center_small = (segmention["centroid"][0]-edge,segmention["centroid"][1]-edge)
    ax5.plot(center_small[0],center_small[1],"o")
    # Calculate Angle + Distances
    y,x = np.indices(ori.shape)
    dx = x - center_small[0]
    dy = y - center_small[1]
    distance = np.sqrt(dx ** 2 + dy ** 2)
    angle = np.arctan2(dy, dx) *360/(2*np.pi)
    dx_norm = (dx/distance)
    dy_norm = (dy/distance)

    # Angular deviation from orietation to center vector
    angle_dev = np.arccos(np.abs(dx_norm * min_evec[:,:,0] + dy_norm*min_evec[:,:,1])) * 360/(2*np.pi)
    # GRADIENT    
    grad_y = np.gradient( gaussian(im_fiber_n, sigma=0.5), axis=0)[edge:-edge,edge:-edge]
    grad_x = np.gradient( gaussian(im_fiber_n, sigma=0.5), axis=1)[edge:-edge,edge:-edge] # between -1 and 1
    s = (grad_x * dx_norm) + (grad_y * dy_norm)
    s = s**2  # since + -  should not matter here   between 0 and 1
    s_norm = s / (grad_x**2 + grad_y**2)


    ori_angle = []
    ori_mean = []
    ori_mean_weight = []
    int_mean = []
    proj_angle = []
    grad_slice = []
    alpha_dev_slice = []
    alpha_dev_slice_weight = []
    alpha_dev_slice_weight2 = []

    # weighting by coherence
    angle_dev_weighted = (angle_dev * ori) / np.mean(ori)
    # weighting by coherence and image intensity
    im_fiber_g = im_fiber_g[edge:-edge,edge:-edge]
    # could also use non filtered image
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4497681/ these guys
    # use a threshold in the intensity image// only coherence and orientation vectors
    # corresponding to pixels in the intesity iamage above a threshold are considered.
    weight_image = gaussian(im_fiber_g,sigma=15)
    angle_dev_weighted2 = (angle_dev_weighted *weight_image) / np.mean(weight_image)

    # also weighting the coherency like this
    ori_weight2 = (ori * weight_image) / np.mean(weight_image)



    plt.figure();plt.imshow(angle_dev_weighted); plt.colorbar()
    mx = min_evec[:,:,0] * ori
    my = min_evec[:,:,1] * ori
    mx, my, x, y = filter_values(mx, my, abs_filter=0,
                                   f_dist=15)  #

    plt.quiver(x, y, mx*300, my*300,scale=1,scale_units="xy", angles="xy")
    plt.figure();plt.imshow(ori); plt.colorbar()
    plt.title("orientation vector, angle_dev_weighted overlay")


    ang_sec = 5
    for alpha in range(-180, 180, ang_sec):
            mask_angle = (angle > (alpha-ang_sec/2)) & (angle <= (alpha+ang_sec/2)) & (~segmention["mask"][edge:-edge,edge:-edge])
            if alpha == -180:
                  mask_angle = ((angle > (180-ang_sec/2)) | (angle <= (alpha+ang_sec/2))) & (~segmention["mask"][edge:-edge,edge:-edge])
            if alpha == 180:
                  mask_angle = ((angle > (alpha-ang_sec/2)) | (angle <= (-180 +ang_sec/2))) & (~segmention["mask"][edge:-edge,edge:-edge])
                    
            ori_angle.append(alpha)
            ori_mean.append(np.mean(ori[mask_angle]))
            ori_mean_weight.append(np.mean(ori_weight2[mask_angle]))
            int_mean.append(np.mean(im_fiber_g[mask_angle]))
            # gradient to center
            grad_slice.append(np.nanmean(s_norm[mask_angle]))
            alpha_dev_slice.append(np.nanmean(angle_dev[mask_angle]))
            alpha_dev_slice_weight.append(np.mean(angle_dev_weighted[mask_angle]))
            alpha_dev_slice_weight2.append(np.mean(angle_dev_weighted2[mask_angle]))

    
    # Total Value for complete image (without the mask)
    # averages
    alpha_dev_total1 = np.nanmean(angle_dev[(~segmention["mask"][edge:-edge,edge:-edge])])
    alpha_dev_total2 = np.nanmean(angle_dev_weighted[(~segmention["mask"][edge:-edge, edge:-edge])])
    alpha_dev_total3 = np.nanmean(angle_dev_weighted2[(~segmention["mask"][edge:-edge, edge:-edge])])
    coh_total = np.nanmean(ori[(~segmention["mask"][edge:-edge,edge:-edge]) ])
    coh_total2 = np.nanmean(ori_weight2[(~segmention["mask"][edge:-edge, edge:-edge])])
    # save to txt file
    values = [coh_total ,coh_total2, alpha_dev_total1, alpha_dev_total2, alpha_dev_total3]
    strings = ["mean_coh.txt", "mean_coh_w_int.txt", "mean_angle.txt",
      "mean_angle_we_coh.txt", "mean_angle_we_coh_int.txt"]
    for i,v in enumerate(values):
        np.savetxt(os.path.join(out_list[n],strings[i]), [v] ) 

    
    
     # Angular deviation from orietation to center vector
    #angle_dev= np.arccos(np.abs(dx_norm*min_evec[:,:,1][edge:-edge,edge:-edge] + dy_norm*min_evec[:,:,0][edge:-edge,edge:-edge] ))  *360/(2*np.pi)
    plt.figure();plt.imshow(angle_dev, origin="upper", cmap="viridis");plt.colorbar(); plt.savefig(os.path.join(out_list[n],"angle_dev.png"), dpi=200)
    
    
    # translating the angles to coordinates in polar plot
    angle_plotting1 = (np.array(ori_angle) * np.pi / 180)
    angle_plotting = angle_plotting1.copy()
    angle_plotting[angle_plotting1 < 0] =  np.abs(angle_plotting[angle_plotting1 < 0])
    angle_plotting[angle_plotting1 > 0] =  np.abs(angle_plotting[angle_plotting1>0] - 2* np.pi)

    # test to appreciate how the angles work in plot
    # a = np.linspace(np.pi, np.pi * 2, len(angle_plotting))
    # b = np.linspace(0, 1, len(angle_plotting))
    # plt.figure()
    # ax = plt.subplot(111, projection="polar")
    # ax.plot(angle_plotting, b, label="angle_plotting")
    # ax.plot((np.array(ori_angle) * np.pi / 180), b, label="ori_angle directly")
    # plt.legend()
    # plt.title("illustration of how angles in the polar plot work")


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
    plt.savefig(os.path.join(out_list[n],"orientation.png"), dpi=200)
    
    
    
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
    fig8= plt.figure() 
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
    ddsf
    plt.close("all")