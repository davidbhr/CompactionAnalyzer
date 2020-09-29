"""
Created on Wed Jul  1 15:45:34 2020

@author: david boehringer + andreas bauer

Evaluates the fiber orientation around cells that compact collagen tissue.
Evaluates the orientation of collagen fibers towards the segmented cell center 
by using the structure tensor (globally / in angle sections / within distance shells).

Needs an Image of the cell and one image of the fibers (e.g. maximum projection of small stack)

"""
from roipoly import RoiPoly
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
from skimage.filters import gaussian, threshold_otsu, threshold_sauvola, try_all_threshold,threshold_yen
from skimage.morphology import remove_small_objects
from skimage.exposure import adjust_gamma   
from PIL import Image
from copy import copy
import imageio
from skimage import color
from scipy.ndimage.morphology import distance_transform_edt
import PIL.ImageDraw as ImageDraw
from tqdm import tqdm

#import skfmm    #      pip install scikit-fmm

def set_vmin_vmax(x, vmin, vmax):
    if not isinstance(vmin, (float, int)):
        vmin = np.nanmin(x)
    if not isinstance(vmax, (float, int)):
        vmax = np.nanmax(x)
    if isinstance(vmax, (float, int)) and not isinstance(vmin, (float, int)):
        vmin = vmax - 1 if vmin > vmax else None
    return vmin, vmax


def custom_mask(img,show_segmentation=True):
    """
    Image segmentation function to create a custom polygon mask, and evalute radius and position of the masked object.
    Need to use %matplotlib qt in jupyter notebook
    Args:
        img(array): Grayscale image as a Numpy array
    Returns:
        dict: Dictionary with keys: mask, radius, centroid (x/y)
    """
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
    # return dictionary containing spheroid information
    
    # # show segmentation
    # if show_segmentation:
    #     plt.figure()
    #     plt.subplot(121), plt.imshow(img)
    #     plt.subplot(122), plt.imshow(mask)
    #     fsdfsd
    
    
    
    return {'mask': mask, 'radius': radius, 'centroid': (cx, cy)} 



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

from skimage import exposure
from skimage.morphology import disk
from skimage.filters import rank


def segment_cell(img, thres=1, gaus1 =10, gaus2=20, iterartions=3,show_segmentation = True):
    """
    Image segmentation function to create  mask, radius, and position of a spheroid in a grayscale image.
    Args:
        img(array): Grayscale image as a Numpy array
        thres(float): To adjust the segmentation, keep 1
        iterations: iterations of closing steps , might increase to get more 
        robust segmentation but less precise segmentation, by default 1
    Returns:
        dict: Dictionary with keys: mask, radius, centroid (x/y)
    """
    height = img.shape[0]
    width = img.shape[1]
    # local gaussian   
    
    #img[:,400:540] = 0
    
    img = np.abs(gaussian(img, sigma=gaus1) - gaussian(img, sigma=gaus2))
    # segment cell
    # mask = img > threshold_otsu(img) * thres    #[::-1]
    mask = img > threshold_yen(img) * thres    #[::-1]
    
    # remove other objects
    
    #mask = scipy_morph.binary_closing(mask, iterations=iterartions)
    #mask = remove_small_objects(mask, min_size=100)
    mask = scipy_morph.binary_dilation(mask, iterations=iterartions)
    #mask = scipy_morph.binary_fill_holes(mask)
    

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
        # fsdfsd
    
    
    # determine radius of spheroid
    radius = np.sqrt(np.sum(mask) / np.pi)

    # determine center position of spheroid
    cy, cx = center_of_mass[distance_to_center.argmin()]

    # return dictionary containing spheroid information
    return {'mask': mask, 'radius': radius, 'centroid': (cx, cy)}


scale = 0.398 # use 7 um for collagen

from natsort import natsorted

# maxprojection Data
# read in list of cells and list of fibers to evaluate (must be in same order)
# use glob.glob or individual list of paths []     
fiber_list = natsorted(glob.glob(r"..//C003_T*.tif"))#[10]   # TODO first 9

#cell positions & image
pos = np.load(r"..//cell_positions//markers.npy") #[:4]
cell_img = normalize(imageio.imread(r"../C001_T001.tif"),0.5,99.5)

# retangle mask
# width of rectangle in middle of conenction lines between cells
# if AUTO = True we use distance to center as length of rectangle 
rect_length_auto = True
rect_length = None #20  # None  #   70/scale

rect_width = 20/scale

# Set Parameters   
sigma_tensor = 7/scale  # sigma of applied gauss filter / window for structure tensor analysis in px
                    # should be in the order of the objects to analyze !! test
edge = 0   # Cutt of pixels at the edge since values at the border cannot be trusted
# segmention_thres = 1  # for cell segemetntion, thres 1 equals normal otsu threshold , user also can specify gaus1 + gaus2 in segmentation if needed
sigma_first_blur  = 0.5  # slight first bluring of whole image before using structure tensor
# angle_sections = 5   # size of angle sections in degree 
# shell_width = 5/scale   # pixel width of distance shells


# create output folder accordingly
out_list = [os.path.join("field4")]


# data lists

connections_all = []
cell_con_all = []
distances_all = []

# pure angles
angles_connect_all =  []
angles_shell_all  =  []
angles_ratio_all  =  []
diff_angle_all = [] 
diff_cos_all = []
# 2(cos(a))-1   value
cos_connect_all =  []
cos_shell_all  =  []
cos_ratio_all  =  []
#intensities
int_connect_all =  []
int_shell_all  =  []
int_ratio_all  =  [] 
#ori - coherency
ori_connect_all =  []
ori_shell_all  =  []
ori_ratio_all  =  [] 

# loop thorugh cells
for n,i in tqdm(enumerate(fiber_list)):
    #create output folder if not existing
    if not os.path.exists(out_list[n]):
        os.makedirs(out_list[n])
    # load images
    # im_cell  = color.rgb2gray(imageio.imread(cell_list[n]))[400:-400,400:-400]
    im_fiber = color.rgb2gray(imageio.imread(fiber_list[n]))
    #"contrast spreading" by setting all values below norm1-percentile to zero and
    # # all values above norm2-percentile to 1
    norm1 = 5
    norm2 = 95
    # # applying normalizing/ contrast spreading
    #im_cell_n = normalize(im_cell, norm1, norm2)
    im_fiber_n = normalize(im_fiber, norm1, norm2)  
    im_fiber_g = gaussian(im_fiber_n, sigma=sigma_first_blur)     # blur fiber image slightly (test with local gauss - similar)   
    im_fiber_g_forstructure = im_fiber_g.copy()
 
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
    plots
    """

    # f = np.nanpercentile(ori,0.75)
    fig5, ax5 = show_quiver (min_evec[:,:,0] * ori, min_evec[:,:,1] * ori, filter=[0, 11], scale_ratio=0.1,width=0.001, cbar_str="coherency", cmap="viridis")
    # plot cell positions

    plt.plot(pos[:, 0], pos[:, 1], 'wo', ms=5)
    plt.savefig(os.path.join(out_list[n],"coh_quiver.png"), bbox_inches='tight', pad_inches=0, dpi=600)
    
    # plot overlay
    my_norm = matplotlib.colors.Normalize(vmin=0.99, vmax=1, clip=False)  
    cmap =  copy(plt.get_cmap('Greys'))
    # everything under vmin gets transparent (all zeros in mask)
    cmap.set_under('k', alpha=0)
    #everything else visible
    cmap.set_over('k', alpha=1)
    # plot mask and center
    show_quiver (min_evec[:,:,0] * ori, min_evec[:,:,1] * ori, filter=[0, 11],alpha=0 , scale_ratio=0.1,width=0.001, plot_cbar=False, cbar_str="coherency", cmap="viridis")
    plt.imshow(normalize(im_fiber_n), origin="upper")
   
    plt.plot(pos[:, 0], pos[:, 1], 'wo', ms=5)
    plt.tight_layout()
    plt.savefig(os.path.join(out_list[n],"struc-tens-o.png"), bbox_inches='tight', pad_inches=0, dpi=600)
 
    # plot overlay
    my_norm = matplotlib.colors.Normalize(vmin=0.99, vmax=1, clip=False)  
    cmap =  copy(plt.get_cmap('Greys'))
    # everything under vmin gets transparent (all zeros in mask)
    cmap.set_under('k', alpha=0)
    #everything else visible
    cmap.set_over('k', alpha=1)
    # plot mask and center
    show_quiver (min_evec[:,:,0] * ori, min_evec[:,:,1] * ori, filter=[0, 11],alpha=0 , scale_ratio=0.1,width=0.001, plot_cbar=False, cbar_str="coherency", cmap="viridis")
    plt.imshow(cell_img, origin="upper", cmap= "Greys_r")
    plt.plot(pos[:, 0], pos[:, 1], 'C1o', ms=5)
    plt.tight_layout()
    plt.savefig(os.path.join(out_list[n],"struc-tens-o-bf.png"), bbox_inches='tight', pad_inches=0, dpi=600)
    
    
    # save some structure data
    np.save(os.path.join(out_list[n],"min_evec.npy"), min_evec) 
    np.save(os.path.join(out_list[n],"ori.npy"), ori) 
   
    
   
    """
    cell connections
    """
    # image to draw masks
    # imagedraw = Image.new("RGB", cell_img.shape)
    # draw = ImageDraw.Draw(imagedraw)
    # show cell image in back
    fig1,ax1 = plt.subplots()
    figc,axc = plt.subplots()
    axc.imshow(normalize(im_fiber_n), origin="upper")
    ax1.imshow(cell_img, origin="upper", cmap= "Greys_r") 
    # loop through all cells
    for n_p, p in tqdm(enumerate(pos)):

        # calculate connections to all other cells (which are not yet looped over) , +1 to neglect identical cell
        for n_c,c in enumerate(pos[n_p+1:]):
            # plot connection line, calculate distance
            ax1.plot([p[0], c[0]], [p[1], c[1]],"C1-", linewidth=0.2)     
            d = ( (p[0]-c[0])**2 + (p[1]-c[1])**2  ) ** 0.5       
            # calculate normal vector and draw polygon mask   list of points    shift points along normal vectr by   4 * sigma_tensor  (4 * 7 um)
            v_dx = c[0] - p[0]
            v_dy = c[1] - p[1]
            vec = [v_dx/d, v_dy/d] #normed vector  (to project the angles on it)
            nvec =[-v_dy/d, v_dx/d] #normed perpendicular vector (to create a broader mask)
                     
            # draw mask
            shift_x = rect_width  * nvec[0]    # total width =  x times Sigma_tensor
            shift_y = rect_width * nvec[1]
            
     
            # rectangle mask next to whole connection line
            if rect_length_auto:
              rect_length = d/2   # half distance as width of rectangle
            # center of conenction line
            center = ( np.mean([p[0],c[0]]) , np.mean([p[1],c[1]]) )
            # upper and lower points for inncer rectangle (with rect_width)
            c_up =  ( center[0]+(vec[0] * rect_length) , center[1]+ (vec[1] * rect_length) )
            c_down = ( center[0]-(vec[0] * rect_length) , center[1]-(vec[1] * rect_length) )
                  
            # smaller rectangle in  rectangle around connection line
            draw_mask_center =    (  (c_down[0] + shift_x ,c_down[1] + shift_y)  ,   
                              (c_down[0] - shift_x,c_down[1] - shift_y),
                              (c_up[0] - shift_x,c_up[1]- shift_y),
                              (c_up[0] + shift_x,c_up[1] + shift_y)  )
            
            # mask the connection polygon (between 0 and 1 now ())
            imagedraw = Image.new("RGB", cell_img.shape)
            draw = ImageDraw.Draw(imagedraw)
            draw.polygon(draw_mask_center, fill=1 )
            mask_polygon = np.array(imagedraw,dtype=bool)[:,:,0]  # take any color channel
            
            
            # calculate distance to center
            y,x = np.indices(ori.shape)
            dx = x - p[0]
            dy = y - p[1]
            distance = np.sqrt(dx ** 2 + dy ** 2)  # dist to center
            dist_p_cdown = np.linalg.norm(np.array(p)-np.array(c_down))
            dist_p_center = np.linalg.norm(np.array(p)-np.array(center))
            # calculate spherical shell with distance between closer border to rectange and middle o rectange
            mask_shell = (distance > dist_p_cdown) & (distance <= dist_p_center) & (~mask_polygon)      
            mask_connection_half = mask_polygon & (distance <= dist_p_center)
            angle = np.arctan2(dy, dx) *360/(2*np.pi)
            dx_norm = (dx/distance)
            dy_norm = (dy/distance)



            # delete here ...# #save individual connection plot if you wish
            # fig2,ax2 = plt.subplots()
            # ax2.plot([p[0], c[0]], [p[1], c[1]],"C1-", linewidth=0.2)   
            # ax2.imshow(cell_img, origin="upper", cmap= "Greys_r")
            # imagedraw = Image.new("RGB", cell_img.shape)
            # draw = ImageDraw.Draw(imagedraw)
            # draw.polygon(draw_mask, fill=200 )
            # mask = np.array(imagedraw)
            # ax2.imshow(mask, alpha=0.3)
            # fig2.savefig(os.path.join(out_list[n], "connection_p1","{}.png".format(str(n_c).zfill(3))))
            # fig2.clf()          
            # if n_p == 1:
            #     sdfsdf
      

            
            """
            calculate and weight angle deviation (+mean coherency) to connection line
            """
  
            # calculate angle deviation towards connection lines , angle value - later convert to (2*cos(a))-1    for value between -1 and 1     
            angle_dev_degree = np.arccos(np.abs( dx_norm * min_evec[:,:,0] + dy_norm*min_evec[:,:,1] )) * 360/(2*np.pi)  # only connection line :  dx_norm * min_evec[:,:,0] + dy_norm*min_evec[:,:,1]

            # calculate values within shell and connection mask
            # mask regions
            mask_angle_conhalf = np.ma.array(angle_dev_degree,mask=mask_connection_half)
            mask_angle_shell = np.ma.array(angle_dev_degree,mask=mask_shell)
            mask_ori_conhalf = np.ma.array(ori,mask=mask_connection_half)
            mask_ori_shell = np.ma.array(ori,mask=mask_shell)  
            mask_int_conhalf = np.ma.array(im_fiber_g_forstructure,mask=mask_connection_half)
            mask_int_shell = np.ma.array(im_fiber_g_forstructure,mask=mask_shell)  
            
            # get data within mask
            data_mask_angle_conhalf  =  mask_angle_conhalf.data[mask_angle_conhalf.mask == True]
            data_mask_angle_shell  =  mask_angle_shell.data[mask_angle_shell.mask == True]
            data_mask_ori_conhalf    =  mask_ori_conhalf.data[mask_ori_conhalf.mask == True]
            data_mask_ori_shell    =  mask_ori_shell.data[mask_ori_shell.mask == True]
            data_mask_int_conhalf    =  mask_int_conhalf.data[mask_int_conhalf.mask == True]
            data_mask_int_shell    =  mask_int_shell.data[mask_int_shell.mask == True]
          # compute mean and weighted mean for conenction and shell
            mean_angle_conhalf = np.nanmean(data_mask_angle_conhalf)
            mean_angle_conhalf_weighted = np.nanmean(sum(data_mask_ori_conhalf * data_mask_angle_conhalf) / sum(data_mask_ori_conhalf))
            mean_angle_shell = np.nanmean(data_mask_angle_shell)
            mean_angle_shell_weighted = np.nanmean(sum(data_mask_ori_shell * data_mask_angle_shell) / sum(data_mask_ori_shell))    
            mean_ori_conhalf = np.nanmean(data_mask_ori_conhalf)
            mean_ori_shell = np.nanmean(data_mask_ori_shell)
              
             # mean orientation cos(2a) 
            mean_cos_conhalf =  np.nanmean( np.cos( 2 * data_mask_angle_conhalf*np.pi/180 )) 
            mean_cos_conhalf_weighted =  np.nanmean(np.cos ( 2 *( sum(data_mask_ori_conhalf * data_mask_angle_conhalf) / sum(data_mask_ori_conhalf)  )  * np.pi /180 ))  
            mean_cos_shell =  np.nanmean(np.cos( 2 *  data_mask_angle_shell*np.pi/180) )
            mean_cos_shell_weighted = np.nanmean(  np.cos( 2 * (sum(data_mask_ori_shell * data_mask_angle_shell) / sum(data_mask_ori_shell)) *np.pi/180))    
            
            
            # compute mean intensity conenction and shell
            mean_int_conhalf = np.nanmean(data_mask_int_conhalf)
            mean_int_shell = np.nanmean(data_mask_int_shell)


            
            
            # add 1 to have value between 2 and zero than calculate ratio of allignment difference
            ratio_angle =  (mean_angle_conhalf ) / (mean_angle_shell) ,    (mean_angle_conhalf_weighted) / (mean_angle_shell_weighted )    
            ratio_cos =  (mean_cos_conhalf + 1) / (mean_cos_shell + 1) ,    (mean_cos_conhalf_weighted + 1) / (mean_cos_shell_weighted + 1) 
            ratio_ori = mean_ori_conhalf/mean_ori_shell
            ratio_int = mean_int_conhalf/mean_int_shell
            
            print("ratio_cos:"+str(ratio_cos)) 
            print("--------------------------")
            print("ratio_int:"+str(ratio_int))
            
            # differences in angle and cosinus between shell and conenction
            diff_angle =  mean_angle_shell - mean_angle_conhalf  ,    mean_angle_shell_weighted - mean_angle_conhalf_weighted
            diff_cos =    mean_cos_conhalf- mean_cos_shell  ,     mean_cos_conhalf_weighted- mean_cos_shell_weighted
            
            
            # plot mask
            # plt.imshow(mask_angle_dw2)
            plt.plot([p[0], c[0]], [p[1], c[1]],"C1-", linewidth=0.5)  
                

            # append data to all list  
            distances_all.append([d])
            cell_con_all.append([p,c])
            connections_all.append( [ [p[0], c[0]], [p[1], c[1]] ] )     
            angles_ratio_all.append([ratio_angle])
            angles_connect_all.append([mean_angle_conhalf, mean_angle_conhalf_weighted])   # raw, dw1, dw2
            angles_shell_all.append([mean_angle_shell, mean_angle_shell_weighted])   # raw, dw1, dw2
            cos_ratio_all.append([ratio_cos])
            cos_connect_all.append([mean_cos_conhalf, mean_cos_conhalf_weighted])   # raw, dw1, dw2
            cos_shell_all.append([mean_cos_shell, mean_cos_shell_weighted])   # raw, dw1, dw2
            ori_ratio_all.append([ratio_ori])
            ori_connect_all.append([mean_ori_conhalf])   
            ori_shell_all.append([mean_ori_shell])   
            int_ratio_all.append([ratio_int])
            int_connect_all.append([mean_int_conhalf])   # raw, dw1, dw2
            int_shell_all.append([mean_int_shell])   # raw, dw1, dw2
            diff_angle_all.append([diff_angle]) 
            diff_cos_all.append([diff_cos]) 
            
            
            # figc,axc = plt.subplots()
            # # show_quiver (min_evec[:,:,0] * ori, min_evec[:,:,1] * ori, filter=[0, 11],alpha=0 , scale_ratio=0.1,width=0.0005, plot_cbar=False, cbar_str="coherency", cmap="viridis")
            # axc.imshow(angle_dev_cos2a,alpha=0.4)
            # # axc.colorbar()
            # axc.imshow(mask_connection_half,alpha=0.5,cmap="Greens")
            # axc.imshow(mask_shell,alpha=0.1,cmap="Reds")
            # axc.scatter(center[0],center[1], c="peru")
            # axc.scatter(p[0],p[1], c="g")
            # axc.scatter(c[0],c[1], c="g")
            # # plt.scatter(c_up[0],c_up[1])
            # # plt.scatter(c_down[0],c_down[1])
            # axc.plot([p[0], c[0]], [p[1], c[1]],"C1-", linewidth=0.2)  
            # plt.savefig(os.path.join(out_list[n],"figc","np_nc_"+str(n_p)+"_"+str(n_c)+".png"))
            # plt.clf()
 
        
# make some arrays  
connections_all = np.array(connections_all)  
cell_con_all = np.array(cell_con_all) 
distances_px_all = np.array(distances_all)
angles_connect_all = np.array(angles_connect_all)
angles_shell_all = np.array(angles_shell_all)
angles_ratio_all = np.array(angles_ratio_all)[:,0]  
cos_ratio_all = np.array(cos_ratio_all)[:,0]  
cos_connect_all = np.array(cos_connect_all)
cos_shell_all = np.array(cos_shell_all)  
ori_ratio_all = np.array(ori_ratio_all)
ori_connect_all = np.array(ori_connect_all)
ori_shell_all = np.array(ori_shell_all)  
int_ratio_all = np.array(int_ratio_all)
int_connect_all = np.array(int_connect_all)
int_shell_all = np.array(int_shell_all)  
diff_angle_all = np.array(diff_angle_all)[:,0]  
diff_cos_all = np.array(diff_cos_all)[:,0]  



# save data
np.save(os.path.join(out_list[n],"connections_all.npy"), connections_all) 
np.save(os.path.join(out_list[n],"cell_con_all.npy"), cell_con_all) 
np.save(os.path.join(out_list[n],r"distances_px_all.npy"), distances_px_all) 
np.save(os.path.join(out_list[n],r"angles_connect_all.npy"), angles_connect_all ) 
np.save(os.path.join(out_list[n],r"angles_shell_all.npy"), angles_shell_all) 
np.save(os.path.join(out_list[n],r"angles_ratio_all.npy"), angles_ratio_all) 
np.save(os.path.join(out_list[n],r"cos_ratio_all.npy"), cos_ratio_all ) 
np.save(os.path.join(out_list[n],r"cos_connect_all.npy"), cos_connect_all) 
np.save(os.path.join(out_list[n],r"cos_shell_all.npy"), cos_shell_all) 
np.save(os.path.join(out_list[n],r"ori_ratio_all.npy"), ori_ratio_all ) 
np.save(os.path.join(out_list[n],r"ori_connect_all.npy"), ori_connect_all) 
np.save(os.path.join(out_list[n],r"ori_shell_all.npy"), ori_shell_all) 
np.save(os.path.join(out_list[n],r"int_ratio_all.npy"), int_ratio_all ) 
np.save(os.path.join(out_list[n],r"int_connect_all.npy"), int_connect_all) 
np.save(os.path.join(out_list[n],r"int_shell_all.npy"), int_shell_all) 
np.save(os.path.join(out_list[n],r"diff_angle_all.npy"), diff_angle_all) 
np.save(os.path.join(out_list[n],r"diff_cos_all.npy"), diff_cos_all) 



# show the drawn polygons
# plt.imshow(imagedraw, alpha=0.1, cmap="Greens")
#save image
plt.savefig(os.path.join(out_list[n],"all-connections.png"), dpi=600)
# fig1.savefig(os.path.join(out_list[n],"fig1.png"), dpi=600)
# figc.savefig(os.path.join(out_list[n],"figc.png"), dpi=600)




        
    
    
    
    
    
    

    
