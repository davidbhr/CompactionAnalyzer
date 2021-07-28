
'''
by Andreas Bauer 29.05.2020
Analyzing the orientation of structures like collagen fibres or cells in cell patches by analyzing the gradient of
images locally or as a whole. This uses the structure tensor https://en.wikipedia.org/wiki/Structure_tensor
and builds heavily on the method presented here http://bigwww.epfl.ch/demo/orientation/
'''
import warnings
import matplotlib.pyplot as plt
from skimage.filters import gaussian
from scipy.ndimage.filters import uniform_filter
import os
import copy
import numpy as np

from skimage.draw import circle
from scipy.signal import convolve2d
from CompactionAnalyzer.utilities import  convolution_fitler_with_nan

def rotate_vector_field(p, r):
    '''
    rotation of a vector or vector field by angel p
    :param p:
    :param r:
    :return:
    '''

    r_n = np.zeros(r.shape) + np.nan
    if len(r.shape) == 3:  # vector field
        # applying rotation matrix
        r_n[:, :, 0] = + np.cos(p) * (r[:, :, 0]) - np.sin(p) * (r[:, :, 1])
        r_n[:, :, 1] = + np.sin(p) * (r[:, :, 0]) + np.cos(p) * (r[:, :, 1])

    if len(r.shape) == 1:  # single vector
        # applying rotation matrix
        r_n[0] = + np.cos(p) * (r[0]) - np.sin(p) * (r[1])
        r_n[1] = + np.sin(p) * (r[0]) + np.cos(p) * (r[1])
    return r_n


def eigen_vec(eval, a, b, d):

    '''
    Calculateing the eigenvectors of a symmetric matrix [[a,b][b,c]] with eigenvalues eval
    :param eval: 1d array of eigenvalues
    :param a:
    :param b:
    :param d:
    :return:
    '''
    x = b / np.sqrt(b ** 2 + (eval - a) ** 2)
    y = (eval - a) / np.sqrt(b ** 2 + (eval - a) ** 2)
    return np.stack([x, y], axis=len(y.shape))


def select_max_min(x1, x2, b1, b2):

    '''
    Sort values from x1 and x2 into tqo arrays based on values in b1 and b2.
    x1[0,0] get sorted to x_max if b1[0,0]>b2[0,0] and x2[0,0] gets sorted to x_min in this case.
    :param x1:
    :param x2:
    :param b1:
    :param b2:
    :return:
    '''

    x1 = np.array(x1)
    x2 = np.array(x2)
    b1 = np.array(b1)
    b2 = np.array(b2)

    x_max = np.zeros(x1.shape)
    x_min = np.zeros(x2.shape)
    
    # Ignore RuntimeWarnings here 
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        bigger1 = np.abs(b1) > np.abs(b2)  # mask where absolute value of first value is bigger
        bigger2 = ~bigger1

    x_max[bigger1] = x1[bigger1]
    x_max[bigger2] = x2[bigger2]

    x_min[bigger2] = x1[bigger2]
    x_min[bigger1] = x2[bigger1]

    return x_max, x_min


def get_structure_tensor_gaussian(im, sigma):
    '''
    Structure tensor with gaussian weight. This how they typically do it.
    See https://en.wikipedia.org/wiki/Structure_tensor for some background.

    :param im: input image
    :param sigma: sigma of the weighting functions, this effectively defines the size of the interrogation window
    :return:
    '''

    grad_y = np.gradient(im, axis=0)  # parameters: spacing-> set higher dx and dy edge-order: some interpolation (?)
    grad_x = np.gradient(im, axis=1)

    # orientation tensor
    ot_xx = convolution_fitler_with_nan(grad_x * grad_x, gaussian, sigma=sigma)
    ot_yx = convolution_fitler_with_nan(grad_y * grad_x, gaussian, sigma=sigma)  # ot_yx an dot_xy are mathematically the same
    ot_yy = convolution_fitler_with_nan(grad_y * grad_y, gaussian, sigma=sigma)

    return ot_xx, ot_yx, ot_yy


def get_structure_tensor_uniform(im, size):
    '''
    Structure tensor with uniform weight
    See https://en.wikipedia.org/wiki/Structure_tensor for some background.


    :param im: input image
    :param size: window size of the interrogation area
    :return:
    '''

    grad_y = np.gradient(im, axis=0)
    grad_x = np.gradient(im, axis=1)

    # orientation tensor
    ot_xx = convolution_fitler_with_nan(grad_x * grad_x, uniform_filter, size=(size, size))
    ot_yx = convolution_fitler_with_nan(grad_y * grad_x, uniform_filter,  size=(size, size))
    ot_yy = convolution_fitler_with_nan(grad_y * grad_y, uniform_filter,  size=(size, size))

    return ot_xx, ot_yx, ot_yy


def get_structure_tensor_roi(im, mask=None):
    '''
    Structure tensor over specific region of interest with uniform weight.
    See https://en.wikipedia.org/wiki/Structure_tensor for some background.


    :param im: input image
    :param mask: mask specifying the region that is analyzed
    :return:
    '''

    #
    grad_y = np.gradient(im, axis=0)  # parameters: spacing-> set higher dx and dy edge-order: some interpolation (?)
    grad_x = np.gradient(im, axis=1)

    # if no mask is provided, use the whole image
    if not isinstance(mask, np.ndarray):
        mask = np.ones(grad_y.shape).astype(bool)
    else:
        mask = mask.astype(bool)

    # components of the structure tensor.
    # actual tensor would look like tensor = [[ot_xx], [ot_yx],
    #                                  [ot_yx] ,[ot_yy]]
    ot_xx = np.mean(grad_x[mask] * grad_x[mask])
    ot_yx = np.mean(grad_y[mask] * grad_x[mask])
    ot_yy = np.mean(grad_y[mask] * grad_y[mask])

    return ot_xx, ot_yx, ot_yy


def get_principal_vectors(ot_xx, ot_yx, ot_yy):
    '''
    Calculating eigenvectors and eigenvalues form the structure tensor, selecting the minimal and maximal eigenvalues
    and the corresponding eigenvectors. from https://www.soest.hawaii.edu/martel/Courses/GG303/Eigenvectors.pdf
    (maybe there is an error in the link?)
    This follows
    :param ot_xx: [0,0] component of structure tensor
    :param ot_yx: [0,1] and [1,0] component of structure tensor
    :param ot_yy: [1,1] component of structure tensor
    :return:
    '''

    eval1 = (ot_xx + ot_yy) / 2 + np.sqrt(((ot_xx - ot_yy) / 2) ** 2 + ot_yx ** 2)
    eval2 = (ot_xx + ot_yy) / 2 - np.sqrt(((ot_xx - ot_yy) / 2) ** 2 + ot_yx ** 2)

    evec1 = eigen_vec(eval1, ot_xx, ot_yx, ot_yy)
    evec2 = eigen_vec(eval2, ot_xx, ot_yx, ot_yy)

    # we  want the minimal eigenvalue and eigenvector
    max_eval, min_eval = select_max_min(eval1, eval2, eval1, eval2)
    max_evec, min_evec = select_max_min(evec1, evec2, eval1, eval2)

    # sometimes minimal vector is not defined, in this case create min eigenvector perpendicular to max eigenvector
    min_not_defined = np.logical_and(np.isnan(min_evec), ~np.isnan(max_evec))
    min_evec[min_not_defined] = rotate_vector_field(np.pi / 2, max_evec)[min_not_defined]

    # fill nans with zeros --> makes sense because later weighting with coherency would set zero  anyway
    # min_evec[np.isnan(min_evec)] = 0
    # max_evec[np.isnan(max_evec)] = np.nan

    return max_evec, min_evec, max_eval, min_eval


def analyze_area(im, mask):
    '''
    Orientation analysis on a specific area of an image.
    :param im: Input image
    :param mask: Mask to specify the analyzed area
    :return: coherency,
    max_evec, eigenvector of structure tensor with the large eigenvalue --> main orientation of the gradient field
    min_evec, eigenvector of structure tensor with the smaller eigenvalue --> main orientation of image structure
    max_eval, larger eigenvector
    min_eval, smaller eigenvector
    '''

    # calculating the structure tensor
    mask = mask.astype(bool)
    ot_xx, ot_yx, ot_yy = get_structure_tensor_roi(im, mask=mask)
    # getting vectors for minimal and maximal orientation
    max_evec, min_evec, max_eval, min_eval = get_principal_vectors(ot_xx, ot_yx, ot_yy)

    coherency = (max_eval - min_eval) / (max_eval + min_eval)

    return coherency, max_evec, min_evec, max_eval, min_eval


def get_main_orientation_squared(ang, vx=0, vy=0):
    # components of the orientation vector. length of this vector is always 1
    ox = np.cos(ang)
    oy = np.sin(ang)
    ori = np.sum((ox * vx + oy * vy) ** 2)
    # also interesting is the sum of absolute values :
    # np.sum(np.sqrt((ox * vx + oy * vy)**2))

    return ori

def analyze_local(im, sigma=0, size=0, filter_type="gaussian"):
    if filter_type =="gaussian":
        ot_xx, ot_yx, ot_yy = get_structure_tensor_gaussian(im, sigma)
    if filter_type == "uniform":
        ot_xx, ot_yx, ot_yy = get_structure_tensor_uniform(im, size)

    max_evec, min_evec, max_eval, min_eval = get_principal_vectors(ot_xx, ot_yx, ot_yy)
    ori = (max_eval - min_eval) / (max_eval + min_eval)

    return ori, max_evec, min_evec, max_eval, min_eval

'''
def custom_edge_filter(arr):
    arr_out = copy.deepcopy(arr).astype(int)
    shape1 = np.array([[0, 1, 0], [1, 1, 0], [0, 0, 0]])
    shape2 = np.array([[0, 1, 0], [0, 1, 1], [0, 0, 0]])
    shape3 = np.array([[0, 0, 0], [0, 1, 1], [0, 1, 0]])
    shape4 = np.array([[0, 0, 0], [1, 1, 0], [0, 1, 0]])
    for s in [shape1, shape2, shape3, shape4]:
        rem_mask = convolve2d(arr, s, mode="same") == 3
        arr_out[rem_mask] = 0
    return arr_out.astype(bool)
'''


def set_axis_attribute(ax, attribute, value):
    for p in ["left", "bottom", "right", "top"]:
        if hasattr(ax.spines[p], attribute):
            try:
                getattr(ax.spines[p], attribute)(value)  # this calls method
            except:
                setattr(ax.spines[p], attribute, value)
        else:
            raise AttributeError("Spines object has no attribute " + attribute)


'''
def display_mask(fig, mask, display_type="outline", type=1, color="C1", d=np.sqrt(2), ax=None, dm=True, lw=9):
    if not dm:
        return
    mask = mask.astype(int)
    if display_type == "outline":

        bm = binary_erosion((mask))
        bm[0,:] = 0
        bm[:,0] = 0
        bm[-1,:] = 0
        bm[:,-1] = 0
        out_line = mask - bm
        out_line = custom_edge_filter(out_line)  # risky
        out_line_graph, points = mask_to_graph(out_line, d=d)
        try:
            circular_path = find_path_circular(out_line_graph, 0)
        except RecursionError as e:
            print("while plotting mask outlines:", e)
            return

        circular_path.append(circular_path[0])  # to plot a fully closed loop
        if type == 1:
            ax = fig.axes[0] if ax is None else ax
            ax.plot(points[circular_path][:, 1], points[circular_path][:, 0], "--", color=color, linewidth=lw)
        if type == 2:
            for ax in fig.axes:
                ax.plot(points[circular_path][:, 1], points[circular_path][:, 0], "--", color=color, linewidth=lw)

'''


def plot1(im, im_f, sigma, ori_res, mask=None, out_folder=None, name="im.png"):
    max_evec, min_evec, max_eval, min_eval, ori = ori_res
    circle(r=50, c=im.shape[1] - 50, radius=sigma, shape=im.shape)
    circ = np.zeros(im.shape) + np.nan
    circ[circle(r=50, c=im.shape[1] - 50, radius=sigma, shape=im.shape)] = 1
    grady = np.gradient(im_f, axis=0)
    gradx = np.gradient(im_f, axis=1)
    vmin = np.min(np.stack([grady ** 2, gradx ** 2]))
    vmax = np.max(np.stack([grady ** 2, gradx ** 2]))

    fig, axs = plt.subplots(1, 4)
    axs[0].imshow(im)
    axs[0].imshow(circ, cmap="spring", vmin=0, vmax=1)
    f = max_eval / max_eval * np.min(im.shape) * 0.35
    axs[0].arrow(im.shape[1] / 2, im.shape[0] / 2, max_evec[0] * f, max_evec[1] * f, width=10,
                 color="C3")
    f = min_eval / max_eval * np.min(im.shape) * 0.35
    axs[0].arrow(im.shape[1] / 2, im.shape[0] / 2, min_evec[0] * f, min_evec[1] * f, width=10,
                 color="C6")
    # if isinstance(mask, np.ndarray):
    #  display_mask(fig, mask, ax=axs[0],lw=3)

    axs[1].imshow(im_f)
    axs[1].set_title("blurred image")
    axs[2].imshow(grady ** 2, vmin=vmin, vmax=vmax)
    axs[2].set_title("y gradient")
    im_disp = axs[3].imshow(gradx ** 2, vmin=vmin, vmax=vmax)
    axs[3].set_title("x gradient")
    plt.text(0.5, -500, "ori = " + str(np.round(ori, 3)))  ##

    for ax in axs:
        set_axis_attribute(ax, "set_visible", False)
        ax.tick_params(axis="both", tick1On=False, tick2On=False, label1On=False, label2On=False)

    cax = fig.add_axes([0.3, 0.1, 0.6, 0.05])

    plt.colorbar(im_disp, cax=cax, orientation="horizontal")
    if isinstance(out_folder, str):
        fig.savefig(os.path.join(out_folder, name))
    return fig


def full_angle_plot(ori_list, angs, out_folder=None, name="orient_dist.png"):
    fig = plt.figure()
    ax = plt.subplot(111, projection="polar")
    ax.plot(angs, ori_list)
    if isinstance(name, str) and isinstance(out_folder, str):
        fig.savefig(os.path.join(out_folder, name))
    return fig


def analyze_area_full_orientation(im, mask=None, points=1000, length=2 * np.pi):
    '''
    Calculates the alignment of the gradient field of an image with an orientation lines over a specified range of angles.
    Alignment = sum(grad*or), where grad is the gradient vector field, or is an orientation vector and * is
    the scalar product

    :param im: input image
    :param mask: mask specifying a region to be analyzed in the image
    :param points: number of sample points
    :param length: range of angles, keep at 2*np.pi for normal plot
    :return:
    '''

    grad_y = np.gradient(im, axis=0)  # parameters: spacing-> set higher dx and dy edge-order: some interpolation (?)
    grad_x = np.gradient(im, axis=1)

    # orientation tensor
    if not isinstance(mask, np.ndarray):
        mask = np.ones(grad_y.shape).astype(bool)
    else:
        mask = mask.astype(bool)
    oris = []
    angs = np.linspace(0, length, points)
    for ang in angs:
        oris.append(get_main_orientation_squared(ang, vx=grad_x[mask], vy=grad_y[mask]))
    oris = np.array(oris)
    return oris, angs


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


if __name__ == "__main__":
    # reading an image
    # use r"path\path\file.png" on windows
    im = plt.imread("/home/user/Desktop/ingo_fiber_orientations/7500_17022020/MAX_7500_17022020.lif - Series002.tif")[:,
         :, 0]

    # standard deviation of gaussian filter used for blurring
    # needs to remove all structures smaller the the structure we want to analyze
    sigma = 5

    # "contrast spreading" by setting all values below norm1-percentile to zero and
    # all values above norm2-percentile to 1
    norm1 = 5
    norm2 = 95
    # maybe change to full binarization
    # with np.median()
    # or maybe:
    # from skimage.filters import threshold_otsu
    #  threshold_otsu

    # applying normalizing/ contrast spreading
    im_n = normalize(im, norm1, norm2)
    # applying gaussian filter
    im_f = gaussian(im_n, sigma=sigma)

    # specifying an area for the analysis, this code selects the whole image
    mask = np.ones(im.shape).astype(bool)

    # orientation analysis
    coherence, max_evec, min_evec, max_eval, min_eval = analyze_area(im_f, mask)

    # full oriention distribution, by sampling orientations from 0 to 2*np.pi
    ori_list, angs = analyze_area_full_orientation(im_f, mask, points=100, length=np.pi * 2)

    # plot of blurred image and gradients
    fig1 = plot1(im, im_f, sigma, [max_evec, min_evec, max_eval, min_eval, coherence], mask)

    # plot of orientation distribution
    fig2 = full_angle_plot(ori_list, angs)
    # save by fig2.savefig("example.png")


