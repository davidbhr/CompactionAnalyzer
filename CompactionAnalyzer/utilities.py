import numpy as np
import os
from natsort import natsorted
#from skimage.filters import gaussian
from skimage import io
import glob as glob

def load_stack(stack_list):
    z = len(stack_list)
    img_example = io.imread(stack_list[0], as_gray=True)
    h,w = np.shape(img_example)
    tiffarray=np.zeros((h,w,z))
    for n,img in enumerate(stack_list):
        tiffarray[:,:,n]=  (io.imread(stack_list[n], as_gray=True))  #normalize
    return tiffarray    
    


def generate_lists(fiber_list_string, cell_list_string, output_main = "Output"):
    """
    generate list of input data and for output we generate the same folder structure  
    as in cell_list but located in 
    the output_main  
    output_main: base folder for output files
    fiber_list_string: string to search for fiber images in glob style
    cell_list_string: string to search for cell images in glob style
    
    """

    # read in images (must be in same order)
    fiber_list = natsorted(glob.glob(fiber_list_string))  
    cell_list = natsorted(glob.glob(cell_list_string))   #check that order is same to fiber
    # get base path (before * in glob - even works without any *)
    base = os.path.split(cell_list_string[:cell_list_string.find("*")])[0]
    # get the rest of the path
    rest_paths =[os.path.split(os.path.relpath(p, base))[0] for p in (cell_list)]
    # get the file name
    names = [os.path.splitext(os.path.split(p)[1])[0] for p in cell_list]
    # put together accordingly in the output_main folder
    out_list = [os.path.join(output_main,rest_path, name) for name, rest_path in zip(names, rest_paths)]

    return fiber_list,cell_list, out_list
    

def flatten_dict(*args):
    ret_list = [i for i in range(len(args))]
    for i, ar in enumerate(args):
        ret_list[i] = np.concatenate([*ar.values()], axis=0)
    return ret_list

def createFolder(directory):
    '''
    function to create directories if they dont already exist
    '''
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
        return directory

    except OSError:
        print('Error: Creating directory. ' + directory)


def make_iterable(value):
    if not hasattr(value, '__iter__') or isinstance(value,str):
        return [value]
    else:
        return value



def convolution_fitler_with_nan(arr1, function, **kwargs):

    '''
        applies a gaussian to an array with nans, so that nan values are ignored.
    :param arr1:
    :param function: any convloution functio, such as skimage.filters.gaussian or scipy.ndimage.filters.ndimage.uniform_filter
    :param f_args:  kwargs for the convolution function
    :return:
    '''


    arr_zeros = arr1.copy()
    arr_ones = np.zeros(arr1.shape)
    arr_zeros[np.isnan(arr1)] = 0  # array where all nans are replaced by zeros
    arr_ones[~np.isnan(arr1)] = 1  # array where all nans are replaced by zeros and all other values are replaced by ones

    filter_zeros = function(arr_zeros, **kwargs)  # gaussian filter applied to both arrays
    filter_ones = function(arr_ones, **kwargs)
    filter_final = filter_zeros / filter_ones  # devision cancles somehow the effect of nan positions
    filter_final[np.isnan(arr1)] = np.nan  # refilling original nans

    return filter_final

def create_circular_mask(h, w, center=None, radius=None):
    # following 
    # https://stackoverflow.com/questions/44865023/how-can-i-create-a-circular-mask-for-a-numpy-array
    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    
    # EXAMPLE CODE TO CREATE A white blob in image of same shape as fiber image
    # from PIL import Image
    # fiber = plt.imread(r"U:\Dropbox\software-github\CompactionAnalyzer\TestData\Random1\Fiber.tif")
    # mask = create_circular_mask(fiber.shape[0],fiber.shape[1],radius=60).astype("uint8")
    # img = Image.fromarray(mask*256)
    # img.save("cell_rand.tif")
    return mask