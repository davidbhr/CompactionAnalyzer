import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
from matplotlib_scalebar.scalebar import ScaleBar
import warnings
matplotlib.pyplot.switch_backend("agg")

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

def copycmap(cname):
    cmap =  plt.get_cmap(cname)
    colors = cmap(np.linspace(0, 1, 256))
    cmap =  matplotlib.colors.LinearSegmentedColormap.from_list(cname, cmap(np.linspace(0, 1, 256)))
    return cmap

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
    
     # Ignore RuntimeWarnings here 
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        select_size = def_abs > abs_filter
    select = select_x * select_y * select_size
    s1 = ar1[select]
    s2 = ar2[select]
    return s1, s2, xv[select], yv[select]


def show_quiver(fx, fy, filter=[0, 1], scale_ratio=0.5, headwidth=0., headlength=0., headaxislength=0.,
                width=None, cmap="rainbow",
                figsize=None, cbar_str="", ax=None, fig=None
                , vmin=None, vmax=None, cbar_axes_fraction=0.2, cbar_tick_label_size=15
                , cbar_width="2%", cbar_height="50%", cbar_borderpad=0.1,
                cbar_style="not-clickpoints", plot_style="not-clickpoints", cbar_title_pad=1, plot_cbar=True, alpha=1,
                ax_origin="upper", pivot = "middle", **kwargs):
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


      
def plot_angle_dev(angle_map,vec0,vec1,coherency_map,path_png,label="Angle Deviation",dpi=300,cmap="viridis"):
     # angle deviation no weights
    fig =plt.figure();plt.imshow(angle_map,cmap=cmap); cbar =plt.colorbar()
    mx = vec0 * coherency_map
    my = vec1 * coherency_map
    mx, my, x, y = filter_values(mx, my, abs_filter=0,
                                   f_dist=15)  #9
    plt.quiver(x, y, mx*300, my*300,scale=1.2,scale_units="xy", angles="xy",
               headwidth=0., headlength=0., headaxislength=0.)  #, width=0.005)
    plt.tight_layout()
    plt.axis('off'); cbar.set_label(label,fontsize=12)
    plt.savefig(path_png, dpi=dpi, bbox_inches='tight', pad_inches=0); plt.tight_layout()
    return fig
    
def plot_coherency(coherency,path_png,label="Coherency",dpi=300):
     # angle deviation no weights
    fig =plt.figure();plt.imshow(coherency); cbar =plt.colorbar()
    plt.tight_layout()
    plt.axis('off'); cbar.set_label(label,fontsize=12)
    plt.savefig(path_png, dpi=dpi, bbox_inches='tight', pad_inches=0); plt.tight_layout()
    return fig
     
def plot_polar(angle_plotting, something, path_png,label="something",dpi=300,
               something2 = None, something3 = None, label2 = None, label3 =None):
    fig = plt.figure();ax1 = plt.subplot(111, projection="polar")
    ax1.plot(angle_plotting, something, label=label , linewidth=2, c = "C0")
    if something2:
        ax1.plot(angle_plotting, something2, label=label2 , linewidth=2, c = "C1")
    if something3:
        ax1.plot(angle_plotting, something3, label=label3 , linewidth=2, c = "C2")    
    plt.tight_layout();plt.legend(fontsize=12);plt.savefig(path_png, dpi=dpi, bbox_inches='tight', pad_inches=0)
    return fig    

def plot_triple(results_angle, results_total, path_png ,dpi=300):
           # Triple plot
           fig = plt.figure(figsize=(20,6))
           axs1 = plt.subplot(131, projection="polar")
           axs1.plot(results_angle['Angles Plotting'],  results_angle['Coherency'], label="Coherency" )
           axs1.plot(results_angle['Angles Plotting'],  results_angle['Coherency (weighted by intensity)'], label="Coherency (weighted)" )
           plt.legend(fontsize=12)
           ax2 = plt.subplot(132, projection="polar")
           ax2.plot(results_angle['Angles Plotting'], results_angle['Orientation'] , label="Orientation")
           ax2.plot(results_angle['Angles Plotting'], results_angle['Orientation (weighted by intensity and coherency)'] , label="Orientation weighed") 
     
           plt.legend(fontsize=12)
           strings = ["Mean Coherency", "Mean Coherency\nweighted by intensity", "Mean Angle",
                      "Mean Angle weighted\nby coherency", "Mean Angle weighted\nby coherency and intensity",
                      "Mean Orientation",
                      "Mean Orientation weighted\nby coherency",
                      "Mean Orientation weighted\nby coherency and intensity"]
           values = [results_total['Mean Coherency'][0] , results_total['Mean Coherency (weighted by intensity)'][0], 
                     results_total['Mean Angle'][0],
                     results_total['Mean Angle (weighted by coherency)'][0],
                     results_total['Mean Angle (weighted by intensity and coherency)'][0],
                     results_total['Orientation'][0],
                     results_total['Orientation (weighted by coherency)'][0],
                     results_total['Orientation (weighted by intensity and coherency)'][0],  
                     ]
           values = [[str(np.round(x,4))] for x in values]
           table_text = [[strings[i], values[i]] for i in range(len(values))]
           ax3 = plt.subplot(133);ax3.axis('tight'); ax3.axis('off')
           table = ax3.table(cellText = values, rowLabels=strings,bbox=[0.6,0.2,0.7,0.9])
           table.set_fontsize(11)
           #plt.tight_layout()
           plt.savefig(path_png, dpi=dpi)#, bbox_inches='tight', pad_inches=0)
           return fig

def quiv_coherency_center(vec0,vec1,center0,center1,coherency_map, path_png, dpi=200):
    f = np.nanpercentile(coherency_map,0.75)
    fig5, ax5 = show_quiver (vec0 * coherency_map, vec1 * coherency_map, filter=[f, 15], scale_ratio=0.1,width=0.003, 
                             cbar_str="Coherency", cmap="viridis")
    ax5.plot(center0,center1,"o")
    plt.tight_layout();plt.savefig(path_png, dpi=dpi, bbox_inches='tight', pad_inches=0)
    return fig5

def plot_fiber_seg(fiber_image,c0,c1,segmention, path_png,dpi=200, scale=None ):
    fig7= plt.figure() 
    my_norm = matplotlib.colors.Normalize(vmin=0.9999, vmax=1, clip=False)  
    # create a copy of matplotlib cmap
    cmap = copycmap("Greys")


    # everything under vmin gets transparent (all zeros in mask)
    cmap.set_under('k', alpha=0)
    #everything else visible
    cmap.set_over('k', alpha=1)
    # plot mask and center
    plt.imshow(fiber_image, origin="upper")
    plt.imshow(segmention, cmap=cmap, norm = my_norm, origin="upper")
    plt.scatter(c0,c1, c= "w");plt.axis('off');
    if scale is not None:
        scalebar = ScaleBar(scale, "um", length_fraction=0.1, location="lower right", box_alpha=0 , 
                    color="k")
        plt.gca().add_artist(scalebar)

    plt.tight_layout() ;plt.savefig(path_png, dpi=dpi, bbox_inches='tight', pad_inches=0)
    return fig7

def plot_overlay(fiber_image , c0,c1, vec0,vec1, coherency_map,
                       segmention,path_png,dpi=200, show_n = 15, scale=None ):
            fig=plt.figure();f = np.nanpercentile(coherency_map,0.75)
            # plot overlay
            my_norm = matplotlib.colors.Normalize(vmin=0.99, vmax=1, clip=False)  
            # create a copy of matplotlib cmap
            cmap = copycmap("Greys")
            # everything under vmin gets transparent (all zeros in mask)
            cmap.set_under('k', alpha=0)
            #everything else visible
            cmap.set_over('k', alpha=1)
            # plot mask and center
            show_quiver (vec0 * coherency_map, vec1 * coherency_map, filter=[f, show_n],alpha=0 , scale_ratio=0.08,width=0.002, plot_cbar=False, cbar_str="coherency", cmap="viridis")
            plt.imshow(fiber_image, origin="upper")
            plt.imshow(segmention, cmap=cmap, norm = my_norm, origin="upper", zorder=100)
            plt.scatter(c0,c1, c= "w", zorder=200)
            if scale is not None:
                scalebar = ScaleBar(scale, "um", length_fraction=0.1, location="lower right", box_alpha=0 , 
                            color="k")
                plt.gca().add_artist(scalebar)
            plt.tight_layout() ;plt.savefig(path_png, dpi=dpi, bbox_inches='tight', pad_inches=0)
            return fig       
        
def plot_shells(shell_masks,path_png,dpi=200 ):
    fig =  plt.figure()
    cmap_list = ["Greens","Greys","Reds","Oranges","Blues","PuBu","GnBu"]
    for s in  range(len(shell_masks)):
        my_norm = matplotlib.colors.Normalize(vmin=0.99, vmax=1, clip=False)  
        # create a copy of matplotlib cmap
        cmap = copycmap(cmap_list[s%len(cmap_list)])
        # everything under vmin gets transparent (all zeros in mask)
        cmap.set_under('k', alpha=1)
        #everything else visible
        cmap.set_over('k', alpha=0)
        # plot mask and center
        plt.imshow(shell_masks[-s], cmap = cmap ,  origin="upper", alpha= 0.2 ) 
    plt.tight_layout(); plt.axis("off");plt.savefig(path_png, dpi=dpi, bbox_inches='tight', pad_inches=0)
    return fig
    
 # Distance analysis
def plot_distance(results_distance,path_png,string_plot = "Orientation (individual)", ylabel=None, dpi=200, ylim=None):
    if not ylabel:
         ylabel= string_plot
    fig = plt.figure()
    plt.plot(results_distance['Shell_mid (µm)'],results_distance[string_plot],"o-", c="lightgreen",label="Orientation")
    plt.grid(); plt.xlabel("Distance (µm)"); plt.ylabel(ylabel)
    plt.tight_layout();plt.savefig(path_png, dpi=dpi)
    if ylim:
        plt.ylim(ylim)
    return fig
           
def plot_cell(im_cell_n,  path_png,scale=None,dpi=200):
        fig = plt.figure(); plt.imshow(im_cell_n);plt.axis("off"),plt.tight_layout()
        if scale is not None:
            scalebar = ScaleBar(scale, "um", length_fraction=0.1, location="lower right", box_alpha=0 , 
                        color="k")
        plt.gca().add_artist(scalebar)
        plt.savefig(path_png, dpi=dpi, bbox_inches='tight', pad_inches=0)
        return fig