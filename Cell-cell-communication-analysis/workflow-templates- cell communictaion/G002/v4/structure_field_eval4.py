from fibre_orientation_structure_tensor import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
import pandas as pd
import imageio
import matplotlib
from skimage.filters import gaussian
from scipy.ndimage.filters import uniform_filter
from copy import copy


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




# load in pos and cell image
pos = np.load(r"../cell_positions//markers.npy") #[:4]
cell_img = normalize(imageio.imread(r"..//C001_T001.tif"),0.5,99.5)
#fiber_img = (imageio.imread(r"C003_T017.tif"))
fiber_img = (imageio.imread(r"..//C003_T001.tif"))
# structure data             
min_evec =  np.load(r"field4/min_evec.npy", allow_pickle = True)    
ori =  np.load(r"field4/ori.npy", allow_pickle = True)
# load in results
connections_all = np.load(r"field4/connections_all.npy", allow_pickle = True)
distances_px_all = np.load(r"field4/distances_px_all.npy", allow_pickle = True)
angles_connect_all = np.load(r"field4/angles_connect_all.npy", allow_pickle = True)
angles_shell_all = np.load(r"field4/angles_shell_all.npy", allow_pickle = True)
angles_ratio_all = np.load(r"field4/angles_ratio_all.npy", allow_pickle = True)
int_connect_all = np.load(r"field4/int_connect_all.npy", allow_pickle = True)
int_shell_all = np.load(r"field4/int_shell_all.npy", allow_pickle = True)
int_ratio_all = np.load(r"field4/int_ratio_all.npy", allow_pickle = True)
cos_connect_all = np.load(r"field4/cos_connect_all.npy", allow_pickle = True)
cos_shell_all = np.load(r"field4/cos_shell_all.npy", allow_pickle = True)
cos_ratio_all = np.load(r"field4/cos_ratio_all.npy", allow_pickle = True)
ori_connect_all = np.load(r"field4/ori_connect_all.npy", allow_pickle = True)
ori_shell_all = np.load(r"field4/ori_shell_all.npy", allow_pickle = True)
ori_ratio_all = np.load(r"field4/ori_ratio_all.npy", allow_pickle = True)
cell_con_all = np.load(r"field4/cell_con_all.npy", allow_pickle = True)
diff_angle_all = np.load(r"field4/diff_angle_all.npy", allow_pickle = True)
diff_cos_all = np.load(r"field4/diff_cos_all.npy", allow_pickle = True)





# create distance mask
scale = 0.398
dist_thres_um = 800
dist_thres = dist_thres_um/scale     #
mask=np.ravel([distances_px_all<dist_thres])




"""
Visualize some individual plots for the corresponding FoV   (results are combined lateron in other scripts)
"""

# plot angle over distance
figx = plt.figure(figsize=(3,2))
# plt.title()
plt.scatter(distances_px_all[:][mask] * scale,  angles_connect_all[:,1][mask],s =6, c="C0", label="connection")
plt.scatter(distances_px_all[:][mask] * scale,  angles_shell_all[:,1][mask],s =6, c="C1", label="shell")
# coef = np.polyfit(np.ravel(distances_px_all[:][mask]* scale),  np.ravel(angles_all[:,0][mask]),1)
# poly1d_fn = np.poly1d(coef) 
# plt.plot(np.ravel(distances_px_all[:][mask]* scale), poly1d_fn(np.ravel(distances_px_all[:][mask])), "black",linestyle="-", linewidth=0.7,label= "m = {}".format(np.round(coef[0],3), dist_thres_um))  # (d<={} μm)
plt.legend(fontsize=8)
plt.grid()
# plt.ylim(-1,1)
plt.xlabel("Distance (μm)",fontsize=12)
plt.ylabel("Angle (°)",fontsize=12)
plt.tight_layout()
plt.savefig(r"field4/angle_distance_{}.png".format(dist_thres_um), dpi=500)

# plot angle over distance
figx = plt.figure(figsize=(3,2))
# plt.title()
plt.scatter(distances_px_all[:][mask] * scale,  cos_connect_all[:,1][mask],s =6, c="C0", label="connection")
plt.scatter(distances_px_all[:][mask] * scale,  cos_shell_all[:,1][mask],s =6, c="C1", label="shell")
# coef = np.polyfit(np.ravel(distances_px_all[:][mask]* scale),  np.ravel(angles_all[:,0][mask]),1)
# poly1d_fn = np.poly1d(coef) 
# plt.plot(np.ravel(distances_px_all[:][mask]* scale), poly1d_fn(np.ravel(distances_px_all[:][mask])), "black",linestyle="-", linewidth=0.7,label= "m = {}".format(np.round(coef[0],3), dist_thres_um))  # (d<={} μm)
plt.legend(fontsize=8)
plt.grid()
# plt.ylim(-1,1)
plt.xlabel("Distance (μm)",fontsize=12)
plt.ylabel("2 cos(a) - 1",fontsize=12)
plt.tight_layout()
plt.savefig(r"field4/cos_distance_{}.png".format(dist_thres_um), dpi=500)


# plot angle over distance
figx = plt.figure(figsize=(3,2))
# plt.title()
plt.scatter(distances_px_all[:][mask] * scale,  ori_connect_all[mask],s =6, c="C0", label="connection")
plt.scatter(distances_px_all[:][mask] * scale,  ori_shell_all[mask],s =6, c="C1", label="shell")
plt.legend(fontsize=8)
plt.grid()
plt.xlabel("Distance (μm)",fontsize=12)
plt.ylabel("Ori",fontsize=12)
plt.tight_layout()
plt.savefig(r"field4/ori_distance_{}.png".format(dist_thres_um), dpi=500)

# plot angle over distance
figx = plt.figure(figsize=(3,2))
# plt.title()
plt.scatter(distances_px_all[:][mask] * scale,  int_connect_all[mask],s =6, c="C0", label="connection")
plt.scatter(distances_px_all[:][mask] * scale,  int_shell_all[mask],s =6, c="C1", label="shell")
plt.legend(fontsize=8)
plt.grid()
plt.xlabel("Distance (μm)",fontsize=12)
plt.ylabel("Mean Int.",fontsize=12)
plt.tight_layout()
plt.savefig(r"field4/int_distance_{}.png".format(dist_thres_um), dpi=500)

# plot ratio over distance
figx = plt.figure(figsize=(3,2))
plt.scatter(distances_px_all[:][mask] * scale,  cos_ratio_all[:,1][mask],s =6, c="r", label=np.round(np.mean(cos_ratio_all[:,1][mask]),2))
plt.legend(fontsize=8)
# plt.hlines(np.mean( angles_ratio_all[:,1][mask]),0,300)
plt.grid()
# plt.ylim(0,1)
plt.xlabel("Distance (μm)",fontsize=12)
plt.ylabel("ratio",fontsize=12)
plt.tight_layout()
plt.savefig(r"field4/cos_ratio_all{}.png".format(dist_thres_um))

# plot ratio over distance
figx = plt.figure(figsize=(3,2))
plt.scatter(distances_px_all[:][mask] * scale,  int_ratio_all[mask],s =6, c="r", label=np.round(np.mean(int_ratio_all[mask]),2))
plt.legend(fontsize=8)
plt.grid()
# plt.ylim(0,1)
plt.xlabel("Distance (μm)",fontsize=12)
plt.ylabel("ratio",fontsize=12)
plt.tight_layout()
plt.savefig(r"field4/int_ratio_all{}.png".format(dist_thres_um))



# plot angle diff over distance
figx = plt.figure(figsize=(3,2))
plt.scatter(distances_px_all[:][mask] * scale,  diff_angle_all[:,1][mask],s =6, c="r")
plt.legend(fontsize=8)
# plt.hlines(np.mean( angles_ratio_all[:,1][mask]),0,300)
plt.grid()
plt.title("angle shell - angle connection")
plt.xlabel("Distance (μm)",fontsize=12)
plt.ylabel("Angle Diff.",fontsize=12)
plt.tight_layout()
plt.savefig(r"field4/anglediff{}.png".format(dist_thres_um))


# plot angle diff over distance
figx = plt.figure(figsize=(3,2))
plt.scatter(distances_px_all[:][mask] * scale,  diff_cos_all[:,1][mask],s =6, c="r")
plt.legend(fontsize=8)
# plt.hlines(np.mean( angles_ratio_all[:,1][mask]),0,300)
plt.grid()
plt.title("cos connection - cos shell")
plt.xlabel("Distance (μm)",fontsize=12)
plt.ylabel("Cos Diff.",fontsize=12)
plt.tight_layout()
plt.savefig(r"field4/cosdiff{}.png".format(dist_thres_um))



# plot int diff over distance
figx = plt.figure(figsize=(3,2))
plt.scatter(distances_px_all[:][mask] * scale,  int_connect_all[mask]- int_shell_all[mask],s =6, c="r")
plt.legend(fontsize=8)
# plt.hlines(np.mean( angles_ratio_all[:,1][mask]),0,300)
plt.grid()
plt.title("int connection - int shell")
plt.xlabel("Distance (μm)",fontsize=12)
plt.ylabel("Cos Diff.",fontsize=12)
plt.tight_layout()
plt.savefig(r"field4/intdiff{}.png".format(dist_thres_um))







# plot connection graph with angles and quiver
fig1 = plt.subplots()
# plot overlay
my_norm = matplotlib.colors.Normalize(vmin=0.99, vmax=1, clip=False)  
cmap =  copy(plt.get_cmap('Greys'))
# everything under vmin gets transparent (all zeros in mask)
cmap.set_under('k', alpha=0)
#everything else visible
cmap.set_over('k', alpha=1)
# plot mask and center
show_quiver (min_evec[:,:,0] * ori, min_evec[:,:,1] * ori, filter=[0, 11],alpha=0 , scale_ratio=0.1,width=0.0005, plot_cbar=False, cbar_str="coherency", cmap="viridis")
plt.imshow(normalize(fiber_img), origin="upper", cmap="cividis")   #Greys_r
plt.plot(pos[:, 0], pos[:, 1], 'C1o', ms=5) 
for na,ang in enumerate(angles_connect_all):  
    #only show if distance is small enough
    if distances_px_all[na] <= dist_thres:
        # plot text in middle of line
        plt.text(np.mean(connections_all[na][0]),np.mean(connections_all[na][1]),"{}".format(np.round(cos_ratio_all[:,1][na],2)),  fontsize=8, color = "w")
        # plot conenction line
        plt.plot(connections_all[na][0],connections_all[na][1],"C1--", linewidth=0.7) 
plt.title("cos ratio")
plt.tight_layout()
plt.savefig(r"field4/network_cosratio_{}.png".format(dist_thres_um), dpi=500)


# plot connection graph with angles and quiver
fig1 = plt.subplots()
# plot overlay
my_norm = matplotlib.colors.Normalize(vmin=0.99, vmax=1, clip=False)  
cmap =  copy(plt.get_cmap('Greys'))
# everything under vmin gets transparent (all zeros in mask)
cmap.set_under('k', alpha=0)
#everything else visible
cmap.set_over('k', alpha=1)
# plot mask and center
show_quiver (min_evec[:,:,0] * ori, min_evec[:,:,1] * ori, filter=[0, 11],alpha=0 , scale_ratio=0.1,width=0.0005, plot_cbar=False, cbar_str="coherency", cmap="viridis")
plt.imshow(normalize(fiber_img), origin="upper", cmap="cividis")   #Greys_r
plt.plot(pos[:, 0], pos[:, 1], 'C1o', ms=5) 
for na,ang in enumerate(angles_connect_all):  
    #only show if distance is small enough
    if distances_px_all[na] <= dist_thres:
        # plot text in middle of line
        plt.text(np.mean(connections_all[na][0]),np.mean(connections_all[na][1]),"{}".format(np.round(diff_angle_all[na][1],2)),  fontsize=8, color = "w")
        # plot conenction line
        plt.plot(connections_all[na][0],connections_all[na][1],"C1--", linewidth=0.7) 
plt.title("angle shell - angle connection")
plt.tight_layout()
plt.savefig(r"field4/network_diffang_{}.png".format(dist_thres_um), dpi=500)





# plot connection graph with angles and quiver
fig2 = plt.subplots()
# plot overlay
my_norm = matplotlib.colors.Normalize(vmin=0.99, vmax=1, clip=False)  
cmap =  copy(plt.get_cmap('Greys'))
# everything under vmin gets transparent (all zeros in mask)
cmap.set_under('k', alpha=0)
#everything else visible
cmap.set_over('k', alpha=1)
# plot mask and center
show_quiver (min_evec[:,:,0] * ori, min_evec[:,:,1] * ori, filter=[0, 11],alpha=0 , scale_ratio=0.1,width=0.0005, plot_cbar=False, cbar_str="coherency", cmap="viridis")
plt.imshow(normalize(fiber_img), origin="upper", cmap="cividis")   #Greys_r
plt.plot(pos[:, 0], pos[:, 1], 'C1o', ms=5) 
for na,ang in enumerate(angles_connect_all):
    #only show if distance is small enough
    if distances_px_all[na] <= dist_thres:
        # plot text in middle of line
        plt.text(np.mean(connections_all[na][0]),np.mean(connections_all[na][1]),"{}".format(np.round(angles_connect_all[na][1],2)),  fontsize=8, color = "w")
        # plot conenction line
        plt.plot(connections_all[na][0],connections_all[na][1],"C1--", linewidth=0.7) 
plt.title("angles_connect_all")
plt.tight_layout()
plt.savefig(r"field4/network_angleconn_{}.png".format(dist_thres_um), dpi=500)
# plot connection graph with angles and quiver
fig3 = plt.subplots()
# plot overlay
my_norm = matplotlib.colors.Normalize(vmin=0.99, vmax=1, clip=False)  
cmap =  copy(plt.get_cmap('Greys'))
# everything under vmin gets transparent (all zeros in mask)
cmap.set_under('k', alpha=0)
#everything else visible
cmap.set_over('k', alpha=1)
# plot mask and center
show_quiver (min_evec[:,:,0] * ori, min_evec[:,:,1] * ori, filter=[0, 11],alpha=0 , scale_ratio=0.1,width=0.0005, plot_cbar=False, cbar_str="coherency", cmap="viridis")
plt.imshow(normalize(fiber_img), origin="upper", cmap="cividis")   #Greys_r
plt.title("angles_shell_all")
plt.plot(pos[:, 0], pos[:, 1], 'C1o', ms=5) 
for na,ang in enumerate(angles_connect_all):
    #only show if distance is small enough
    if distances_px_all[na] <= dist_thres:
        # plot text in middle of line
        plt.text(np.mean(connections_all[na][0]),np.mean(connections_all[na][1]),"{}".format(np.round(angles_shell_all[na][1],2)),  fontsize=8, color = "w")
        # plot conenction line
        plt.plot(connections_all[na][0],connections_all[na][1],"C1--", linewidth=0.7) 
plt.tight_layout()
plt.savefig(r"field4/network_angleshell_{}.png".format(dist_thres_um), dpi=500)



# plot connection graph with angles and quiver
fig2 = plt.subplots()
# plot overlay
my_norm = matplotlib.colors.Normalize(vmin=0.99, vmax=1, clip=False)  
cmap =  copy(plt.get_cmap('Greys'))
# everything under vmin gets transparent (all zeros in mask)
cmap.set_under('k', alpha=0)
#everything else visible
cmap.set_over('k', alpha=1)
# plot mask and center
show_quiver (min_evec[:,:,0] * ori, min_evec[:,:,1] * ori, filter=[0, 11],alpha=0 , scale_ratio=0.1,width=0.0005, plot_cbar=False, cbar_str="coherency", cmap="viridis")
plt.imshow(normalize(fiber_img), origin="upper", cmap="cividis")   #Greys_r
plt.plot(pos[:, 0], pos[:, 1], 'C1o', ms=5) 
for na,ang in enumerate(angles_connect_all):
    #only show if distance is small enough
    if distances_px_all[na] <= dist_thres:
        # plot text in middle of line
        plt.text(np.mean(connections_all[na][0]),np.mean(connections_all[na][1]),"{}".format(np.round(cos_connect_all[na][1],2)),  fontsize=8, color = "w")
        # plot conenction line
        plt.plot(connections_all[na][0],connections_all[na][1],"C1--", linewidth=0.7) 
plt.title("cos_connect_all")
plt.tight_layout()
plt.savefig(r"field4/network_cosconn_{}.png".format(dist_thres_um), dpi=500)
# plot connection graph with angles and quiver
fig2 = plt.subplots()
# plot overlay
my_norm = matplotlib.colors.Normalize(vmin=0.99, vmax=1, clip=False)  
cmap =  copy(plt.get_cmap('Greys'))
# everything under vmin gets transparent (all zeros in mask)
cmap.set_under('k', alpha=0)
#everything else visible
cmap.set_over('k', alpha=1)
# plot mask and center
show_quiver (min_evec[:,:,0] * ori, min_evec[:,:,1] * ori, filter=[0, 11],alpha=0 , scale_ratio=0.1,width=0.0005, plot_cbar=False, cbar_str="coherency", cmap="viridis")
plt.imshow(normalize(fiber_img), origin="upper", cmap="cividis")   #Greys_r
plt.plot(pos[:, 0], pos[:, 1], 'C1o', ms=5) 
for na,ang in enumerate(angles_connect_all):
    #only show if distance is small enough
    if distances_px_all[na] <= dist_thres:
        # plot text in middle of line
        plt.text(np.mean(connections_all[na][0]),np.mean(connections_all[na][1]),"{}".format(np.round(cos_shell_all[na][1],2)),  fontsize=8, color = "w")
        # plot conenction line
        plt.plot(connections_all[na][0],connections_all[na][1],"C1--", linewidth=0.7) 
plt.title("cos_shell_all")
plt.tight_layout()
plt.savefig(r"field4/network_cosshell_{}.png".format(dist_thres_um), dpi=500)

# plot connection graph with angles and quiver
fig2 = plt.subplots()
# plot overlay
my_norm = matplotlib.colors.Normalize(vmin=0.99, vmax=1, clip=False)  
cmap =  copy(plt.get_cmap('Greys'))
# everything under vmin gets transparent (all zeros in mask)
cmap.set_under('k', alpha=0)
#everything else visible
cmap.set_over('k', alpha=1)
# plot mask and center
show_quiver (min_evec[:,:,0] * ori, min_evec[:,:,1] * ori, filter=[0, 11],alpha=0 , scale_ratio=0.1,width=0.0005, plot_cbar=False, cbar_str="coherency", cmap="viridis")
plt.imshow(normalize(fiber_img), origin="upper", cmap="cividis")   #Greys_r
plt.plot(pos[:, 0], pos[:, 1], 'C1o', ms=5) 
for na,ang in enumerate(angles_connect_all):
    #only show if distance is small enough
    if distances_px_all[na] <= dist_thres:
        # plot text in middle of line
        plt.text(np.mean(connections_all[na][0]),np.mean(connections_all[na][1]),"{}".format(np.round(diff_cos_all[na][1],2)),  fontsize=8, color = "w")
        # plot conenction line
        plt.plot(connections_all[na][0],connections_all[na][1],"C1--", linewidth=0.7) 
plt.title("cos connection - cos shell")
plt.tight_layout()
plt.savefig(r"field4/network_diffcos_{}.png".format(dist_thres_um), dpi=500)





























