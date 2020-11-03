import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib

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


def quiver_plot()

